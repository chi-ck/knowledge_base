from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser

from app.infra.llm.providers import llm_provider
from app.infra.persistence.history_repository import history_repository
from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.process.query.agent.state import QueryGraphState
from app.rag.import_.item_name_service import apply_item_name
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import step_log, logger

# ====================== 全局配置 ======================
# 拉取历史消息最大条数
QUERY_HISTORY_LIMIT = 10
# 主体名称确认阈值：高于该分数 → 直接确认
ITEM_NAME_CONFIRM_THRESHOLD = 0.65
# 主体名称候选阈值：介于两者之间 → 让用户选择
ITEM_NAME_CANDIDATE_THRESHOLD = 0.50
# 给用户选择时，最多展示几个候选
ITEM_NAME_OPTIONS_TOPK = 2

# ====================== 步骤1：参数校验 ======================
@step_log("validate_query_identity")
def validate_query_identity(state: QueryGraphState)->tuple[str,str]:
    """
    校验查询状态中是否包含主体确认所需的核心字段。
    Args:
        state: 查询图当前状态，必须包含 original_query（用户问题）和 session_id（会话ID）
    Returns:
        tuple[str, str]: (原始问题, 会话ID)
    """
    original_query = state.get("original_query")
    session_id = state.get("session_id")

    # 必须校验：缺少任意一个都无法继续流程
    if not original_query or not session_id:
        logger.error("session_id和original_query不能为空")
        raise ValueError("session_id和original_query不能为空")

    return original_query, session_id



# ====================== 步骤2：加载历史对话 ======================
@step_log("load_history")
def load_history(session_id:str)->list[dict]:
    """
    从MongoDB加载当前会话的最近聊天记录
    用于解决代词、简称、上下文依赖问题
    """
    return history_repository.list_recent(session_id,limit=QUERY_HISTORY_LIMIT)


# ====================== 步骤3：拼接历史对话文本 ======================
@step_log("build_history_text")
def build_history_text(history_messages: list[dict]) -> str:
    """
    把历史消息拼接成大模型能看懂的上下文格式
    包含：角色、内容、关联的产品主体
    """
    lines:list[str]=[]
    for msg in history_messages:
        # 用户消息使用改写后的query，助手消息使用原始text
        content=msg.get("rewritten_query") if msg.get("role")=="user" else msg.get("text")
        # 拼接识别出的产品名称
        item_names="、".join(msg.get("item_names",[]))
        lines.append(f"角色:{msg.get('role', '')},内容:{content},关联主体: {item_names}")

    return "\n".join(lines)

# ====================== 步骤4：大模型改写问题 + 提取主体名称 ======================
@step_log("rewrite_query_and_extract_item_names")
def rewrite_query_and_extract_item_names(history_messages: list[dict], original_query: str) -> dict:
    """
    调用大模型，同时完成两个核心任务：
    1. 优化/改写用户问题（让检索更准确）
    2. 从问题中提取产品主体名称
    返回格式：{"rewritten_query": "...", "item_names": []}
    """
    # 获取开启JSON模式的大模型客户端
    client=llm_provider.chat(json_mode=True)

    # 加载提示词模板，并传入历史上下文 + 当前问题
    prompt = load_prompt(
        "rewritten_query_and_itemnames",
        history_text=build_history_text(history_messages),
        query=original_query,
    )

    # 构造大模型消息
    messages=[
        SystemMessage(content="你是一个专业的客服助手，擅长理解用户意图和提取关键信息。"),
        HumanMessage(content=prompt),
    ]

    # 调用模型 + JSON解析
    result=(client|JsonOutputParser()).invoke(messages)

    # 兜底：模型没返回改写query → 使用原始问题
    if "rewritten_query" not in result:
        logger.warning(f"模型重写问题失败,给rewritten_query赋予原始问题:{original_query}")
        result["rewritten_query"] = original_query

    # 兜底：模型没识别出商品 → 设为空列表
    if "item_names" not in result:
        logger.warning("模型识别商品失败,给item_names赋予空列表")
        result["item_names"] = []

    return result


# ====================== 步骤5：向量检索匹配标准主体名 ======================
@step_log("search_item_name_candidates")
def search_item_name_candidates(item_names: list[str]) -> dict[str, list[dict]]:
    """
    将模型提取的名称 → 向量化 → 在Milvus向量库检索标准产品名
    返回：每个提取名称对应的【标准名称+匹配分数】
    """
    vector_dict: dict[str, list[dict]] = {}

    # 批量生成向量（稠密+稀疏）
    item_name_vectors=llm_provider.embed_documents(item_names)

    # 逐个名称做混合检索
    for index,item_name in enumerate(item_names):
        dense_vector=item_name_vectors["dense"][index]
        sparse_vector=item_name_vectors["sparse"][index]

        # 构造检索请求
        reqs=milvus_gateway.create_requests(dense_vector,sparse_vector)

        # 执行混合检索
        response=milvus_gateway.hybrid_search(
            collection_name=milvus_gateway.item_name_collection,
            reqs=reqs,
            ranker_weights=(0.5,0.5),
            norm_score=True,
            output_fields=["item_name"],
        )

        # 整理检索结果：标准产品名 + 分数
        current_item_name_list:list[dict]=[]
        for item in (response[0] if response else []):
            current_item_name_list.append({
                "item_name": item.get("entity", {}).get("item_name", ""),
                "score": item.get("distance", 0),
            })

        vector_dict[item_name]=current_item_name_list

    return vector_dict




# ====================== 步骤6：根据分数筛选确认/候选/未找到 ======================
@step_log("select_item_names")
def select_item_names(vector_dict: dict[str, list[dict]]) -> dict:
    """
    根据向量检索分数，将产品名分为三类：
    1. 高分 → 直接确认
    2. 中分 → 加入候选，让用户选择
    3. 低分 → 未找到
    """
    confirmed_item_name_list:list[str]=[]
    options_item_name_list: list[str] = []

    for _,item_name_list in vector_dict.items():
        # 按分数从高到低排序
        item_name_list.sort(key=lambda x: x["score"], reverse=True)

        # 高分：直接确认
        high_list=[item for item in item_name_list if item["score"]>=ITEM_NAME_CONFIRM_THRESHOLD]

        # 中分：候选列表
        low_list=[item for item in item_name_list
                  if ITEM_NAME_CANDIDATE_THRESHOLD<=item["score"]<ITEM_NAME_CONFIRM_THRESHOLD
                  ]

        if high_list:
            # 取最高分作为确认结果
            confirmed_item_name_list.append(high_list[0]["item_name"])


        if low_list:
            # 取前N个作为候选给用户选择
            options_item_name_list.extend([item["item_name"] for item in low_list[:ITEM_NAME_OPTIONS_TOPK]])

    return {
        "confirmed_item_name_list": confirmed_item_name_list,
        "options_item_name_list": options_item_name_list,
    }

# ====================== 步骤7：将结果写入state ======================
@step_log("apply_item_name_result")
def apply_item_name_result(state: dict, final_result: dict, rewritten_query: str) -> None:
    """
    根据主体确认结果，更新state：
    1. 有确认产品 → 写入item_names、rewritten_query
    2. 有候选产品 → 返回选择问句
    3. 无产品 → 返回未找到提示
    """
    confirmed = final_result.get("confirmed_item_name_list", [])
    options = final_result.get("options_item_name_list", [])

    if confirmed:
        # 情况1：成功识别 → 进入检索流程
        state["item_names"] = confirmed
        state["rewritten_query"] = rewritten_query
        # 清除之前的answer，避免干扰流程
        if "answer" in state:
            del state["answer"]
        return

    if options:
        # 情况2：有多个候选 → 反问用户确认
        option_str = "、".join(options)
        state["answer"] = f"您是想问以下哪个产品：{option_str}？请明确一下型号。"
        state["rewritten_query"] = rewritten_query
        state["item_names"] = []
        return

    # 情况3：完全未匹配到产品
    state["answer"] = "抱歉，未找到相关产品，请提供准确型号以便我为您查询。"
    state["rewritten_query"] = rewritten_query
    state["item_names"] = []

# ====================== 步骤8：保存用户消息到历史 ======================
@step_log("save_user_message")
def save_user_message(state: dict) -> None:
    """
    把当前用户提问 + 处理结果（改写query、识别主体）存入MongoDB历史记录
    用于下一轮对话上下文
    """
    history_repository.save_message(
        session_id=state["session_id"],
        role="user",
        text=state["original_query"],
        rewritten_query=state.get("rewritten_query", ""),
        item_names=state.get("item_names", []),
    )

# ====================== 主体确认主入口 ======================
@step_log("confirm_item_name")
def confirm_item_name(state: QueryGraphState) -> QueryGraphState:
    """
    【主体确认节点】主函数
    作用：整理用户问题 → 提取标准产品名 → 决定是继续检索还是反问用户
    是整个检索链的“入口治理”环节
    """
    # 1. 校验必填参数
    original_query,session_id=validate_query_identity(state)

    # 2. 加载历史对话
    history_messages=load_history(session_id)

    # 3. 大模型改写问题 + 提取产品名
    llm_result=rewrite_query_and_extract_item_names(history_messages,original_query)
    item_names=llm_result["item_names"]
    rewritten_query=llm_result["rewritten_query"]

    final_result = {}
    # 4. 有提取到名称 → 去向量库匹配标准名称
    if item_names:
        vector_result=search_item_name_candidates(item_names)
        final_result=select_item_names(vector_result)

    # 5. 把结果写入state
    apply_item_name_result(state,final_result,rewritten_query)

    # 6. 保存历史记录
    save_user_message(state)

    return state