from app.infra.llm.providers import llm_provider
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import EMBEDDING_BATCH_SIZE
from app.shared.runtime.logger import step_log, logger


@step_log("require_chunks")
def require_chunks(state: ImportGraphState)->list[dict]:
    """
    校验导入状态中是否已经生成切块结果
    功能：确保后续流程有有效的输入数据，缺失时抛出异常
    :param state: LangGraph 流程状态字典
    :return: 已通过校验的切块列表
    """
    # 从 state 中获取核心数据
    chunks=state.get("chunks",[])

    # ===================== 校验 chunks =====================
    # 如果 chunks 为空，无法继续业务，直接抛出异常终止流程
    if not chunks:
        logger.error("chunks为空,无法继续业务处理!")
        raise ValueError("chunks为空,无法继续业务处理!")

    # 返回校验后的数据
    return chunks


def embed_chunks(chunks:list[dict],*,step:int=EMBEDDING_BATCH_SIZE)->list[dict]:
    """
    批量为文本切片生成稠密和稀疏向量
    功能：分批拼接文本 → 调用 Embedding 模型 → 绑定向量字段 → 异常隔离
    :param chunks: 文档切片列表，每个元素包含 item_name、content 等字段
    :param step: 批次大小，默认由配置 EMBEDDING_BATCH_SIZE 控制
    :return: 带向量字段的切片列表
    """
    # 初始化结果列表，存储带有向量的 chunk 数据
    chunks_vector:list[dict]=[]

    # 获取总切片数
    total=len(chunks)

    # ===================== 分批处理 =====================
    # 按批次大小循环处理，每次处理 step 条切片
    for index in range(0,total,step):
        try:
            # 截取当前批次的切片，最后一批自动适配剩余数量
            step_chunks=chunks[index:index+step]
            # ===================== 构造模型输入文本 =====================
            # 拼接格式："主体:{item_name},内容:{content}"
            # 核心词前置原则：Embedding 模型对前 128 个 token 的注意力最集中
            vector_str_list=[]
            for item in step_chunks:
                item_name=item["item_name"]
                content=item["content"]
                # 有主体名称则拼接，无则直接使用内容
                vector_str_list.append(f"主体:{item_name},内容:{content}" if item_name else content)

            # ===================== 调用 Embedding 模型 =====================
            # 调用 llm_provider.embed_documents() 生成批量向量
            # 返回格式：{"dense": [稠密向量列表], "sparse": [稀疏向量列表]}
            result=llm_provider.embed_documents(vector_str_list)

            # ===================== 绑定向量字段 =====================
            # 为当前批次每个切片绑定对应向量，复制原数据避免修改上游源数据
            for i,chunk in enumerate(step_chunks):
                chunk_new =chunk.copy()
                chunk_new["dense_vector"]=result["dense"][i]
                chunk_new["sparse_vector"]=result["sparse"][i]
                chunks_vector.append(chunk_new)

        except Exception as exc:
            # ===================== 异常处理 =====================
            # 捕获异常，记录警告信息并跳过当前批次
            logger.warning(f"index={index}步骤,发生错误,跳过,继续生成向量!!,错误信息:{str(exc)}")
            # 跳过当前批次，继续处理下一批次，保证整体流程不中断
            continue

    # 返回所有带向量的切片列表
    return chunks_vector

@step_log("generate_chunk_embeddings")
def generate_chunk_embeddings(state: ImportGraphState) -> ImportGraphState:
    """
    向量化服务总入口
    功能：校验切块数据 → 批量生成稠密/稀疏向量 → 回写到 state
    输出：更新后的 state，包含带有 dense_vector 和 sparse_vector 的 chunks
    """
    # 先确认 chunks 存在，再批量写回 dense/sparse 向量字段
    state["chunks"]=embed_chunks(require_chunks(state))
    return state