from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

from app.infra.llm.providers import llm_provider
from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import step_log, logger

# ====================== 重排全局配置 ======================
RERANK_MAX_TOPK: int = 10                # 动态截断最多保留10条结果
RERANK_MIN_TOPK: int = 1                 # 动态截断最少保留1条结果
RERANK_GAP_RATIO: float = 2              # 分数断崖比例阈值（用于动态截断）
RERANK_GAP_ABS: float = 2                # 分数断崖绝对值阈值
RERANK_MAX_INPUT_TOKENS: int = 512       # 重排模型最大输入token长度
RERANK_SUMMARY_CHAR_RATIO: float = 1.3   # 中文token与字符换算比例 1token≈1.3字符
RERANK_MIN_SUMMARY_CHARS: int = 50       # 文本精简后最小字符数


@step_log("validate_rerank_inputs")
def validate_rerank_inputs(state: dict) -> tuple[list[dict], list[dict]]:
    """
    校验重排节点输入是否合法
    允许本地结果为空 或 联网结果为空，但不允许两者都为空
    """
    rrf_chunks = state.get("rrf_chunks", [])
    web_search_docs = state.get("web_search_docs", [])

    # 必须至少一路有数据，否则 rerank 无内容可处理
    if not rrf_chunks and not web_search_docs:
        logger.error("rrf_chunks 和 web_search_docs 均为空，rerank 重排无有效数据！")
        raise ValueError("rerank 重排失败：本地融合结果与联网搜索结果均为空")

    return rrf_chunks, web_search_docs

@step_log("merge_rrf_and_web")
def merge_rrf_and_web(rrf_chunks: list[dict], web_search_docs: list[dict]) -> list[dict]:
    """
    统一合并本地知识库结果 + 联网搜索结果
    统一字段格式，方便后续重排模型统一打分
    """
    final_chunk_list: list[dict] = []

    # 处理本地RRF融合结果
    for chunk in rrf_chunks or []:
        final_chunk_list.append({
            "title": chunk.get("title"),
            "text": chunk.get("content"),    # 本地知识库取content字段
            "url": None,                     # 本地数据无URL
            "type": chunk.get("type", "milvus"),
            "score": chunk.get("score", 0.0),
        })

    # 处理联网搜索网页结果
    for doc in web_search_docs or []:
        final_chunk_list.append({
            "title": doc.get("title"),
            "text": doc.get("snippet"),      # 网页结果取摘要snippet
            "url": doc.get("url"),           # 网页保留URL
            "type": "web",
            "score": 0.0,
        })

    return final_chunk_list

@step_log("summarize_long_rerank_text")
def summarize_long_rerank_text(question: str, answer: str, limit: int) -> str:
    """
    超长文本精简：当文本超过重排模型最大输入长度时，调用LLM精简内容
    保证输入长度合规，同时保留与问题相关的核心信息
    """
    prompt = load_prompt(
        "rerank_text_refine",
        question=question,
        answer=answer,
        limit=limit,
    )
    messages = [
        SystemMessage(content="你现在是文本精简提炼专家。根据用户发送的文本完成文本精炼要求。"),
        HumanMessage(content=prompt),
    ]
    # 调用大模型精简文本并返回
    refined_answer = (llm_provider.chat() | StrOutputParser()).invoke(messages)
    return refined_answer

@step_log("build_question_pairs")
def build_question_pairs(question: str, final_chunk_list: list[dict], reranker) -> list[list[str]]:
    """
    构建重排模型输入对：[问题, 文本]
    自动处理超长文本：超过模型最大输入则进行精简
    """
    tokenizer=reranker.tokenizer
    # 对问题进行token编码（不添加特殊符号）
    query_tokens = tokenizer.encode(question, add_special_tokens=False)
    question_pairs: list[list[str]] = []

    for item in final_chunk_list:
        answer = item.get("text") or ""
        answer_for_rerank = answer

        # 计算答案文本的token长度
        answer_tokens = tokenizer.encode(answer, add_special_tokens=False)
        logger.info(f"答案token长度:{len(answer_tokens)}")
        # 重排模型固定格式
        total_tokens = len(query_tokens) + len(answer_tokens) + 4

        # 超过最大输入长度 → 调用LLM精简文本
        if total_tokens > RERANK_MAX_INPUT_TOKENS:
            # 计算允许保留的最大字符数（token → 字符换算）
            limit=max(RERANK_MIN_SUMMARY_CHARS,int((RERANK_MAX_INPUT_TOKENS-len(query_tokens)-4)/RERANK_SUMMARY_CHAR_RATIO))
            # 精简超长文本
            answer_for_rerank = summarize_long_rerank_text(question=question, answer=answer, limit=limit)

        # 组装 [问题, 处理后答案] 对
        question_pairs.append([question,answer_for_rerank])

    return question_pairs

@step_log("score_and_sort_chunks")
def score_and_sort_chunks(state: dict, final_chunk_list: list[dict]) -> list[dict]:
    """
    调用重排模型对所有候选文档打分，并按分数从高到低排序
    """
    if not final_chunk_list:
        return []

    # 获取用户查询问题
    rewritten_query = state.get("rewritten_query") or state.get("original_query") or ""
    # 获取重排模型实例
    reranker = llm_provider.reranker_model()
    # 构建模型输入对
    question_pairs = build_question_pairs(rewritten_query, final_chunk_list, reranker)
    # 模型打分（归一化）
    score_list = reranker.compute_score(question_pairs, normalize=True)

    # 将分数写入文档
    for score, chunk in zip(score_list, final_chunk_list):
        chunk["score"] = round(score, 4)

    # 按分数降序排序
    final_chunk_list.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return final_chunk_list


@step_log("dynamic_topk")
def dynamic_topk(chunk_list_score_sorted: list[dict]) -> list[dict]:
    """
    动态值截断：根据分数断崖自动决定保留多少条结果
    不是固定取前N条，而是找到分数突变的位置截断
    """
    min_topk = RERANK_MIN_TOPK
    max_topk = min(RERANK_MAX_TOPK, len(chunk_list_score_sorted))
    gap_ratio = RERANK_GAP_RATIO
    max_gap = RERANK_GAP_ABS
    topk = max_topk  # 默认取最大条数

    # 遍历寻找分数断崖
    if topk > min_topk:
        for index in range(min_topk-1, max_topk-1):
            score_1 = chunk_list_score_sorted[index].get("score", 0.0)
            score_2 = chunk_list_score_sorted[index + 1].get("score", 0.0)
            abs_score = score_1 - score_2  # 分数差
            ratio_score = abs_score / (score_1 + 1e-7)  # 比例差

            if abs_score > max_gap or ratio_score>gap_ratio:
                topk=index+1
                break

    # 返回截断后的结果
    return chunk_list_score_sorted[:topk]

@step_log("rerank_documents")
def rerank_documents(state: QueryGraphState) -> list[dict]:
    """
    重排节点主入口
    流程：校验输入 → 合并本地+网页 → 模型打分排序 → 动态截断
    输出最终高质量候选文档列表
    """
    # 1. 校验输入
    rrf_chunks, web_search_docs = validate_rerank_inputs(state)
    # 2. 统一格式合并两路结果
    merged = merge_rrf_and_web(rrf_chunks, web_search_docs)
    # 3. 重排模型打分 + 排序
    sorted_docs = score_and_sort_chunks(state, merged)
    # 4. 动态截断，返回最优结果
    return dynamic_topk(sorted_docs)