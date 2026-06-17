from app.infra.llm.providers import llm_provider
from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.logger import step_log, logger

# ====================== 检索配置 ======================
# 默认返回的最大知识库片段数量
RETRIEVAL_DEFAULT_LIMIT = 5
# 混合检索权重：dense向量权重 0.9，sparse向量权重 0.1
RETRIEVAL_RANKER_WEIGHTS = (0.9, 0.1)

@step_log("validate_retrieval_state")
def validate_retrieval_state(state: dict) -> tuple[list[str], str]:
    """
    校验检索必须的核心参数是否存在
    必须包含：item_names（主体名称）、rewritten_query（改写后的查询语句）
    """
    item_names = state.get("item_names")
    rewritten_query = state.get("rewritten_query")

    # 缺少任意一个都无法执行检索
    if not item_names or not rewritten_query:
        logger.error("item_names或rewritten_query不存在,无法继续业务!")
        raise ValueError("item_names或rewritten_query不存在,无法继续业务!")

    return item_names, rewritten_query

@step_log("build_item_name_expr")
def build_item_name_expr(item_names: list[str]) -> str:
    """
    构造 Milvus 过滤表达式
    作用：只检索属于当前 item_names 的知识库片段，避免跨主体干扰
    """
    return f"item_name in {item_names}"

@step_log("search_chunks")
def search_chunks(
    *,
    rewritten_query: str,
    item_names: list[str],
    limit: int = RETRIEVAL_DEFAULT_LIMIT,
) -> list[dict]:
    """
    基于改写后的查询，执行【带主体过滤】的混合向量检索
    这是本地知识库的核心检索逻辑

    Args:
        rewritten_query: 优化后的查询语句
        item_names: 已确认的主体列表，用于过滤检索范围
        limit: 最多返回几条片段

    Returns:
        标准化后的知识库片段列表
    """
    # 1. 将改写后的查询生成稠密向量 + 稀疏向量（和知识库入库时保持一致）
    embedding_result=llm_provider.embed_documents([rewritten_query])
    dense_vector= embedding_result["dense"][0]
    sparse_vector =embedding_result["sparse"][0]

    # 2. 构造混合检索请求：带主体过滤，只查当前确认的产品
    reqs=milvus_gateway.create_requests(
        dense_vector,
        sparse_vector,
        expr=build_item_name_expr(item_names),
        limit=limit,
    )

    @step_log("normalize_retrieved_chunk")
    def normalize_retrieved_chunk(chunk: dict) -> dict:
        """
        将 Milvus 原始检索结果，统一格式化为查询链内部标准结构
        确保后续所有节点使用相同的数据结构
        """
        entity = chunk.get("entity", chunk)
        return {
            "chunk_id": chunk.get("id") or entity.get("chunk_id"),  # 片段ID
            "item_name": entity.get("item_name", ""),  # 归属主体名称
            "title": entity.get("title"),  # 片段标题
            "parent_title": entity.get("parent_title"),  # 父标题/章节
            "part": entity.get("part"),  # 部分标识
            "file_title": entity.get("file_title"),  # 来源文件标题
            "content": entity.get("content", ""),  # 片段文本内容
            "score": chunk.get("distance", 0.0),  # 相似度分数
            "type": "milvus",  # 来源类型（向量库）
            "url": None,  # 附件URL（无）
        }

    # 3. 执行混合检索（dense + sparse 加权融合）
    resp=milvus_gateway.hybrid_search(
        collection_name=milvus_gateway.chunks_collection,
        reqs=reqs,
        ranker_weights=RETRIEVAL_RANKER_WEIGHTS,
        norm_score=True,
        limit=limit,
        output_fields=["chunk_id","item_name","content","title","parent_title","part", "file_title"],
    )

    # 4. 把原始结果格式化为统一结构返回
    return [normalize_retrieved_chunk(chunk) for chunk in (resp[0] if resp else [])]

@step_log("search_embedding")
def search_by_embedding(state: QueryGraphState) -> list[dict]:
    """
    【本地知识库向量检索节点】主入口
    作用：根据主体名称 + 改写查询，执行精准的知识库检索
    返回标准化后的检索结果，供后续生成答案使用
    """
    # 校验参数
    item_names, rewritten_query = validate_retrieval_state(state)

    # 执行检索并返回结果
    return search_chunks(
        rewritten_query=rewritten_query,
        item_names=item_names
    )