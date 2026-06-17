from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.logger import step_log, logger

# RRF 融合默认配置
# 最终返回的融合结果数量
top = 5
# RRF 公式平滑系数，避免排名过低导致分数趋近于 0
k = 60


@step_log("validate_rrf_inputs")
def validate_rrf_inputs(state: dict) -> tuple[list[dict], list[dict]]:
    """
    校验 RRF 融合所需的两路本地检索结果是否合法
    允许单路有结果、另一路为空，但不允许两路都为空
    """
    # 获取普通向量检索结果、HyDE 增强检索结果
    embedding_chunks = state.get("embedding_chunks", [])
    hyde_chunks = state.get("hyde_embedding_chunks", [])

    # 核心校验：两路都为空则无法融合，直接抛出异常
    if not embedding_chunks and not hyde_chunks:
        logger.error("embedding_chunks 和 hyde_embedding_chunks 均为空，RRF 融合无有效数据！")
        raise ValueError("RRF 融合失败：本地两路检索结果均为空")

    return embedding_chunks, hyde_chunks


@step_log("reciprocal_rank_fusion")
def reciprocal_rank_fusion(
        param_list: list[tuple[list[dict], float]],
        *,
        k: int = 60,
        top: int = 5,
) -> list[dict]:
    """
    RRF（倒数排名融合）核心算法实现
    不依赖原始分数，只根据排名位置计算融合得分，支持多路加权融合

    Args:
        param_list: 列表，每一项是 (检索结果列表, 权重)
        k: 平滑系数，默认60
        top: 最终返回Top N条结果

    Returns:
        按RRF融合分数倒序排列后的统一结果列表
    """
    # 存储每个 chunk_id 的总融合分数
    score_dict:dict[str, float]={}
    # 存储每个 chunk_id 对应的原始片段信息
    entity_dict:dict[str, dict]={}

    # 遍历每一路检索结果及其权重
    for chunks_list,weight in param_list:
        # 遍历当前路的所有片段，rank 从 1 开始计数（第一名=1）
        for rank,chunk in enumerate(chunks_list,start=1):
            chunk_id=chunk.get("chunk_id")
            if not chunk_id:
                continue  # 无chunk_id则跳过，无法融合

            # RRF 核心公式：1/(k + 排名) * 权重，累加到总分
            score_dict[chunk_id]=score_dict.get(chunk_id,0.0)+(1.0/(k+rank))*weight
            # 保留片段原文信息（只存一次，避免覆盖）
            entity_dict.setdefault(chunk_id,chunk)

    # 组装最终结果：把分数和原文信息合并
    document_list=[]
    for chunk_id,score in score_dict.items():
        document=entity_dict.get(chunk_id, {}).copy()
        # 将计算好的 RRF 总分写入结果
        document["score"]=score
        document_list.append(document)


    # 按融合分数从高到低排序
    document_list.sort(key=lambda x: x.get("score", 0.0), reverse=True)


    # 返回 Top N 条最终融合结果
    return document_list[:top]




@step_log("fuse_by_rrf")
def fuse_by_rrf(state: QueryGraphState) -> list[dict]:
    """
    RRF 融合节点入口
    负责将本地两路检索结果（普通向量 + HyDE）进行加权融合
    两路默认权重均为 1.0
    """
    # 1. 校验两路输入结果是否合法
    embedding_chunks, hyde_embedding_chunks = validate_rrf_inputs(state)

    # 2. 构造 RRF 入参：(结果列表, 权重)
    param_list = [
        (embedding_chunks, 1.0),  # 普通向量检索，权重1.0
        (hyde_embedding_chunks, 1.0),  # HyDE增强检索，权重1.0
    ]

    # 3. 执行 RRF 融合并返回最终结果
    return reciprocal_rank_fusion(param_list)