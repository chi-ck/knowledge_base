from app.infra.llm.providers import llm_provider
from app.process.query.agent.state import QueryGraphState
from app.rag.query.embedding_search_service import search_chunks
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import step_log, logger

from langchain.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

@step_log("validate_retrieval_state")
def validate_retrieval_state(state: dict) -> tuple[list[str], str]:
    """
    校验HYDE检索所需核心参数合法性
    确保前置主体确认、问句改写流程已正常执行
    Args:
        state: 查询流程图全局状态
    Returns:
        tuple: 已确认主体列表、改写后的标准查询问句
    """
    item_names = state.get("item_names")
    rewritten_query = state.get("rewritten_query")
    # 主体名称和改写问句为检索必填参数，缺失则无法执行检索
    if not item_names or not rewritten_query:
        logger.error("item_names或rewritten_query不存在,无法继续业务!")
        raise ValueError("item_names或rewritten_query不存在,无法继续业务!")
    return item_names, rewritten_query

@step_log("generate_hyde_answer")
def generate_hyde_answer(rewritten_query: str) -> str:
    """
    生成HYDE假设答案（核心语义增强能力）
    不用于最终回复，仅用于丰富检索语义、优化向量匹配效果
    Args:
        rewritten_query: 改写后的标准用户问句
    Returns:
        str: 大模型生成的场景化、标准化假设回答文本
    """
    # 加载HYDE专属提示词模板，传入优化后问句
    prompt_str = load_prompt("hyde_prompt", rewritten_query=rewritten_query)
    # 构造大模型对话请求
    messages = [HumanMessage(content=prompt_str)]
    # 调用大模型生成假设答案，纯文本输出
    return (llm_provider.chat() | StrOutputParser()).invoke(messages)

@step_log("search_chunks_with_hyde")
def search_chunks_with_hyde(
    *,
    rewritten_query: str,
    item_names: list[str],
    limit: int = 5,
) -> tuple[str, list[dict]]:
    """
    执行HYDE增强检索核心逻辑
    拼接原始改写问句与模型假设答案，构造高语义检索文本，完成知识库召回
    Args:
        rewritten_query: 前置流程优化后的标准查询问句
        item_names: 已确认的业务主体列表，用于检索范围过滤
        limit: 检索返回知识片段最大数量
    Returns:
        tuple: 模型生成的假设答案、HYDE增强检索后的知识片段列表
    """
    # 大模型基于用户问题生成假设性标准答案，扩充弱语义信息
    hyde_answer=generate_hyde_answer(rewritten_query)
    # 拼接原问句与假设答案，构造高语义密度的检索输入
    hybrid_query = f"{rewritten_query},{hyde_answer}"
    # 复用基础检索能力，基于增强后的文本执行Milvus混合检索
    chunks = search_chunks(rewritten_query=hybrid_query, item_names=item_names, limit=limit)
    return chunks


@step_log("search_by_hyde")
def search_by_hyde(state: QueryGraphState) -> QueryGraphState:
    """
    HYDE增强检索节点入口函数
    作为多路检索的补充支路，专门优化短问句、口语化问句漏召回问题
    Args:
        state: 查询流程图全局状态
    Returns:
        list[dict]: 标准化的增强检索知识片段列表
    """
    # 校验检索前置参数完整性
    item_names, rewritten_query = validate_retrieval_state(state)
    # 执行HYDE增强检索，忽略假设答案，仅返回检索片段供后续流程使用
    chunks = search_chunks_with_hyde(rewritten_query=rewritten_query, item_names=item_names)
    return chunks