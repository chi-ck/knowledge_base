from pathlib import Path

from app.process.import_.agent.state import ImportGraphState
from app.shared.runtime.logger import step_log, logger


@step_log("resolve_input_file")
def resolve_input_file(state: ImportGraphState) -> ImportGraphState:
    """
    文件类型识别与状态初始化节点（导入流程入口）
    核心功能：根据本地文件路径识别文件类型，自动装配对应状态字段，为后续流程路由提供依据

    业务逻辑：
        1. 校验文件路径是否存在
        2. 根据后缀识别 MD / PDF 文件
        3. 自动填充对应路径、路由开关、文件标题
        4. 不支持的文件类型直接终止流程

    Args:
        state: 导入流程全局状态，必须包含 local_file_path 字段

    Returns:
        ImportGraphState: 补全文件信息后的完整状态对象
    """

    # 1. 获取文件本地路径
    local_file_path=state.get("local_file_path")

    # 2. 校验文件路径是否为空，为空则直接结束流程
    if not local_file_path:
        logger.warning("节点:resolve_input_file, 文件路径为空，直接终止当前导入流程")
        return state

    # 3. 识别文件类型并设置对应状态与路由开关
    if local_file_path.endswith(".md"):
        # Markdown 文件：启用 MD 处理链路，禁用 PDF 处理链路
        state["md_path"] = local_file_path
        state["is_md_read_enabled"] = True
        state["is_pdf_read_enabled"] = False
    elif local_file_path.endswith(".pdf"):
        state["pdf_path"] = local_file_path
        state["is_pdf_read_enabled"] = True
        state["is_md_read_enabled"] = False
    else:
        # 不支持的文件类型，直接终止流程
        logger.warning(
            f"节点:resolve_input_file, 不支持的文件类型: {local_file_path}，终止流程"
        )
        return state

    # 4. 自动提取文件标题（不带后缀）
    state["file_title"]=Path(local_file_path).stem

    # 5. 返回补全后的状态
    return state


