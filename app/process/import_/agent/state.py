import copy
from typing import TypedDict


class ImportGraphState(TypedDict):
    """
    图的状态定义，包含所有节点产生和消费的数据字段。
    TypedDict 让我们在代码中能有自动补全和类型检查。
    使用字典式访问（如 state["task_id"]、state.get("chunks")）。
    """
    task_id: str

    # --- 流程控制标记 ---
    is_md_read_enabled: bool
    is_pdf_read_enabled: bool

    # --- 路径相关 ---
    local_file_path: str

    pdf_path: str
    local_dir: str

    md_path: str
    file_title: str

    # --- 内容数据 ---
    md_content:str
    chunks:list
    item_name:str

    # --- 数据库相关 ---
    embeddings_content:list


graph_default_state:ImportGraphState = {
    "task_id":"",
    "is_md_read_enabled":False,
    "is_pdf_read_enabled":False,
    "local_file_path":"",
    "pdf_path":"",
    "local_dir":"",
    "md_path":"",
    "file_title":"",
    "md_content":"",
    "chunks":[],
    "item_name":"",
    "embeddings_content":[],
}

def create_default_state(**overrides)-> ImportGraphState:
    """
    创建默认状态，支持覆盖。
    :param overrides:
    :return:
    """
    state=copy.deepcopy(graph_default_state)
    state.update(overrides)
    return state

def get_default_state()-> ImportGraphState:
    """
    返回一个新的状态实例，避免全局变量污染。
    :return:
    """
    return copy.deepcopy(graph_default_state)

