from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """查询请求参数"""

    session_id: str | None = Field(None, description="会话ID")
    query: str = Field(..., description="原始查询")
    is_stream: bool = Field(False, description="是否流式返回")


class AsyncQueryResponse(BaseModel):
    """查询响应参数"""

    session_id: str = Field(..., description="会话ID")
    message: str = Field(..., description="响应信息")


class QueryResponse(BaseModel):
    """查询响应参数"""

    session_id: str = Field(..., description="会话ID")
    message: str = Field(..., description="响应信息")
    answer: str = Field("", description="答案")
    image_urls: list[str]
    done_list: list[str]


class ClearHistoryResponse(BaseModel):
    """清空聊天记录接口响应体"""

    message: str = Field(..., description="操作提示信息")
    deleted_count: int = Field(..., description="成功删除的消息条数")


class HistoryItem(BaseModel):
    id: str = Field("", description="消息ID")
    session_id: str = ""
    role: str = ""
    text: str = ""
    rewritten_query: str = ""

    item_names: list[str]
    image_urls: list[str]

    ts: Any = None


class HistoryResponse(BaseModel):
    session_id: str
    items: list[HistoryItem]