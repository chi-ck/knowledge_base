from mimetypes import guess_type
from pathlib import Path
import sys
import uuid

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from starlette.middleware.cors import CORSMiddleware

from app.api.schemas.query import QueryRequest, AsyncQueryResponse, QueryResponse, HistoryResponse, HistoryItem, \
    ClearHistoryResponse
from app.infra.persistence.history_repository import history_repository
from app.shared.runtime.logger import PROJECT_ROOT, logger
from app.infra.config.providers import settings
from app.process.query.agent.main_graph import query_app as query_graph_app, query_app
from app.process.query.agent.state import create_query_default_state
from app.shared.utils.sse_utils import SSEEvent, create_sse_queue, push_to_session, sse_generator
from app.shared.utils.task_utils import (
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_PROCESSING,
    clear_task,
    get_done_task_list,
    get_task_result,
    update_task_status,
)


# 定义fastapi对象
app=FastAPI(
    title=settings.query_app_name,
    description="描述,进行rag查询的服务对象",
    version="0.2.0"
)

# 跨域处理
app.add_middleware(
    CORSMiddleware,
    allow_origins = ['*'],
    allow_methods = ['*'],
    allow_headers = ['*']
)

@app.get("/html")
def query_html():
    html_path = PROJECT_ROOT / "app" / "process" / "query" / "page" / "chat.html"
    return FileResponse(path=html_path, media_type=guess_type(html_path.name)[0])

def run_query_graph(query:str,session_id:str,is_stream:bool):
    # 一会回调用 main_graph执行
    # 本次任务开启了！ is_stream = True 把结果加入到队列，sse可以取到
    # 清理上一次任务状态，避免缓存污染
    clear_task(session_id)
    update_task_status(session_id,"processing",is_stream)

    state=create_query_default_state(
        session_id=session_id,
        original_query=query,
        is_stream=is_stream
    )
    try:
        query_app.invoke(state)
        # 本次任务开启了！ is_stream = True 把结果加入到队列，sse可以取到
        update_task_status(session_id,"completed",is_stream)
    except Exception as e:
        logger.exception(f"---session_id = {session_id},查询流程出现异常！！{str(e)}")
        # 修改 event = process
        update_task_status(session_id, "failed", is_stream)
        # 推送指定类型的事件
        push_to_session(session_id, SSEEvent.ERROR, {"error": str(e)})
@app.post("/query")  # 客户端 -》 问题 -》 graph开启了 -》 查到rag的结果 -》 返回即可！！
async def query(request:QueryRequest,background_tasks:BackgroundTasks):
    """
    :param request: 请求参数
    :param background_tasks: 异步执行函数  is_stream = True
    :return:
    """
    query=request.query
    session_id=request.session_id or str(uuid.uuid4())
    is_stream=request.is_stream
    # 判断是不是流式处理 （异步 -》 先返回一个结果 开始处理 | 后台运行图，结果向前端推送）
    if is_stream:
        # 只要开启流式处理，我们业务中就是将数据，插入到队列中！ {session_id , queue [update_task_state , add_running_task,add_done_list]}
        # 创建当前session_id对应的队列 =》 _session_stream
        create_sse_queue(session_id)
        # 异步执行  立即返回结果前端 || 中间的过程 sse 一点一点推送给前端
        background_tasks.add_task(run_query_graph,query,session_id,is_stream)
        logger.info(f"query:{query}已经开启了异步和流式处理！！")
        return AsyncQueryResponse(
            session_id=session_id,
            message="本次查询处理中...."
        )
    else:
        # 同步执行
        run_query_graph(query,session_id,is_stream)
        # 获取最后一个节点插入的结果！ node_answer_output (answer)
        answer=get_task_result(session_id,"answer")
        logger.info(f"query:{query}开启同步处理！处理结果为：{answer}!")  # task_utils 封装的一个存储会话结果函数
        # 返回对应的json数据即可
        return QueryResponse(
            answer=answer,
            session_id=session_id,
            message="本次查询完毕!",
            done_list=get_done_task_list(session_id)
        )

@app.get("/stream/{session_id}")
async def stream_query_result(session_id:str,request:Request):
    return StreamingResponse(
        sse_generator(session_id,request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )

# 健康检查
@app.get("/health")
async def health():
    """服务健康检查"""
    logger.info("健康检查接口调用成功")
    return {"ok":True}


@app.get("/history/{session_id}",response_model=HistoryResponse)
def history(session_id:str,limit:int=10):
    records=history_repository.list_recent(session_id,limit=limit)
    items=[
        HistoryItem(
            id=str(record.get("id")) if record.get("id") is not None else "",
            session_id=record.get("session_id",""),
            role=record.get("role",""),
            text=record.get("text", ""),
            rewritten_query=record.get("rewritten_query", ""),
            item_names=record.get("item_names", []),
            image_urls=record.get("image_urls", []),
            ts=record.get("ts"),
        )
        for record in records
    ]
    return HistoryResponse(session_id=session_id,items=items)

@app.delete("/history/{session_id}")
def clear_history(session_id: str):
    """
        清空指定会话的历史记录。

        Args:
            session_id: 目标会话 ID。

        Returns:
            dict: 删除结果说明。
        """
    delete_count=history_repository.clear_session(session_id)
    return ClearHistoryResponse(
        message=f"删除:{session_id}会话对应的聊天记录成功!!",
        deleted_count=delete_count
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.app_host, port=settings.query_app_port)