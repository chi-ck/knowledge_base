"""
导入服务 HTTP 入口模块，直接承载导入接口与相关接口业务逻辑。
"""
import shutil
import sys
import uuid
from datetime import datetime
from mimetypes import guess_type
from pathlib import Path
from typing import List

from fastapi import BackgroundTasks, FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from starlette.middleware.cors import CORSMiddleware

from app.api.schemas.import_ import ImportStatusResponse, UploadResponse
from app.shared.config.settings_config import settings
from app.shared.runtime.logger import PROJECT_ROOT, logger
from app.process.import_.agent.main_graph import kb_import_app
from app.process.import_.agent.state import get_default_state

from app.shared.utils.task_utils import (
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_PROCESSING,
    get_done_task_list,
    get_running_task_list,
    get_task_status,
    update_task_status, add_running_task, add_done_task,
)


app=FastAPI(
    title=settings.import_app_name,
    description="企业化 RAG 导入服务，负责文件上传、导入执行与状态查询。",
    version="0.2.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins) or ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/html")
def import_html():
    """
    返回导入演示页面。
    Returns:
        FileResponse: 本地导入演示页面文件响应。
    """
    html_path=PROJECT_ROOT/"app"/"process"/"import_"/"page"/"import.html"
    return FileResponse(path=html_path,media_type=guess_type(html_path.name)[0])


# --------------------------
# 后台任务：LangGraph全流程执行
# 独立于主请求线程，由BackgroundTasks触发，避免阻塞接口响应
# --------------------------
def run_graph_task(task_id: str, local_dir: str, local_file_path: str):
    """
    LangGraph全流程执行后台任务
    核心流程：初始化状态 → 流式执行图节点 → 实时更新任务状态 → 异常捕获
    任务状态更新：pending → processing → completed/failed
    节点进度更新：每完成一个节点，将节点名加入done_list，供前端轮询查看

    :param task_id: 全局唯一任务ID，关联单个文件的全流程处理
    :param local_dir: 该任务的本地文件存储目录（含临时文件/解析结果）
    :param local_file_path: 上传文件的本地绝对路径
    """

    try:
        # 1. 更新任务全局状态为：处理中
        update_task_status(task_id, TASK_STATUS_PROCESSING)
        logger.info(f"[{task_id}] 开始执行LangGraph全流程，本地文件路径：{local_file_path}")

        # 2. 初始化LangGraph状态：加载默认状态 + 注入当前任务的核心参数
        init_state = get_default_state()
        init_state["task_id"] = task_id  # 任务ID关联
        init_state["local_dir"] = local_dir  # 任务本地目录
        init_state["local_file_path"] = local_file_path  # 上传文件本地路径

        # 3. 流式执行LangGraph全流程（stream模式：实时获取每个节点的执行结果）
        for event in kb_import_app.stream(init_state):
            for node_name, node_result in event.items():
                # 记录每个节点完成的日志，包含任务ID和节点名，方便追踪执行顺序
                logger.info(f"[{task_id}] LangGraph节点执行完成：{node_name}")
                # 将完成的节点名加入【已完成列表】，前端轮询/status/{task_id}可实时获取
                add_done_task(task_id, node_name)

        # 4. 全流程执行完成，更新任务全局状态为：已完成
        update_task_status(task_id, "completed")
        logger.info(f"[{task_id}] LangGraph全流程执行完毕，任务完成")

    except Exception as e:
        # 5. 捕获全流程异常，更新任务全局状态为：失败，并记录错误日志（含堆栈）
        update_task_status(task_id, "failed")
        logger.error(f"[{task_id}] LangGraph全流程执行失败，异常信息：{str(e)}", exc_info=True)



@app.post("/upload",summary="文件上传接口", description="支持多文件批量上传")
async def upload_files(
        background_tasks: BackgroundTasks,
        files:List[UploadFile] = File(...),
):
    # 1. 构建本地存储根目录：output/YYYYMMDD
    today_str=datetime.now().strftime("%Y%m%d")
    data_based_root_dir=PROJECT_ROOT/"output"/today_str

    task_ids=[]

    # 2. 遍历处理每个上传的文件
    for file in files:
        task_id=str(uuid.uuid4())
        task_ids.append(task_id)

        # 3. 标记「文件上传」阶段为「运行中」
        add_running_task(task_id,"upload_file")

        # 4. 构建任务的本地独立目录
        task_local_dir:Path=data_based_root_dir/task_id
        task_local_dir.mkdir(parents=True,exist_ok=True)

        # 5. 保存文件到本地
        local_file_abs_path:Path=task_local_dir/file.filename
        with local_file_abs_path.open("wb") as file_buffer:
            # copyfileobj 好处：
            # 流式读取：一次只读一小段（默认 64KB）
            # 读完一段写一段，循环直到写完
            # 内存永远只占用 64KB，不管文件多大
            # 速度极快，系统底层优化
            # 不会阻塞服务器，支持高并发
            # 自带缓冲区，不用自己处理
            shutil.copyfileobj(file.file, file_buffer)

        # 6. 标记「文件上传」阶段为「已完成」
        add_done_task(task_id,"upload_file")

        # 7. 启动后台任务
        background_tasks.add_task(
            run_graph_task,
            task_id,
            str(task_local_dir),
            str(local_file_abs_path)
        )

        return UploadResponse(
            code=200,
            message=f"Files uploaded successfully, total: {len(files)}",
            task_ids=task_ids
        )


@app.get("/status/{task_id}",summary="任务状态逻辑",
         response_model=ImportStatusResponse)
async def get_task_progress(task_id:str):
    status = get_task_status(task_id)
    done_list = get_done_task_list(task_id)
    running_list = get_running_task_list(task_id)

    logger.info(f"[{task_id}] 任务状态查询，当前状态：{status}，已完成节点：{done_list}")

    return ImportStatusResponse(code=200,task_id=task_id,status=status, done_list=done_list, running_list=running_list)



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.app_host, port=settings.import_app_port)