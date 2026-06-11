"""
应用主包 / 接口层 / 数据模型层中的 import_ 模块，负责承载对应场景的具体实现逻辑。
"""
from pydantic import BaseModel

# 继承 BaseModel ，是为了让 FastAPI / Pydantic 把这个类当成“数据模型”处理。
# 这样它才会帮你做这些事：
#- 解析 JSON
#- 类型校验
#- 自动补默认值
#- 自动生成接口文档
#- 自动把对象转成 JSON 响应

class UploadResponse(BaseModel):
    code:int=200
    message: str
    task_ids:list[str]

class ImportStatusResponse(BaseModel):
    code:int=200
    task_id:str
    status:str|None=None
    done_list:list[str]
    running_list:list[str]
