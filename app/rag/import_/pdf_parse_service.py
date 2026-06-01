import shutil
import time
from pathlib import Path
import requests

from app.infra.config.providers import infra_config
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import MINERU_MODEL_VERSION, MINERU_POLL_TIMEOUT_SECONDS, MINERU_POLL_INTERVAL_SECONDS
from app.shared.runtime.logger import logger, step_log, PROJECT_ROOT


@step_log("validate_pdf_paths")
def validate_pdf_paths(state:dict)->tuple[Path, Path]:
    """
    校验PDF文件路径与输出目录，确保文件存在、目录可用并自动补全目录
    :param state: 流程状态字典，包含 pdf_path、local_dir 字段
    :return: 元组(解析后的PDF路径对象，本地输出目录对象)
    :raises ValueError: pdf_path 为空时抛出异常
    :raises FileNotFoundError: PDF文件不存在时抛出异常
    """

    # 从状态中读取PDF文件路径和本地输出目录
    pdf_path = state["pdf_path"]
    local_dir = state["local_dir"]

    # 检验PDF路径是否为空，为空则终止流程并抛出异常
    if not pdf_path:
        logger.error("pdf_path的参数值为空,无法读取文件!")
        raise ValueError("pdf_path的参数值为空,无法读取文件!")
    # 输出目录为空时，使用项目根目录下的output作为默认目录，并回写到状态中
    if not local_dir:
        logger.warning("没有传入local_dir地址,给与默认值!")
        local_dir = PROJECT_ROOT/"output"
        state["local_dir"] = str(local_dir)

    # 转为Path对象，方便后续路径操作
    pdf_path_obj = Path(pdf_path)
    local_dir_obj = Path(local_dir)

    # 校验PDF文件是否真实存在，不存在则抛出异常
    if not pdf_path_obj.exists():
        logger.error(f"pdf_path:{pdf_path_obj},但是没有文件存在!")
        raise FileNotFoundError(f"pdf_path:{pdf_path_obj},但是没有文件存在!")

    # 输出目录不存在则自动创建（支持多级目录，已存在也不报错）
    if not local_dir_obj.exists():
        logger.warning(f"local_dir:{local_dir_obj}地址没有文件夹,我们需要主动创建!")
        local_dir_obj.mkdir(parents=True, exist_ok=True)

    # 返回路径对象供后续业务使用
    return pdf_path_obj, local_dir_obj

@step_log("upload_pdf_and_poll")
def upload_pdf_and_poll(pdf_path_obj:Path)->str:
    """
    上传PDF文件到MinerU服务，并轮询等待解析完成，最终返回解析结果的下载地址
    :param pdf_path_obj: 本地PDF文件的路径对象
    :return: 解析完成后的ZIP压缩包下载地址
    """

    # 1. 校验MinerU服务配置（base_url和api_key必须存在）
    if not infra_config.mineru.base_url or not infra_config.mineru.api_key:
        logger.error("minerU配置错误,请检查minerU配置!")
        raise ValueError("minerU配置错误,请检查minerU配置!")

    # 2. 构造请求地址和请求头
    url=f"{infra_config.mineru.base_url}/file-urls/batch"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {infra_config.mineru.api_key}"
    }

    # 3. 构造请求参数：文件名 + 使用的模型版本
    payload = {
        "files":[{"name":pdf_path_obj.stem}],
        "model_version":MINERU_MODEL_VERSION
    }

    # 4. 请求MinerU获取文件预上传地址
    response=requests.post(url=url, headers=headers, json=payload)
    if response.status_code != 200:
        raise RuntimeError(f"申请上传地址失败,返回状态码为:{response.status_code},请检查minerU配置!")

    # 5. 解析返回结果，校验业务状态码
    result_dict=response.json()
    if result_dict["code"] != 0:
        raise RuntimeError(
            f"申请地址网络状态成功!但是业务失败!错误码:{result_dict['code']},失败信息:{result_dict['msg']}"
        )

    # 6. 提取上传URL和任务批次ID
    file_upload_url = result_dict["data"]["file_urls"][0]
    batch_id=result_dict["data"]["batch_id"]

    # 7. 上传PDF文件到MinerU的存储地址
    with requests.Session() as session:
        # session.trust_env = False 是告诉 requests ： 不要信任系统环境变量里的代理、证书、认证等配置 。
        # 它的实际含义是：1. 先创建一个会话对象 session 2. 明确关闭“从系统环境自动继承网络配置”3. 再用这个“干净会话”去执行 PUT 上传
        # 上传的地址是, 预签名上传地址 , OSS / S3 / 对象存储直传地址
        session.trust_env=False
        upload_response=session.put(file_upload_url, data=pdf_path_obj.read_bytes())
        if upload_response.status_code != 200:
            raise RuntimeError(f"上传文件失败,返回状态码为:{upload_response.status_code},请检查minerU配置!")

    # 8. 构造轮询查询地址
    poll_url=f"{infra_config.mineru.base_url}/extract-results/batch/{batch_id}"
    timeout=MINERU_POLL_TIMEOUT_SECONDS
    interval_time=MINERU_POLL_INTERVAL_SECONDS
    start_time=time.time()

    # 9. 开始轮询任务解析状态
    while True:
        # 超时判断
        if time.time() - start_time > timeout:
            raise TimeoutError("轮询超时,请检查minerU配置!")

        try:
            # 请求轮询接口
            poll_response=requests.get(poll_url,headers=headers)
        except Exception:
            logger.warning("请求出现异常!可以稍后重试!!")
            time.sleep(interval_time)
            continue

        # 网络异常处理：5xx可重试，其他直接报错
        if poll_response.status_code != 200:
            if 500<=poll_response.status_code<600:
                logger.warning(f"可有修复的网络异常,状态码为:{poll_response.status_code}")
                time.sleep(interval_time)
                continue
            raise RuntimeError(f"不可修复的网络状态异常,状态码为:{poll_response.status_code}")

        # 解析返回结果
        poll_response_dict=poll_response.json()
        if poll_response_dict["code"] != 0:
            raise RuntimeError(
                f"轮询业务异常,错误码:{poll_response_dict['code']},失败信息:{poll_response_dict['msg']}")

        # 10. 获取解析任务状态
        extract_result=poll_response_dict["data"]["extract_result"][0]
        extract_result_state=extract_result["state"]

        # 任务完成 → 返回下载地址
        if extract_result_state == "done":
            extract_result_url=extract_result["full_zip_url"]
            if not extract_result_url:
                raise RuntimeError("已经完成了解析,但是zip地址为空!!")
            return extract_result_url

        # 任务失败 → 抛出异常
        if extract_result_state=="failed":
            raise RuntimeError(f"已经完成了解析,但是失败了!!失败信息:{extract_result['err_msg']}")

        # 任务仍在处理中 → 等待后继续轮询
        logger.warning(f"解析正在进行中,状态:{extract_result_state}!")
        time.sleep(interval_time)

@step_log("download_and_extract_markdown")
def download_and_extract_markdown(zip_url: str, local_dir_path_obj: Path, stem: str) -> Path:
    """
    下载 MinerU 解析完成的 ZIP 压缩包，解压并提取标准的 MD 文件
    1. 从 zip_url 下载解析结果压缩包
    2. 解压到指定目录
    3. 自动查找最合适的 MD 文件（优先同名 → full.md → 第一个）
    4. 重命名为统一规范的文件名并返回
    :param zip_url: MinerU 返回的ZIP 下载地址
    :param local_dir_path_obj: 本地存放解压文件的目录
    :param stem: 原始 PDF 的文件名（不带后缀，用于重命名 MD）
    :return: 最终整理好的 MD 文件路径对象
    """
    # ---------------------- 1. 下载 ZIP 压缩包 ----------------------
    # 发送请求下载解析好的 ZIP 文件
    response = requests.get(zip_url, timeout=MINERU_POLL_TIMEOUT_SECONDS)
    # 拼接 ZIP 保存路径：输出目录 + 文件名_result.zip
    zip_path_obj=local_dir_path_obj/f"{stem}_result.zip"
    # 将二进制内容写入 ZIP 文件
    zip_path_obj.write_bytes(response.content)

    # ---------------------- 2. 解压 ZIP 文件 ----------------------
    # 解压目录 = 输出目录 / PDF 文件名（无后缀）
    extract_path_obj=local_dir_path_obj/stem
    # 如果解压目录已存在，先删除（防止旧文件干扰）
    if extract_path_obj.exists():
        shutil.rmtree(extract_path_obj)
    # 创建新的解压目录
    extract_path_obj.mkdir(parents=True, exist_ok=True)
    # 解压 ZIP 包到目标目录
    shutil.unpack_archive(zip_path_obj,extract_path_obj)

    # ---------------------- 3. 查找所有 MD 文件 ----------------------
    # 递归查找解压目录下所有 .md 文件
    md_file_list=list(extract_path_obj.rglob("*.md"))
    # 没有找到 MD 文件则抛出异常
    if not md_file_list:
        raise FileNotFoundError(f"文件解压失败,在:{extract_path_obj}没有任何md文件!")

    # ---------------------- 4. 按优先级选择 MD 文件 ----------------------
    # 优先级 1：找和 PDF 同名的 MD（最标准）
    for md_file in md_file_list:
        if md_file.stem==stem:
            return md_file

    # 优先级 2：找不到同名，找 full.md（MinerU 默认完整导出文件）
    target_md_obj=None
    for md_file in md_file_list:
        if md_file.name.lower()=="full.md":
            target_md_obj=md_file
            break

    # 优先级 3：还找不到，直接取第一个 MD
    if not target_md_obj:
        target_md_obj=md_file_list[0]

    # ---------------------- 5. 重命名为统一规范名称 ----------------------
    # 将选中的 MD 重命名为 {stem}.md（和 PDF 同名）
    return target_md_obj.rename(target_md_obj.with_name(f"{stem}.md"))

@step_log("parse_pdf_to_markdown")
def parse_pdf_to_markdown(state: ImportGraphState) -> ImportGraphState:
    """
    PDF 解析服务：
    1. 调用 MinerU
    2. 下载并解压解析结果
    3. 获取 Markdown 路径和正文内容
    4. 回写 md_path / md_content / local_dir
    """
    # 先校验 PDF 路径和输出目录，避免把非法输入送进解析服务
    pdf_path_obj,local_dir_path_obj=validate_pdf_paths(state)

    # 上传 PDF 到 MinerU，并轮询直到服务端返回最终压缩包地址。
    zip_url=upload_pdf_and_poll(pdf_path_obj)
    logger.info(f"minerU返回得zip地址:{zip_url}")

    # 下载结果包并提取最终 Markdown 文件。
    md_path_obj=download_and_extract_markdown(zip_url, local_dir_path_obj,pdf_path_obj.stem)

    state["md_path"]=str(md_path_obj)
    state["md_content"]=md_path_obj.read_text(encoding="utf-8")

    return state