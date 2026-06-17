import re

from app.infra.llm.providers import llm_provider
from app.infra.persistence.history_repository import history_repository
from app.process.query.agent.state import QueryGraphState
from app.rag.query.item_name_confirm_service import build_history_text
from app.shared.runtime.load_prompt import load_prompt
from app.shared.utils.task_utils import add_done_task, add_running_task, push_to_session, set_task_result
from app.shared.utils.sse_utils import SSEEvent
from app.shared.runtime.logger import logger, step_log
import time


@step_log("try_return_existing_answer")
def try_return_existing_answer(state: dict) -> bool:
    """
    复用已有答案：如果 state 中已存在 answer，直接返回，不再调用模型
    流式模式：逐字推送内容；非流式：直接设置结果
    """
    answer = state.get("answer")
    is_stream = state.get("is_stream", False)
    session_id = state.get("session_id")

    # 无现成答案，返回 False，继续生成
    if not answer:
        return False

    # 流式模式：逐字符推送增量消息
    if is_stream:
        for ch in answer:
            push_to_session(session_id, SSEEvent.DELTA, {"delta": ch})
            time.sleep(0.1)

    # 将最终答案写入任务结果
    set_task_result(session_id, "answer", answer)
    return True

@step_log("validate_generation_inputs")
def validate_generation_inputs(state: dict) -> tuple[list[dict], list[str], str, list[dict]]:
    """
    答案生成前的必填参数校验
    必须包含：reranked_docs（参考资料）、query（用户问题）
    """
    history = state.get("history", [])
    reranked_docs = state.get("reranked_docs")
    item_names = state.get("item_names", [])
    rewritten_query = state.get("rewritten_query") or state.get("original_query")

    # 关键校验：缺少资料或问题则无法生成
    if not reranked_docs or not rewritten_query:
        raise ValueError("生成答案需要 reranked_docs 和 rewritten_query/original_query")

    return reranked_docs, item_names, rewritten_query, history

@step_log("build_answer_prompt")
def build_answer_prompt(
    reranked_docs: list[dict],
    rewritten_query: str,
    item_names: list[str],
    history: list[dict],
) -> str:
    """
    构建最终答案生成的 Prompt
    整合：参考资料、得分、来源、对话历史、关联主体、用户问题
    """
    context_chunk_list = []
    # 遍历重排后的文档，按序号拼接参考内容
    for number, chunk in enumerate(reranked_docs, start=1):
        context_chunk_list.append(
            f"第{number}块: 标题:{chunk['title']} 匹配度得分:{chunk['score']} 来源:{'网络搜索' if chunk['type'] == 'web' else '向量查询'}\n内容:{chunk['text']}"
        )

    # 合并所有参考块
    context_chunk_str = "\n\n".join(context_chunk_list)
    # 格式化对话历史
    history_text = build_history_text(history)
    # 格式化关联主体
    item_name_str = "本次关联主体:" + ",".join(item_names) if item_names else "没有关联主体"

    # 加载 answer_out 模板，生成最终 Prompt
    return load_prompt(
        "answer_out",
        context=context_chunk_str,
        history=history_text,
        item_names=item_name_str,
        question=rewritten_query,
    )

@step_log("final_answer")
def final_answer(state: dict, prompt: str) -> str:
    """
    调用大模型生成最终答案
    支持：流式输出 / 普通输出
    结果存入 state 并同步到任务结果
    """
    is_stream = state.get("is_stream", False)
    session_id = state.get("session_id")
    lm_client = llm_provider.chat()
    final_result = ""

    # 流式生成：逐块接收并推送
    if is_stream:
        for chunk in lm_client.stream(prompt):
            delta_content = chunk.content
            final_result += delta_content
            push_to_session(session_id, SSEEvent.DELTA, {"delta": delta_content})
    # 普通生成：一次性调用
    else:
        response = lm_client.invoke(prompt)
        final_result = response.content

    # 保存答案到任务与状态
    set_task_result(session_id, "answer", final_result)
    state["answer"] = final_result
    return final_result

@step_log("extract_image_urls")
def extract_image_urls(reranked_docs: list[dict]) -> list[str]:
    """
    从参考文档中提取所有图片 URL
    提取来源：
    1. doc.url 直接是图片链接
    2. text 内的 markdown 图片格式 ![](url)
    """
    image_urls: list[str] = []
    # 匹配 markdown 图片正则
    reg = re.compile(r"\!\[.*?\]\((.*?)\)")

    for doc in reranked_docs:
        url = doc.get("url")
        text = doc.get("text")

        # 提取直接作为 URL 的图片
        if url and url.endswith((".png", ".jpg", ".gif", ".jpeg", ".svg")) and url not in image_urls:
            image_urls.append(url)

        # 提取文本中的 markdown 图片
        if text:
            for image_url in reg.findall(text):
                if image_url not in image_urls:
                    image_urls.append(image_url)

    return image_urls

@step_log("save_assistant_message")
def save_assistant_message(state: dict) -> None:
    """
    将助手回答保存到历史记录（Mongo）
    包含：答案、问题、主体、图片链接
    """
    history_repository.save_message(
        session_id=state["session_id"],
        role="assistant",
        text=state.get("answer"),
        rewritten_query=state.get("rewritten_query") or state.get("original_query"),
        item_names=state.get("item_names", []),
        image_urls=state.get("image_urls", []),
    )


@step_log("generate_answer")
def generate_answer(state: dict) -> dict:
    """
    答案输出节点主入口（全流程编排）
    流程：
    1. 尝试复用已有答案
    2. 校验生成参数
    3. 构建 Prompt
    4. 调用模型生成答案
    5. 提取图片
    6. 保存历史记录
    """
    # 如果已有答案，直接返回
    if not try_return_existing_answer(state):
        # 校验输入
        reranked_docs, item_names, rewritten_query, history = validate_generation_inputs(state)
        # 构建提示词
        prompt = build_answer_prompt(reranked_docs, rewritten_query, item_names, history)
        # 生成答案
        final_answer(state, prompt)
        # 提取图片 URL
        state["image_urls"] = extract_image_urls(reranked_docs)

    # 保存助手消息到历史
    save_assistant_message(state)
    return state