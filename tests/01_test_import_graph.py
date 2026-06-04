import json

from app.process.import_.agent.main_graph import kb_import_app
from app.process.import_.agent.state import ImportGraphState,create_default_state
from app.shared.runtime.logger import logger

# 1. 执行下图对象
state = create_default_state(task_id="9527",local_file_path="赵伟风哈哈哈.pdf")
logger.info(f"进入state: \n {json.dumps(state, ensure_ascii=False, indent=4)}")

output_state = kb_import_app.invoke(state)

# output_state_stream = kb_import_app.stream(state)
#
# for chunk in output_state_stream:
#     # chunk 每个节点后的结果  -> chunk > {"节点名": state}
#     for node_name, node_state in chunk.items():
#         logger.info(f"{node_name} 结果: \n {json.dumps(node_state, ensure_ascii=False, indent=4)}")

logger.info("*"*100)
logger.info(f"结束state: \n {json.dumps(output_state, ensure_ascii=False, indent=4)}")


# 2. 静态查看图的结构和设计的一样
# 今天打印图结构
kb_import_app.get_graph().print_ascii()