# MinerU 模型版本配置（vlm = 视觉语言模型，适合PDF/图片高精度解析）
MINERU_MODEL_VERSION="vlm"

# MinerU 任务轮询最大超时时间（单位：秒），超过则判定任务失败
MINERU_POLL_TIMEOUT_SECONDS=600

# MinerU 任务轮询间隔时间（单位：秒），每隔多久查询一次任务状态
MINERU_POLL_INTERVAL_SECONDS=3

# MinerU 文件下载超时时间（单位：秒），下载文件超过此时长则中断
MINERU_DOWNLOAD_TIMEOUT_SECONDS=30

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}


# 文本切块最大长度：单个文本块最多包含 1000 字符（防止过长导致向量失真）
CHUNK_MAX_SIZE = 1000
# 文本切块基准长度：单个文本块理想大小为 600 字符（兼顾语义完整性 + 检索精度）
CHUNK_SIZE = 600
# 文本块重叠长度：相邻块之间重叠 20 字符，保证语义不被切断、上下文连贯
CHUNK_OVERLAP = 20


# 主体识别上下文切片数：取前 K 个切片用于 LLM 识别
ITEM_NAME_CONTEXT_CHUNK_K = 5
# 主体识别上下文总字符数上限：防止上下文过长导致大模型输入超限
ITEM_NAME_CONTEXT_TOTAL_MAX_CHARS = 10000

# Milvus VARCHAR 字段最大长度（用于 title、item_name 等短文本）
MILVUS_DEFAULT_VARCHAR_MAX_LENGTH = 512
# Milvus content 字段最大长度（用于存储长文本内容）
MILVUS_CHUNK_CONTENT_MAX_LENGTH = 65535
# Milvus 向量维度（BGE-M3 稠密向量维度）
MILVUS_VECTOR_DIM = 1024


# 向量化批次大小：每批处理 5 条切片，避免显存溢出
EMBEDDING_BATCH_SIZE = 5