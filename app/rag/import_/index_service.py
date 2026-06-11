from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import MILVUS_DEFAULT_VARCHAR_MAX_LENGTH, MILVUS_CHUNK_CONTENT_MAX_LENGTH, MILVUS_VECTOR_DIM
from app.shared.runtime.logger import step_log, logger
from pymilvus import MilvusClient, DataType

@step_log("require_chunks")
def require_chunks(state:ImportGraphState)->list[dict]:
    """
    校验导入状态中是否已经生成切块结果
    功能：确保后续流程有有效的输入数据，缺失时抛出异常
    :param state: LangGraph 流程状态字典
    :return: 已通过校验的切块列表
    """
    # 从 state 中获取核心数据
    chunks=state.get("chunks")
    # ===================== 校验 chunks =====================
    # 如果 chunks 为空，无法继续业务，直接抛出异常终止流程
    if not chunks:
        logger.error("chunks为空,无法继续业务!!")
        raise ValueError("chunks为空,无法继续业务!!")
    return chunks

@step_log("prepare_chunks_collection")
def prepare_chunks_collection()->None:
    """
    准备 Milvus 主体名称集合
    功能：检查集合是否存在，不存在则创建 schema 和索引
    :return: 无返回值
    """
    # 获取 Milvus 客户端
    milvus_client=milvus_gateway.client()
    # 获取集合名称（从配置中读取）
    collection_name=milvus_gateway.chunks_collection
    # 如果集合已存在，直接返回，无需重复创建
    if milvus_client.has_collection(collection_name=collection_name):
        return

    # ===================== 创建 Schema =====================
    # 创建 schema，启用自动 ID 和动态字段
    schema=milvus_client.create_schema(
        auto_id=True,
        enable_dynamic_fields=True,
    )

    # 添加主键字段：pk，INT64 类型，自增
    schema.add_field(
        field_name="chunk_id",
        datatype=DataType.INT64,
        is_primary=True,
        auto_id=True,
    )

    # 添加文件标题字段：VARCHAR 类型，最大长度 512
    schema.add_field(
        field_name="file_title",
        datatype=DataType.VARCHAR,
        max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH
    )

    # 添加主体名称字段：VARCHAR 类型，最大长度 512
    schema.add_field(
        field_name="item_name",
        datatype=DataType.VARCHAR,
        max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH
    )

    # 添加切片标题字段：VARCHAR 类型，最大长度 512
    schema.add_field(
        field_name="title",
        datatype=DataType.VARCHAR,
        max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH
    )

    # 添加父标题字段：VARCHAR 类型，最大长度 512
    schema.add_field(
        field_name="parent_title",
        datatype=DataType.VARCHAR,
        max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH
    )

    # 添加切片序号字段：INT8 类型
    schema.add_field(
        field_name="part",
        datatype=DataType.INT8,
    )

    # 添加内容字段：VARCHAR 类型，最大长度 65535（支持长文本）
    schema.add_field(
        field_name="content",
        datatype=DataType.VARCHAR,
        max_length=MILVUS_CHUNK_CONTENT_MAX_LENGTH
    )

    # 添加稠密向量字段：FLOAT_VECTOR 类型，维度 1024
    schema.add_field(
        field_name="dense_vector",
        datatype=DataType.FLOAT_VECTOR,
        dim=MILVUS_VECTOR_DIM
    )

    # 添加稀疏向量字段：SPARSE_FLOAT_VECTOR 类型
    schema.add_field(
        field_name="sparse_vector",
        datatype=DataType.SPARSE_FLOAT_VECTOR,
    )

    # ===================== 创建索引 =====================
    # 准备索引参数
    index_params = milvus_client.prepare_index_params()

    # 为稠密向量创建索引
    index_params.add_index(
        field_name="dense_vector",
        index_type="HNSW",
        metric_type="COSINE",
        params={
            "M": 64,
            "efConstruction": 100
        }
    )

    # 为稀疏向量创建索引：使用 SPARSE_INVERTED_INDEX，算法为 DAAT_MAXSCORE
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
        params={"inverted_index_algo": "DAAT_MAXSCORE"},
    )

    # 创建集合并应用索引
    milvus_client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params
    )

@step_log("remove_old_chunks")
def remove_old_chunks(item_name:str)->None:
    """
    根据主体名称删除已存在的切片记录
    功能：实现幂等性，确保同一主体重复导入时覆盖旧数据
    :param item_name: 主体名称
    :return: 无返回值
    """
    # 获取 Milvus 客户端并执行删除操作
    milvus_gateway.client().delete(
        collection_name=milvus_gateway.chunks_collection,
        filter=f"item_name=='{item_name}'"
    )

@step_log("insert_chunks")
def insert_chunks(chunks):
    """
    批量插入切片数据到 Milvus 集合
    功能：将带向量的切片数据持久化存储到向量库
    :param chunks: 带向量字段的切片列表
    :return: 无返回值
    """
    # 执行批量插入操作
    result=milvus_gateway.client().insert(
        collection_name=milvus_gateway.chunks_collection,
        data=chunks
    )

    # 记录插入结果
    logger.info(f"插入数据成功! 总条数:{result.get('insert_count', 0)}")
    logger.info(f"插入数据主键回显:{result.get('ids', [])}")

@step_log("index_chunks")
def index_chunks(state: ImportGraphState) -> ImportGraphState:
    """
    入库服务总入口
    功能：校验切块数据 → 准备 Milvus 集合 → 清理旧数据 → 批量插入新数据
    输出：更新后的 state（chunks 中会回填 chunk_id）
    """
    # 先校验切片存在，避免把空数据写入向量库
    chunks=require_chunks(state)
    # 集合不存在时先自动创建，保证首次导入也能直接跑通
    prepare_chunks_collection()
    # 获取主体名称，用于幂等性清理
    item_name=state.get("item_name","")
    # 同一主体重复导入时先删旧数据，保持当前导入结果覆盖旧版本
    if item_name:
        remove_old_chunks(item_name)
    # 批量插入新数据
    insert_chunks(chunks)

    return state