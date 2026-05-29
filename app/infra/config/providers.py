from app.shared.config import LLMConfig, EmbeddingConfig, RerankerConfig, McpConfig, MilvusConfig, MinerUConfig, \
    MinIOConfig
from app.shared.config.embedding_config import embedding_config
from app.shared.config.lm_config import lm_config
from app.shared.config.bailian_mcp_config import mcp_config
from app.shared.config.milvus_config import milvus_config
from app.shared.config.mineru_config import mineru_config
from app.shared.config.minio_config import minio_config
from app.shared.config.reranker_config import reranker_config
from app.shared.config.settings_config import settings, AppSettings

from dataclasses import dataclass , field

@dataclass
class InfraConfig:
    app: AppSettings = field(default_factory=lambda: settings)
    llm: LLMConfig = field(default_factory=lambda: lm_config)
    embedding: EmbeddingConfig = field(default_factory=lambda: embedding_config)
    reranker: RerankerConfig = field(default_factory=lambda: reranker_config)
    mcp: McpConfig = field(default_factory=lambda: mcp_config)
    milvus: MilvusConfig = field(default_factory=lambda: milvus_config)
    mineru: MinerUConfig = field(default_factory=lambda: mineru_config)
    minio: MinIOConfig = field(default_factory=lambda: minio_config)

infra_config = InfraConfig()

print(infra_config.app.import_app_name)