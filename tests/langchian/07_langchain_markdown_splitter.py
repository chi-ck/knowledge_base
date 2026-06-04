from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# UnstructuredMarkdownLoader 的职责：负责“读取文件 -> 产出 Document 对象”。
# 它解决的是“怎么把 markdown 文件加载进来”，不负责精细切块策略。
# UnstructuredMarkdownLoader：主要负责“读取文档”，可顺带按内置策略切分（如按标题）。
# 局限：你很难精确指定“先按 \n\n，再按 \n，再按空格”，也不方便验证 chunk_overlap 的实际生效细节。
# 所以本示例用 RecursiveCharacterTextSplitter：显式控制 separators/chunk_size/chunk_overlap，便于讲清递归切分原理。
docs = UnstructuredMarkdownLoader("./assets/splitter_demo.md", mode="single").load()

# RecursiveCharacterTextSplitter 的职责：负责“把长文本切成可检索的小块”。
# 它解决的是“怎么切块”，不是“怎么读取文件”。
#
# 递归切分原理（先粗后细）：
# 1) 先用 separators[0]（这里是 "\n\n"）尝试切分；
# 2) 若某段仍大于 chunk_size，则对该段递归使用下一层分隔符（"\n" -> " " -> ""）；
# 3) 最后在满足长度约束的前提下，尽量保留 chunk_overlap 的重叠内容。
#
# 关键参数说明：
# - separators: 分隔符优先级列表，越靠前语义越强（段落 > 换行 > 空格 > 字符级）。
# - chunk_size: 单个 chunk 的目标最大长度（按 length_function 计算）。
# - chunk_overlap: 相邻 chunk 尽量重叠的长度（目标值，不保证每次精确命中）。
chunks = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " "],
    chunk_size=10,
    chunk_overlap=5 # overlap 要 >= 当前层最小可保留片段长度 。
).split_documents(docs)

for i, c in enumerate(chunks, 1):
    print(f"chunk{i}: {repr(c.page_content)} len={len(c.page_content)}")


# Aaaa Bbbb C
# cccc
