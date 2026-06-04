from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

docs = UnstructuredMarkdownLoader("./assets/splitter_overlap_demo.md", mode="single").load()

chunks = RecursiveCharacterTextSplitter(
    separators=[""],   # 字符级切分，最容易观察 overlap
    chunk_size=10,
    chunk_overlap=4
).split_documents(docs)

for i, c in enumerate(chunks, 1):
    print(f"chunk{i}: {repr(c.page_content)} len={len(c.page_content)} , chunk_meta = {c.metadata}")