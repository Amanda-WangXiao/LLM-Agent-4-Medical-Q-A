import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# 设置常量
CHROMA_PATH = "chroma"  # 确保这与创建数据库时使用的路径相同
MAX_DOCS_TO_PRINT = 5  # 限制打印的文档数量


def view_chroma_content():
    # check exist
    if not os.path.exists(CHROMA_PATH):
        print(f"Chroma database not found at {CHROMA_PATH}")
        return

    # reate embedding: convert text to numeric vector
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # load current Chroma database
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # get documents
    results = db.get()

    if not results['ids']:
        print("No documents found in the database.")
        return

    # print content & metadata, up to MAX_DOCS_TO_PRINT
    for i in range(min(MAX_DOCS_TO_PRINT, len(results['ids']))):
        print(f"\nDocument {i + 1}:")
        print(f"\nID: {results['ids'][i]}")
        print(f"\nContent: {results['documents'][i][:200]}...")  # 打印前200个字符

        # metadata
        metadata = results['metadatas'][i]
        print("\nMetadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")

        print("-" * 50)

    total_docs = len(results['ids'])
    print(f"\nPrinted {min(MAX_DOCS_TO_PRINT, total_docs)} out of {total_docs} total documents in the database.")

    # Statistics
    chunk_ids = [meta.get('chunk_id', 'N/A') for meta in results['metadatas']]
    unique_sources = set(meta.get('source', 'Unknown') for meta in results['metadatas'])

    print(f"\nStatistics:")
    print(f"Total unique documents: {len(unique_sources)}")
    print(f"Total chunks: {total_docs}")
    print(f"Chunk IDs range: {min(chunk_ids)} to {max(chunk_ids)}")


if __name__ == "__main__":
    view_chroma_content()