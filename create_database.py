import os
import shutil
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

# env
load_dotenv()

# path
CHROMA_PATH = "chroma"
DATA_PATH = "ref"

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

# load all content from .md
def load_documents():
    file_path = os.path.join(DATA_PATH, "BasicsofAnesthesia.md")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        document = Document(page_content=content, metadata={"source": file_path})
        print(f"Loaded document from {file_path}")
        #print(f"Document content preview: {content[:100]}...")
        return [document]
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return []


def split_text(documents: list[Document]):
    # Define CharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,  # for contextual coherence
        length_function=len, # len is calculated in Characters.
        add_start_index=True,
    )

    document_chunks = []
    total_chunks = 0

    for doc in documents:
        # split
        chunks = text_splitter.split_text(doc.page_content) # {page_content, metadata}
        total_chunks = total_chunks + len(chunks)

        # create Document for every chunk
        for i, chunk in enumerate(chunks):
            new_doc = Document(
                page_content=chunk,
                metadata={
                    **doc.metadata,  # save original metadata
                    "chunk_id": i,   # add new metadata
                    "total_chunks": len(chunks)
                }
            )
            document_chunks.append(new_doc)

    print(f"Split {len(documents)} documents into {total_chunks} chunks.")
    return document_chunks


def save_to_chroma(chunks: list[Document]):
    if not chunks:
        print("No chunks to save to Chroma. Exiting.")
        return

    # clean current database
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # create embedding: convert text to numeric vector
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    try:
        # create new database
        db = Chroma.from_documents(
            chunks, embedding_function, persist_directory=CHROMA_PATH
        )
        print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

        # simple test
        query = "Anesthesia"
        results = db.similarity_search(query)
        if results:
            print(f"Found {len(results)} results for query '{query}'")
            #print("First result:", results[0].page_content[:100] + "...")
        else:
            print(f"No results found for query '{query}'")
    except Exception as e:
        print(f"Error saving to or querying Chroma: {str(e)}")


if __name__ == "__main__":
    generate_data_store()