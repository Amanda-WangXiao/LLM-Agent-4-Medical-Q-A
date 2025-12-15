"""
RAG Module
Uses ChromaDB for medical knowledge retrieval, reranking, and summarization
"""
import os
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer


class RAGSystem:
    """RAG system: retrieval, reranking, summarization"""
    
    def __init__(
        self,
        knowledge_file: str,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 100
    ):
        """
        Initialize RAG system
        
        Args:
            knowledge_file: Path to medical knowledge file
            embedding_model: Embedding model name
            chunk_size: Text chunk size
            chunk_overlap: Text chunk overlap size
        """
        self.knowledge_file = knowledge_file
        self.embedding_model_name = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize ChromaDB (using new PersistentClient API)
        persist_directory = "./chroma_db"
        os.makedirs(persist_directory, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="medical_knowledge",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Text splitter parameters saved, will be used in _build_index
        
        # Check if index needs to be built
        if self.collection.count() == 0:
            print("Building knowledge base index...")
            self._build_index()
        else:
            print(f"Knowledge base already exists with {self.collection.count()} chunks")
    
    def _load_knowledge_file(self) -> str:
        """Load medical knowledge file"""
        with open(self.knowledge_file, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _split_text(self, text: str) -> List[str]:
        """
        Simple text splitter
        Split by paragraphs and sentences, maintain chunk_size and overlap
        """
        # First split by double newlines (paragraphs)
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If current chunk plus new paragraph doesn't exceed chunk_size, add directly
            if len(current_chunk) + len(para) + 2 <= self.chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
            else:
                # If current chunk has content, save it first
                if current_chunk:
                    chunks.append(current_chunk)
                
                # If paragraph itself exceeds chunk_size, need to split further
                if len(para) > self.chunk_size:
                    # Split by sentences
                    sentences = re.split(r'[.!?]\s+', para)
                    current_chunk = ""
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) + 2 <= self.chunk_size:
                            if current_chunk:
                                current_chunk += ". " + sentence
                            else:
                                current_chunk = sentence
                        else:
                            if current_chunk:
                                chunks.append(current_chunk)
                            current_chunk = sentence
                else:
                    current_chunk = para
        
        # Add last chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        # Handle overlap: add overlap between chunks
        if self.chunk_overlap > 0 and len(chunks) > 1:
            overlapped_chunks = [chunks[0]]
            for i in range(1, len(chunks)):
                prev_chunk = chunks[i-1]
                current_chunk = chunks[i]
                
                # Take overlap portion from the end of previous chunk
                if len(prev_chunk) > self.chunk_overlap:
                    overlap_text = prev_chunk[-self.chunk_overlap:]
                    overlapped_chunks.append(overlap_text + "\n\n" + current_chunk)
                else:
                    overlapped_chunks.append(current_chunk)
            
            chunks = overlapped_chunks
        
        return chunks
    
    def _build_index(self) -> None:
        """Build knowledge base index"""
        # Load text
        text = self._load_knowledge_file()
        
        # Split text
        chunks = self._split_text(text)
        
        print(f"Split text into {len(chunks)} chunks")
        
        # Generate embeddings and store
        embeddings = self.embedding_model.encode(chunks, show_progress_bar=True)
        
        # Prepare metadata
        metadatas = []
        ids = []
        for i, chunk in enumerate(chunks):
            # Extract chapter information (if any)
            chapter_match = re.search(r'Chapter\s+\d+', chunk[:200])
            chapter = chapter_match.group(0) if chapter_match else "Unknown"
            
            metadatas.append({
                "chapter": chapter,
                "chunk_id": i,
                "source": Path(self.knowledge_file).name
            })
            ids.append(f"chunk_{i}")
        
        # Add to ChromaDB
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Indexed {len(chunks)} chunks into ChromaDB")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        rerank: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents
        
        Args:
            query: Query text
            top_k: Return top k results
            rerank: Whether to rerank
        
        Returns:
            List of retrieval results, each containing document, metadata, score
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Retrieve (get more candidates for reranking)
        n_results = top_k * 3 if rerank else top_k
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        if not results['documents'] or len(results['documents'][0]) == 0:
            return []
        
        # Format results
        retrieved = []
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            retrieved.append({
                "document": doc,
                "metadata": metadata,
                "distance": distance,
                "score": 1 - distance  # Convert to similarity score
            })
        
        # Rerank
        if rerank:
            retrieved = self._rerank(query, retrieved, top_k)
        
        return retrieved[:top_k]
    
    def _rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidate documents
        
        Uses simple keyword matching combined with semantic similarity
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        for candidate in candidates:
            doc_lower = candidate["document"].lower()
            doc_words = set(doc_lower.split())
            
            # Calculate keyword overlap
            overlap = len(query_words & doc_words)
            keyword_score = overlap / max(len(query_words), 1)
            
            # Combine semantic similarity and keyword matching
            candidate["rerank_score"] = (
                0.7 * candidate["score"] + 0.3 * keyword_score
            )
        
        # Sort by rerank score
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates
    
    def summarize_context(
        self,
        retrieved_docs: List[Dict[str, Any]],
        max_length: int = 1000
    ) -> str:
        """
        Summarize retrieved documents
        
        Args:
            retrieved_docs: List of retrieved documents
            max_length: Maximum summary length
        
        Returns:
            Summary text
        """
        if not retrieved_docs:
            return "No relevant documents found."
        
        # Combine documents
        combined_text = "\n\n---\n\n".join([
            f"[Source: {doc['metadata'].get('chapter', 'Unknown')}]\n{doc['document']}"
            for doc in retrieved_docs
        ])
        
        # If text is too long, truncate
        if len(combined_text) > max_length:
            combined_text = combined_text[:max_length] + "..."
        
        return combined_text
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        summarize: bool = True
    ) -> Dict[str, Any]:
        """
        Complete RAG search flow: retrieval -> reranking -> summarization
        
        Args:
            query: Query text
            top_k: Return top k results
            summarize: Whether to generate summary
        
        Returns:
            Dictionary containing retrieval results and summary
        """
        # Retrieve
        retrieved = self.retrieve(query, top_k=top_k, rerank=True)
        
        result = {
            "query": query,
            "retrieved_docs": retrieved,
            "num_results": len(retrieved)
        }
        
        # Summarize
        if summarize:
            result["summary"] = self.summarize_context(retrieved)
        
        return result

