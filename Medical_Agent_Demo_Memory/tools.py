"""
Tools Module
Implements various tools: RAGSearchTool, WebSearchTool, TranslationTool
"""
import os
from typing import Dict, Any, List, Optional
from duckduckgo_search import DDGS
from rag import RAGSystem


class RAGSearchTool:
    """Medical knowledge base search tool"""
    
    def __init__(self, rag_system: RAGSystem):
        """
        Initialize RAG search tool
        
        Args:
            rag_system: RAG system instance
        """
        self.rag_system = rag_system
        self.name = "RAGSearchTool"
        self.description = "Search medical knowledge base from local documents using RAG"
    
    def execute(
        self,
        query: str,
        top_k: int = 3,
        summarize: bool = True
    ) -> Dict[str, Any]:
        """
        Execute RAG search
        
        Args:
            query: Search query
            top_k: Return top k results
            summarize: Whether to generate summary
        
        Returns:
            Search result dictionary
        """
        try:
            result = self.rag_system.search(
                query=query,
                top_k=top_k,
                summarize=summarize
            )
            
            # Format output
            formatted_result = {
                "tool": self.name,
                "query": query,
                "status": "success",
                "num_results": result["num_results"],
                "summary": result.get("summary", ""),
                "documents": []
            }
            
            # Add document details
            for doc in result["retrieved_docs"]:
                formatted_result["documents"].append({
                    "content": doc["document"][:300] + "..." if len(doc["document"]) > 300 else doc["document"],
                    "metadata": doc["metadata"],
                    "relevance_score": round(doc.get("rerank_score", doc["score"]), 3)
                })
            
            return formatted_result
        
        except Exception as e:
            return {
                "tool": self.name,
                "query": query,
                "status": "error",
                "error": str(e)
            }
    
    def get_structured_context(self, result: Dict[str, Any]) -> str:
        """
        Get structured context
        
        Args:
            result: RAG search result
        
        Returns:
            Formatted context string
        """
        if result["status"] != "success":
            return f"Error: {result.get('error', 'Unknown error')}"
        
        context = f"=== RAG Search Results ===\n"
        context += f"Query: {result['query']}\n"
        context += f"Found {result['num_results']} relevant documents\n\n"
        
        if result.get("summary"):
            context += f"Summary:\n{result['summary']}\n\n"
        
        context += "Detailed Results:\n"
        for i, doc in enumerate(result["documents"], 1):
            context += f"\n[{i}] Relevance: {doc['relevance_score']}\n"
            context += f"Source: {doc['metadata'].get('chapter', 'Unknown')}\n"
            context += f"Content: {doc['content']}\n"
        
        return context


class WebSearchTool:
    """Web search tool (using DuckDuckGo)"""
    
    def __init__(self, max_results: int = 5):
        """
        Initialize web search tool
        
        Args:
            max_results: Maximum number of results to return
        """
        self.name = "WebSearchTool"
        self.description = "Search the web for latest medical information using DuckDuckGo"
        self.max_results = max_results
    
    def execute(self, query: str, max_results: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute web search
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
        
        Returns:
            Search result dictionary
        """
        max_results = max_results or self.max_results
        
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(
                    query,
                    max_results=max_results,
                    safesearch='moderate'
                ))
            
            formatted_result = {
                "tool": self.name,
                "query": query,
                "status": "success",
                "num_results": len(results),
                "results": []
            }
            
            for result in results:
                formatted_result["results"].append({
                    "title": result.get("title", ""),
                    "snippet": result.get("body", ""),
                    "url": result.get("href", "")
                })
            
            return formatted_result
        
        except Exception as e:
            return {
                "tool": self.name,
                "query": query,
                "status": "error",
                "error": str(e)
            }
    
    def get_structured_context(self, result: Dict[str, Any]) -> str:
        """
        Get structured context
        
        Args:
            result: Web search result
        
        Returns:
            Formatted context string
        """
        if result["status"] != "success":
            return f"Error: {result.get('error', 'Unknown error')}"
        
        context = f"=== Web Search Results ===\n"
        context += f"Query: {result['query']}\n"
        context += f"Found {result['num_results']} results\n\n"
        
        for i, res in enumerate(result["results"], 1):
            context += f"[{i}] {res['title']}\n"
            context += f"URL: {res['url']}\n"
            context += f"Snippet: {res['snippet']}\n\n"
        
        return context


class TranslationTool:
    """Translation tool"""
    
    def __init__(self, llm_client):
        """
        Initialize translation tool
        
        Args:
            llm_client: LLM client (for translation)
        """
        self.name = "TranslationTool"
        self.description = "Translate text to Chinese"
        self.llm_client = llm_client
    
    def execute(self, text: str, target_lang: str = "Chinese") -> Dict[str, Any]:
        """
        Execute translation
        
        Args:
            text: Text to translate
            target_lang: Target language
        
        Returns:
            Translation result dictionary
        """
        try:
            prompt = f"""Please translate the following text to {target_lang}. 
Keep medical terminology accurate and maintain the original meaning.

Text to translate:
{text}

Translation:"""
            
            response = self.llm_client.chat.completions.create(
                model="Qwen/Qwen2.5-7B-Instruct:together",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            translation = response.choices[0].message.content.strip()
            
            return {
                "tool": self.name,
                "status": "success",
                "original_text": text,
                "translated_text": translation,
                "target_language": target_lang
            }
        
        except Exception as e:
            return {
                "tool": self.name,
                "status": "error",
                "error": str(e)
            }
    
    def get_structured_context(self, result: Dict[str, Any]) -> str:
        """
        Get structured context
        
        Args:
            result: Translation result
        
        Returns:
            Formatted context string
        """
        if result["status"] != "success":
            return f"Error: {result.get('error', 'Unknown error')}"
        
        context = f"=== Translation Result ===\n"
        context += f"Original ({len(result['original_text'])} chars):\n"
        context += f"{result['original_text'][:200]}...\n\n"
        context += f"Translated ({result['target_language']}):\n"
        context += f"{result['translated_text']}\n"
        
        return context

