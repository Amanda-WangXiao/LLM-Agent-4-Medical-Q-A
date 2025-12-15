"""
Planner Module
Task planner: analyzes task complexity, creates execution steps, tool selection strategy
"""
import os
import json
from typing import Dict, Any, List, Optional
from enum import Enum
from openai import OpenAI


class ToolStrategy(Enum):
    """Tool selection strategy"""
    RAG_ONLY = "RAG_ONLY"  # Use RAG only
    WEB_ONLY = "WEB_ONLY"  # Use web search only
    HYBRID = "HYBRID"  # Hybrid use of RAG and web search


class Planner:
    """Task planner"""
    
    def __init__(self, llm_client: OpenAI):
        """
        Initialize planner
        
        Args:
            llm_client: LLM client
        """
        self.llm_client = llm_client
    
    def analyze_task_complexity(self, query: str, context: str = "") -> Dict[str, Any]:
        """
        Analyze task complexity
        
        Args:
            query: User query
            context: Conversation context
        
        Returns:
            Dictionary containing complexity analysis results
        """
        prompt = f"""Analyze the complexity of the following medical query and determine:
1. Task complexity (SIMPLE, MODERATE, COMPLEX)
2. Whether it needs RAG search (local knowledge base)
3. Whether it needs web search (latest information)
4. Suggested execution steps

Query: {query}
Context: {context if context else "No previous context"}

Respond in JSON format:
{{
    "complexity": "SIMPLE|MODERATE|COMPLEX",
    "needs_rag": true/false,
    "needs_web": true/false,
    "reasoning": "brief explanation",
    "steps": ["step1", "step2", ...]
}}"""
        
        try:
            response = self.llm_client.chat.completions.create(
                model="Qwen/Qwen2.5-7B-Instruct:together",
                messages=[
                    {"role": "system", "content": "You are a medical task planner. Analyze queries and suggest execution strategies."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Try to parse JSON (simple handling)
            try:
                # Extract JSON part
                if "```json" in result_text:
                    result_text = result_text.split("```json")[1].split("```")[0].strip()
                elif "```" in result_text:
                    result_text = result_text.split("```")[1].split("```")[0].strip()
                
                analysis = json.loads(result_text)
            except:
                # If parsing fails, use default values
                analysis = self._default_analysis(query)
            
            return analysis
        
        except Exception as e:
            print(f"Error in task analysis: {e}")
            return self._default_analysis(query)
    
    def _default_analysis(self, query: str) -> Dict[str, Any]:
        """Default analysis result"""
        query_lower = query.lower()
        
        # Simple heuristic rules
        needs_web = any(keyword in query_lower for keyword in [
            "latest", "recent", "update", "2024", "new", "current"
        ])
        
        needs_rag = True  # Default requires RAG
        
        complexity = "MODERATE"
        if len(query.split()) < 5:
            complexity = "SIMPLE"
        elif any(keyword in query_lower for keyword in [
            "compare", "difference", "analysis", "evaluate"
        ]):
            complexity = "COMPLEX"
        
        return {
            "complexity": complexity,
            "needs_rag": needs_rag,
            "needs_web": needs_web,
            "reasoning": "Default analysis based on keywords",
            "steps": self._generate_steps(needs_rag, needs_web)
        }
    
    def _generate_steps(self, needs_rag: bool, needs_web: bool) -> List[str]:
        """Generate execution steps"""
        steps = []
        
        if needs_rag:
            steps.append("Search local medical knowledge base (RAG)")
        
        if needs_web:
            steps.append("Search web for latest information")
        
        steps.append("Synthesize information from all sources")
        steps.append("Provide comprehensive answer")
        
        return steps
    
    def determine_strategy(
        self,
        needs_rag: bool,
        needs_web: bool
    ) -> ToolStrategy:
        """
        Determine tool selection strategy
        
        Args:
            needs_rag: Whether RAG is needed
            needs_web: Whether web search is needed
        
        Returns:
            Tool selection strategy
        """
        if needs_rag and needs_web:
            return ToolStrategy.HYBRID
        elif needs_rag:
            return ToolStrategy.RAG_ONLY
        elif needs_web:
            return ToolStrategy.WEB_ONLY
        else:
            return ToolStrategy.RAG_ONLY  # Default to RAG
    
    def create_execution_plan(
        self,
        query: str,
        context: str = ""
    ) -> Dict[str, Any]:
        """
        Create execution plan
        
        Args:
            query: User query
            context: Conversation context
        
        Returns:
            Execution plan dictionary
        """
        # Analyze task
        analysis = self.analyze_task_complexity(query, context)
        
        # Determine strategy
        strategy = self.determine_strategy(
            analysis["needs_rag"],
            analysis["needs_web"]
        )
        
        # Create plan
        plan = {
            "query": query,
            "complexity": analysis["complexity"],
            "strategy": strategy.value,
            "needs_rag": analysis["needs_rag"],
            "needs_web": analysis["needs_web"],
            "reasoning": analysis["reasoning"],
            "steps": analysis["steps"],
            "tool_sequence": self._generate_tool_sequence(strategy)
        }
        
        return plan
    
    def _generate_tool_sequence(self, strategy: ToolStrategy) -> List[str]:
        """Generate tool execution sequence"""
        if strategy == ToolStrategy.RAG_ONLY:
            return ["RAGSearchTool"]
        elif strategy == ToolStrategy.WEB_ONLY:
            return ["WebSearchTool"]
        else:  # HYBRID
            return ["RAGSearchTool", "WebSearchTool"]

