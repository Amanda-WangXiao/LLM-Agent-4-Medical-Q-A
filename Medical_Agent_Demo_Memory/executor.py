"""
Execution Loop Module
Execution loop: ReAct pattern (Thought → Action → Observation → Thought ... → Final Answer)
"""
import os
from typing import Dict, Any, Optional
from openai import OpenAI
from memory import ConversationMemory
from planner import Planner, ToolStrategy
from tools import RAGSearchTool, WebSearchTool, TranslationTool
from state import ExecutionState, StepStatus


class ReActExecutor:
    """ReAct executor: implements think-act-observe loop"""
    
    def __init__(
        self,
        llm_client: OpenAI,
        rag_tool: RAGSearchTool,
        web_tool: WebSearchTool,
        translation_tool: TranslationTool,
        memory: ConversationMemory,
        planner: Planner,
        max_iterations: int = 5
    ):
        """
        Initialize ReAct executor
        
        Args:
            llm_client: LLM client
            rag_tool: RAG search tool
            web_tool: Web search tool
            translation_tool: Translation tool
            memory: Conversation memory
            planner: Planner
            max_iterations: Maximum number of iterations
        """
        self.llm_client = llm_client
        self.rag_tool = rag_tool
        self.web_tool = web_tool
        self.translation_tool = translation_tool
        self.memory = memory
        self.planner = planner
        self.max_iterations = max_iterations
        
        # Tool mapping
        self.tool_map = {
            "RAGSearchTool": self.rag_tool,
            "WebSearchTool": self.web_tool,
            "TranslationTool": self.translation_tool
        }
    
    def execute(self, query: str, translate_to_chinese: bool = True) -> Dict[str, Any]:
        """
        Execute query
        
        Args:
            query: User query
            translate_to_chinese: Whether to translate to Chinese
        
        Returns:
            Execution result dictionary
        """
        # Initialize state
        state = ExecutionState()
        
        # Get conversation context
        context = self.memory.get_recent_context(n=3)
        
        # Create execution plan
        plan = self.planner.create_execution_plan(query, context)
        state.start_execution(query, plan)
        
        # Add initial thought
        initial_thought = f"Analyzing query: {query}. Strategy: {plan['strategy']}"
        state.add_thought(initial_thought)
        
        # Output initial thought
        print("\n" + "=" * 60)
        print("🧠 Thought:")
        print(f"  {initial_thought}")
        print(f"  Plan Strategy: {plan['strategy']}")
        print(f"  Tool Sequence: {', '.join(plan['tool_sequence'])}")
        print("=" * 60)
        
        # Execute tool sequence
        all_observations = []
        
        for tool_name in plan["tool_sequence"]:
            if tool_name not in self.tool_map:
                continue
            
            tool = self.tool_map[tool_name]
            state.add_step(f"Execute {tool_name}", StepStatus.IN_PROGRESS)
            
            # Thought: Why use this tool
            thought = f"I need to use {tool_name} to gather information about: {query}"
            state.add_thought(thought)
            
            # Output thought
            print(f"\n🧠 Thought: {thought}")
            
            # Action: Execute tool
            print(f"🔧 Action: Executing {tool_name}...")
            if tool_name == "RAGSearchTool":
                tool_output = tool.execute(query, top_k=5)
            elif tool_name == "WebSearchTool":
                tool_output = tool.execute(query, max_results=5)
            else:
                continue
            
            # Record tool call
            state.add_tool_call(
                tool_name=tool_name,
                tool_input={"query": query},
                tool_output=tool_output,
                status=tool_output.get("status", "success")
            )
            
            # Observation: Get tool output
            if tool_output.get("status") == "success":
                observation = tool.get_structured_context(tool_output)
                state.add_observation(observation)
                all_observations.append(observation)
                state.update_step_status(len(state.steps) - 1, StepStatus.COMPLETED)
                
                # Output observation summary
                print(f"👁️  Observation: {tool_name} completed successfully")
                # Show first 200 characters of observation as preview
                obs_preview = observation[:200] + "..." if len(observation) > 200 else observation
                print(f"   Preview: {obs_preview}")
            else:
                state.update_step_status(len(state.steps) - 1, StepStatus.FAILED)
                print(f"❌ Observation: {tool_name} failed")
                print(f"   Error: {tool_output.get('error', 'Unknown error')}")
        
        # Check if fallback is needed
        if state.check_fallback():
            return {
                "status": "error",
                "error": state.fallback_reason,
                "state": state.get_summary()
            }
        
        # Thought: Synthesize information
        synthesis_thought = "I have gathered information from all sources. Now I need to synthesize and provide a comprehensive answer."
        state.add_thought(synthesis_thought)
        print(f"\n🧠 Thought: {synthesis_thought}")
        
        # Generate final answer
        print("\n" + "=" * 60)
        print("📝 Generating Final Answer...")
        print("=" * 60)
        final_answer = self._generate_final_answer(
            query=query,
            observations=all_observations,
            context=context,
            plan=plan
        )
        
        # Output final answer (English)
        print("\n" + "=" * 60)
        print("📄 Final Answer (English):")
        print("=" * 60)
        print(final_answer)
        
        # Translate (if needed) - append Chinese translation after English answer
        if translate_to_chinese:
            state.add_step("Translate to Chinese", StepStatus.IN_PROGRESS)
            print("\n" + "=" * 60)
            print("🌐 Translating to Chinese...")
            print("=" * 60)
            # Save English answer for recording
            english_answer = final_answer
            translation_result = self.translation_tool.execute(english_answer)
            if translation_result.get("status") == "success":
                chinese_translation = translation_result["translated_text"]
                # Append Chinese translation after English answer
                final_answer = f"{english_answer}\n\nTranslation:\n{chinese_translation}"
                state.add_tool_call(
                    tool_name="TranslationTool",
                    tool_input={"text": english_answer[:100] + "..."},
                    tool_output=translation_result
                )
                print("✓ Translation completed")
                # Output complete answer (including translation)
                print("\n" + "=" * 60)
                print("✅ Complete Response (English + Translation):")
                print("=" * 60)
                print(final_answer)
            else:
                print(f"❌ Translation failed: {translation_result.get('error', 'Unknown error')}")
            state.update_step_status(len(state.steps) - 1, StepStatus.COMPLETED)
        
        # Finish execution
        state.finish_execution(final_answer)
        
        # Update memory
        self.memory.add_user_message(query)
        self.memory.add_ai_message(final_answer)
        
        return {
            "status": "success",
            "final_answer": final_answer,
            "state": state.get_summary(),
            "execution_trace": state.get_execution_trace()
        }
    
    def _generate_final_answer(
        self,
        query: str,
        observations: list,
        context: str,
        plan: Dict[str, Any]
    ) -> str:
        """
        Generate final answer
        
        Args:
            query: User query
            observations: All observations
            context: Conversation context
            plan: Execution plan
        
        Returns:
            Final answer
        """
        # Build prompt
        observations_text = "\n\n".join(observations)
        
        prompt = f"""You are a medical information assistant. Based on the following information, provide a comprehensive and accurate answer to the user's query.

User Query: {query}

Context from previous conversation:
{context if context else "No previous context"}

Information gathered:
{observations_text}

Instructions:
1. Synthesize information from all sources
2. Provide a clear, structured answer
3. Cite sources when possible
4. If information conflicts, mention the differences

Answer:"""
        
        try:
            response = self.llm_client.chat.completions.create(
                model="Qwen/Qwen2.5-7B-Instruct:together",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful medical information assistant."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            return f"Error generating answer: {str(e)}"

