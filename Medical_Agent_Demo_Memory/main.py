"""
Medical Agentic System - Main Entry
A concise medical agent system demo: RAG + Web + Planner + Tools
"""
import os
from dotenv import load_dotenv
from openai import OpenAI

# Import all modules
from memory import ConversationMemory
from rag import RAGSystem
from tools import RAGSearchTool, WebSearchTool, TranslationTool
from planner import Planner
from executor import ReActExecutor


def initialize_system():
    """Initialize all system components"""
    print("=" * 60)
    print("Medical Agentic System - Initializing...")
    print("=" * 60)
    
    # Load environment variables
    load_dotenv()
    
    # Check HF_TOKEN
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("⚠️  Warning: HF_TOKEN not found in environment variables")
        print("Please set HF_TOKEN in .env file or environment")
        hf_token = input("Enter your HuggingFace token (or press Enter to skip): ").strip()
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
    
    # Initialize LLM client
    llm_client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=os.environ.get("HF_TOKEN", "")
    )
    print("✓ LLM Client initialized")
    
    # Initialize Memory
    memory = ConversationMemory()
    print("✓ Conversation Memory initialized")
    
    # Initialize RAG system
    knowledge_file = "ref/medical_knowledge.txt"
    print(f"Initializing RAG system with: {knowledge_file}")
    rag_system = RAGSystem(knowledge_file)
    print("✓ RAG System initialized")
    
    # Initialize Tools
    rag_tool = RAGSearchTool(rag_system)
    web_tool = WebSearchTool(max_results=5)
    translation_tool = TranslationTool(llm_client)
    print("✓ Tools initialized (RAGSearchTool, WebSearchTool, TranslationTool)")
    
    # Initialize Planner
    planner = Planner(llm_client)
    print("✓ Planner initialized")
    
    # Initialize Executor
    executor = ReActExecutor(
        llm_client=llm_client,
        rag_tool=rag_tool,
        web_tool=web_tool,
        translation_tool=translation_tool,
        memory=memory,
        planner=planner,
        max_iterations=5
    )
    print("✓ ReAct Executor initialized")
    
    print("=" * 60)
    print("System initialization complete!")
    print("=" * 60)
    
    return executor, memory


def print_system_info():
    """Print system information"""
    print("\n" + "=" * 60)
    print("Medical Agentic System - System Architecture")
    print("=" * 60)
    print("""
📦 Modules:
  1. Memory: ConversationMemory (context coherence)
  2. RAG: ChromaDB + Retrieval + Rerank + Summarization
  3. Planner: Task complexity analysis + tool selection strategy
  4. Tools: RAGSearchTool, WebSearchTool, TranslationTool
  5. Execution Loop: ReAct (Thought → Action → Observation)
  6. State Management: Tool call tracking + step sequence

🔄 Execution Flow:
  Query → Planner → Tool Selection → ReAct Loop → Final Answer
  
🛠 Tool Strategies:
  - RAG_ONLY: Use local knowledge base only
  - WEB_ONLY: Use web search only
  - HYBRID: Hybrid use of RAG and web search
    """)


def interactive_mode(executor: ReActExecutor, memory: ConversationMemory):
    """Interactive mode"""
    print("\n" + "=" * 60)
    print("Interactive Mode - Enter your medical queries")
    print("Type 'quit' or 'exit' to exit")
    print("Type 'clear' to clear conversation history")
    print("Type 'info' to show system information")
    print("=" * 60)
    
    while True:
        try:
            query = input("\n👤 User: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if query.lower() == 'clear':
                memory.clear()
                print("✓ Conversation history cleared")
                continue
            
            if query.lower() == 'info':
                print_system_info()
                continue
            
            # Execute query
            print("\n🤖 Assistant: Processing...")
            result = executor.execute(query, translate_to_chinese=True)
            
            if result["status"] == "success":
                # The executor has already output all processes and final answer, here we only show execution summary (optional)
                if os.getenv("SHOW_TRACE", "false").lower() == "true":
                    print("\n" + "-" * 60)
                    print("Execution Summary:")
                    summary = result["state"]
                    print(f"  Strategy: {summary.get('plan_strategy')}")
                    print(f"  Tool Calls: {summary.get('total_tool_calls')}")
                    print(f"  Duration: {summary.get('duration_seconds', 0):.2f}s")
                    print("-" * 60)
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()


def main():
    
    # Initialize system
    executor, memory = initialize_system()
    
    # Enter interactive mode directly
    interactive_mode(executor, memory)


if __name__ == "__main__":
    main()
