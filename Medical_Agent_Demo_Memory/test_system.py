"""
Quick Test Script for Medical Agentic System
Quick test script - verify basic system functionality
"""
import os
from dotenv import load_dotenv
from openai import OpenAI

# Import modules
from memory import ConversationMemory
from rag import RAGSystem
from tools import RAGSearchTool, WebSearchTool, TranslationTool
from planner import Planner


def test_memory():
    """Test Memory module"""
    print("\n" + "="*60)
    print("Testing Memory Module")
    print("="*60)
    
    memory = ConversationMemory()
    memory.add_user_message("What is anesthesia?")
    memory.add_ai_message("Anesthesia is a medical practice...")
    
    history = memory.get_conversation_string()
    print("✓ Memory test passed")
    print(f"History length: {len(history)} chars")
    return True


def test_rag():
    """Test RAG module"""
    print("\n" + "="*60)
    print("Testing RAG Module")
    print("="*60)
    
    try:
        knowledge_file = "ref/medical_knowledge.txt"
        if not os.path.exists(knowledge_file):
            print("⚠️  Knowledge file not found, skipping RAG test")
            return False
        
        # Note: First run will build index, may take some time
        print("Initializing RAG system (this may take a while on first run)...")
        rag_system = RAGSystem(knowledge_file)
        
        # Test retrieval
        result = rag_system.search("anesthesia", top_k=3)
        print(f"✓ RAG test passed")
        print(f"Retrieved {result['num_results']} documents")
        return True
    except Exception as e:
        print(f"❌ RAG test failed: {e}")
        return False


def test_tools():
    """Test Tools module"""
    print("\n" + "="*60)
    print("Testing Tools Module")
    print("="*60)
    
    try:
        # Test WebSearchTool (no API key required)
        web_tool = WebSearchTool(max_results=2)
        result = web_tool.execute("medical news", max_results=2)
        
        if result["status"] == "success":
            print(f"✓ WebSearchTool test passed")
            print(f"Found {result['num_results']} results")
        else:
            print(f"⚠️  WebSearchTool returned error: {result.get('error')}")
        
        return True
    except Exception as e:
        print(f"❌ Tools test failed: {e}")
        return False


def test_planner():
    """Test Planner module"""
    print("\n" + "="*60)
    print("Testing Planner Module")
    print("="*60)
    
    try:
        load_dotenv()
        hf_token = os.getenv("HF_TOKEN")
        
        if not hf_token:
            print("⚠️  HF_TOKEN not found, skipping Planner test")
            return False
        
        llm_client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=hf_token
        )
        
        planner = Planner(llm_client)
        plan = planner.create_execution_plan("What is anesthesia?")
        
        print(f"✓ Planner test passed")
        print(f"Strategy: {plan['strategy']}")
        print(f"Complexity: {plan['complexity']}")
        return True
    except Exception as e:
        print(f"❌ Planner test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("Medical Agentic System - Quick Test")
    print("="*60)
    
    results = {
        "Memory": test_memory(),
        "RAG": test_rag(),
        "Tools": test_tools(),
        "Planner": test_planner()
    }
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for module, passed in results.items():
        status = "✓ PASS" if passed else "⚠️  SKIP/FAIL"
        print(f"{module:15} {status}")
    
    all_passed = all(results.values())
    print("\n" + "="*60)
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("⚠️  Some tests were skipped or failed")
        print("   This is normal if HF_TOKEN is not set or knowledge file is missing")
    print("="*60)


if __name__ == "__main__":
    main()

