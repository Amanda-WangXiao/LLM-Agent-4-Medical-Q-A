from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from typing import List
import traceback
import os
from dotenv import load_dotenv
from query_data import setup_qa_chain

# Load environment variables
load_dotenv()

# Set up the QA chain
qa_chain = setup_qa_chain()

# Set up the search tool
search = DuckDuckGoSearchAPIWrapper()

def is_medical_question(query: str) -> bool:
    # simple judge
    medical_keywords = ['disease', 'symptom', 'treatment', 'medicine', 'doctor', 'hospital', 'health', 'medical', 'pain']
    return any(keyword in query.lower() for keyword in medical_keywords)

def run_agent(query: str):
    try:
        print(f"Query: {query}")

        if is_medical_question(query):
            print("Using MedicalQA for this query.")
            # MedicalQA
            result = qa_chain({"query": query})
            print("MedicalQA result:", result)
        else:
            print("Using WebSearch for this query.")
            # WebSearch
            result = search.run(query)
            print("WebSearch result:", result)

        sources = []
        if isinstance(result, dict):    # dictionary (for MedicalQA)
            if 'result' in result:
                answer = result['result']
            elif 'answer' in result:
                answer = result['answer']
            else:
                answer = str(result)

            if 'source_documents' in result:
                for doc in result['source_documents']:
                    source = doc.page_content[:200] + "..."
                    sources.append(source)
                    print(f"Source: {source}")
        elif isinstance(result, str):   # string (for WebSearch or simple MedicalQA response)
            answer = result
            sources.append(result[:200] + "...")
        else:
            answer = str(result)

        print("Answer:", answer)
        print("Sources found:", len(sources))

        return {"answer": answer, "sources": sources}
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        traceback.print_exc()
        return {
            "answer": "I'm sorry, I encountered an error while processing your request. Could you please try rephrasing your question?",
            "sources": []
        }

# Main
if __name__ == "__main__":
    while True:
        user_input = input("Ask a question (or type 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
        response = run_agent(user_input)
        print("Agent's response:", response["answer"])
        if response["sources"]:
            print("Sources:")
            for source in response["sources"]:
                print(f"- {source}")
        print("\n")