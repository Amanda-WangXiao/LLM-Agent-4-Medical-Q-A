"""
ConversationMemory Module
Manages conversation context to maintain conversation coherence
Simplified implementation, no dependency on LangChain
"""
from typing import List, Dict, Any


class ConversationMemory:
    """Conversation memory management class"""
    
    def __init__(self):
        """Initialize conversation memory"""
        self.conversation_history: List[Dict[str, str]] = []
    
    def add_user_message(self, message: str) -> None:
        """Add user message"""
        self.conversation_history.append({"role": "user", "content": message})
    
    def add_ai_message(self, message: str) -> None:
        """Add AI message"""
        self.conversation_history.append({"role": "assistant", "content": message})
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history (dictionary format)"""
        return self.conversation_history
    
    def get_conversation_string(self) -> str:
        """Get conversation history in string format"""
        history_str = ""
        for msg in self.conversation_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_str += f"{role}: {msg['content']}\n\n"
        return history_str
    
    def get_recent_context(self, n: int = 3) -> str:
        """Get context from the most recent n rounds of conversation"""
        recent = self.conversation_history[-n*2:] if len(self.conversation_history) > n*2 else self.conversation_history
        return self._format_messages(recent)
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format message list"""
        formatted = ""
        for msg in messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted += f"{role}: {msg['content']}\n\n"
        return formatted
    
    def clear(self) -> None:
        """Clear conversation history"""
        self.conversation_history = []
    
    def get_memory_variables(self) -> Dict[str, Any]:
        """Get memory variables (compatibility interface)"""
        return {
            "chat_history": self.get_conversation_string()
        }

