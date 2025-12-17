# Milestone 2/src/conversation_memory.py
"""
Conversation Memory Management
Tracks chat history for context-aware follow-up questions
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import uuid


class ConversationMemory:
    """
    Manages conversation history for context-aware responses
    """
    
    def __init__(self, output_dir: str = None):
        """
        Initialize conversation memory
        
        Args:
            output_dir: Directory to save conversation logs
        """
        # Fix: Use relative path from src directory to Milestone 2/outputs
        if output_dir is None:
            # Get the directory where this script is located (src folder)
            script_dir = Path(__file__).parent
            # Go up one level to Milestone 2, then into outputs/conversation_logs
            output_dir = script_dir.parent / "outputs" / "conversation_logs"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Current conversation
        self.session_id = str(uuid.uuid4())[:8]
        self.conversation_history: List[Dict] = []
        self.metadata = {
            "session_id": self.session_id,
            "start_time": datetime.now().isoformat(),
            "total_turns": 0
        }
        
        print(f"✅ Conversation session started: {self.session_id}")
    
    def add_user_message(self, message: str, query_metadata: Optional[Dict] = None):
        """
        Add user message to history
        
        Args:
            message: User's query
            query_metadata: Optional metadata (e.g., retrieved chunks info)
        """
        entry = {
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat(),
            "metadata": query_metadata or {}
        }
        
        self.conversation_history.append(entry)
        self.metadata["total_turns"] += 1
    
    def add_assistant_message(self, message: str, response_metadata: Optional[Dict] = None):
        """
        Add assistant's response to history
        
        Args:
            message: Assistant's answer
            response_metadata: Optional metadata (tokens, sources, etc.)
        """
        entry = {
            "role": "assistant",
            "content": message,
            "timestamp": datetime.now().isoformat(),
            "metadata": response_metadata or {}
        }
        
        self.conversation_history.append(entry)
    
    def get_conversation_history(self, 
                                 last_n_turns: Optional[int] = None,
                                 format_for_llm: bool = False) -> List[Dict]:
        """
        Get conversation history
        
        Args:
            last_n_turns: Get only last N conversation turns (None = all)
            format_for_llm: If True, returns in OpenAI message format
        
        Returns:
            List of conversation entries
        """
        history = self.conversation_history
        
        # Limit to last N turns if specified
        if last_n_turns:
            history = history[-(last_n_turns * 2):]  # *2 because each turn = user + assistant
        
        if format_for_llm:
            # Return in OpenAI format (without metadata)
            return [
                {"role": entry["role"], "content": entry["content"]}
                for entry in history
            ]
        
        return history
    
    def get_conversation_context(self, max_chars: int = 1000) -> str:
        """
        Get conversation history as formatted string for context
        
        Args:
            max_chars: Maximum characters to include
        
        Returns:
            Formatted conversation string
        """
        if not self.conversation_history:
            return ""
        
        context_parts = []
        total_chars = 0
        
        # Go backwards through history
        for entry in reversed(self.conversation_history):
            role = "User" if entry["role"] == "user" else "Assistant"
            content = entry["content"]
            
            line = f"{role}: {content}"
            
            if total_chars + len(line) > max_chars:
                break
            
            context_parts.insert(0, line)
            total_chars += len(line)
        
        return "\n".join(context_parts)
    
    def save_conversation(self, filename: Optional[str] = None):
        """
        Save conversation to JSON file
        
        Args:
            filename: Custom filename (default: session_id.json)
        
        Returns:
            Path to saved file
        """
        if not filename:
            filename = f"conversation_{self.session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.output_dir / filename
        
        # Update metadata
        self.metadata["end_time"] = datetime.now().isoformat()
        self.metadata["total_messages"] = len(self.conversation_history)
        
        # Save
        data = {
            "metadata": self.metadata,
            "conversation": self.conversation_history
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Conversation saved: {filepath}")
        return filepath
    
    def load_conversation(self, filepath: str):
        """
        Load conversation from file
        
        Args:
            filepath: Path to conversation JSON file
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.metadata = data["metadata"]
        self.conversation_history = data["conversation"]
        self.session_id = self.metadata["session_id"]
        
        print(f"✅ Conversation loaded: {self.session_id}")
        print(f"   Total messages: {len(self.conversation_history)}")
    
    def clear_history(self):
        """Clear conversation history (start fresh)"""
        self.conversation_history = []
        self.metadata["total_turns"] = 0
        print(f"🔄 Conversation history cleared")
    
    def get_summary(self) -> Dict:
        """Get conversation summary"""
        return {
            "session_id": self.session_id,
            "total_messages": len(self.conversation_history),
            "total_turns": self.metadata["total_turns"],
            "start_time": self.metadata.get("start_time"),
            "duration_seconds": (
                (datetime.fromisoformat(self.metadata.get("end_time", datetime.now().isoformat())) -
                 datetime.fromisoformat(self.metadata["start_time"])).total_seconds()
                if "end_time" in self.metadata else 0
            )
        }


# Test function
def test_conversation_memory():
    """Test conversation memory functionality"""
    
    print("\n" + "="*80)
    print("TESTING CONVERSATION MEMORY")
    print("="*80)
    
    # Create memory
    memory = ConversationMemory()
    
    # Simulate conversation
    print("\n🔍 Simulating conversation...")
    
    memory.add_user_message(
        "What is murder under IPC?",
        query_metadata={"retrieved_chunks": 5}
    )
    
    memory.add_assistant_message(
        "Murder is defined under Section 300 of IPC...",
        response_metadata={"tokens_used": 150, "sources": ["IPC_1860.pdf"]}
    )
    
    memory.add_user_message("What is the punishment?")
    
    memory.add_assistant_message(
        "Under Section 302, murder is punishable with death or life imprisonment...",
        response_metadata={"tokens_used": 120, "sources": ["IPC_1860.pdf"]}
    )
    
    # Get history
    print("\n📖 Full conversation history:")
    history = memory.get_conversation_history()
    for i, entry in enumerate(history, 1):
        print(f"\n{i}. {entry['role'].upper()}: {entry['content'][:80]}...")
    
    # Get LLM format
    print("\n🤖 LLM format (last 2 turns):")
    llm_format = memory.get_conversation_history(last_n_turns=1, format_for_llm=True)
    for msg in llm_format:
        print(f"   {msg['role']}: {msg['content'][:60]}...")
    
    # Get context string
    print("\n📄 Context string:")
    context = memory.get_conversation_context(max_chars=200)
    print(f"   {context[:200]}...")
    
    # Save conversation
    print("\n💾 Saving conversation...")
    filepath = memory.save_conversation()
    
    # Show summary
    summary = memory.get_summary()
    print(f"\n📊 Summary:")
    print(f"   Session ID: {summary['session_id']}")
    print(f"   Total messages: {summary['total_messages']}")
    print(f"   Total turns: {summary['total_turns']}")
    
    print("\n✅ Test complete!")


if __name__ == "__main__":
    test_conversation_memory()