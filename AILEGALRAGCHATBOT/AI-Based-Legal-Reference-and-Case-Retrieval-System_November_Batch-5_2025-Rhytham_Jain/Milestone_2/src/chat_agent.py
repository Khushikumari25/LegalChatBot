#!/usr/bin/env python3
# Milestone 2/src/chat_agent.py
"""
Interactive Chat Agent for Legal RAG System
Provides ChatGPT-style terminal interface with conversation memory
"""

import sys
import os
from pathlib import Path
from typing import Optional

# Import existing RAG pipeline
try:
    from rag_pipeline import LegalRAGPipeline
    from feedback_handler import FeedbackHandler
except ImportError:
    src_path = Path(__file__).parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from rag_pipeline import LegalRAGPipeline
    from feedback_handler import FeedbackHandler


class LegalChatAgent:
    """
    Interactive chat agent wrapping the Legal RAG Pipeline
    Provides terminal-based conversational interface
    """
    
    def __init__(self):
        """Initialize chat agent with RAG pipeline"""
        print("\n" + "="*80)
        print("🏛️  LEGAL RAG CHAT AGENT")
        print("="*80)
        print("\nInitializing system...")
        
        # Initialize RAG pipeline with conversation memory enabled
        self.pipeline = LegalRAGPipeline(use_conversation_memory=True)
        
        # Initialize feedback handler
        self.feedback_handler = FeedbackHandler()
        
        # Session info
        self.session_id = self.pipeline.memory.session_id if self.pipeline.memory else "no-session"
        
        print("\n✅ System ready!")
        print(f"📝 Session ID: {self.session_id}")
        print("="*80)
    
    def print_welcome(self):
        """Print welcome message with instructions"""
        print("\n" + "="*80)
        print("💬 INTERACTIVE LEGAL ASSISTANT")
        print("="*80)
        print("\nI can help answer questions about Indian law based on legal documents.")
        print("\nCommands:")
        print("  • Type your legal question and press Enter")
        print("  • Type 'rate <1-5>' to rate the last response")
        print("  • Type 'sources' to see the last retrieved sources")
        print("  • Type 'history' to view conversation history")
        print("  • Type 'stats' to see usage statistics")
        print("  • Type 'clear' to start a new conversation")
        print("  • Type 'exit' or 'quit' to end session")
        print("\n⚠️  Disclaimer: This is for informational purposes only.")
        print("    Always consult a qualified legal counsel for legal advice.")
        print("="*80 + "\n")
    
    def format_answer(self, response: dict) -> str:
        """Format the answer for display"""
        if not response.get('success'):
            return f"\n❌ Error: {response.get('error', 'Unknown error')}\n"
        
        output = []
        
        # Main answer
        output.append("\n" + "="*80)
        output.append("📋 ANSWER")
        output.append("="*80)
        output.append(response['answer'])
        
        # Sources
        if response.get('sources'):
            output.append("\n" + "-"*80)
            output.append("📚 SOURCES")
            output.append("-"*80)
            
            for i, source in enumerate(response['sources'], 1):
                output.append(f"\n{i}. {source['source_file']}")
                output.append(f"   • Page: {source['page_number']}")
                output.append(f"   • Section: {source['section']}")
                output.append(f"   • Chunk ID: {source['chunk_id']}")
                output.append(f"   • Preview: {source['text_preview'][:150]}...")
        
        # Metadata
        metadata = response.get('metadata', {})
        output.append("\n" + "-"*80)
        output.append(f"📊 Metadata: {metadata.get('retrieval_count', 0)} chunks retrieved | "
                     f"{metadata.get('tokens_used', 0)} tokens | "
                     f"Model: {metadata.get('model', 'unknown')}")
        output.append("="*80 + "\n")
        
        return "\n".join(output)
    
    def handle_command(self, user_input: str) -> Optional[str]:
        """
        Handle special commands
        
        Returns:
            Command output string, or None if not a command
        """
        cmd = user_input.strip().lower()
        
        # Exit commands
        if cmd in ['exit', 'quit', 'bye']:
            return "EXIT"
        
        # Clear history
        if cmd == 'clear':
            if self.pipeline.memory:
                self.pipeline.memory.clear_history()
                return "✅ Conversation history cleared. Starting fresh!\n"
            return "⚠️  Conversation memory not enabled.\n"
        
        # Show history
        if cmd == 'history':
            if self.pipeline.memory:
                history = self.pipeline.memory.get_conversation_history()
                if not history:
                    return "📝 No conversation history yet.\n"
                
                output = ["\n" + "="*80, "📜 CONVERSATION HISTORY", "="*80]
                for i, entry in enumerate(history, 1):
                    role = "You" if entry['role'] == 'user' else "Assistant"
                    output.append(f"\n{i}. {role}: {entry['content'][:100]}...")
                output.append("="*80 + "\n")
                return "\n".join(output)
            return "⚠️  Conversation memory not enabled.\n"
        
        # Show stats
        if cmd == 'stats':
            stats = self.pipeline.get_usage_stats()
            output = ["\n" + "="*80, "📊 USAGE STATISTICS", "="*80]
            
            llm_stats = stats.get('llm_stats', {})
            output.append(f"\n🤖 LLM Usage:")
            output.append(f"   • Total calls: {llm_stats.get('total_calls', 0)}")
            output.append(f"   • Total tokens: {llm_stats.get('total_tokens', 0)}")
            output.append(f"   • Estimated cost: ${llm_stats.get('estimated_cost_usd', 0):.4f}")
            output.append(f"   • Model: {llm_stats.get('model', 'unknown')}")
            
            if 'conversation_stats' in stats:
                conv_stats = stats['conversation_stats']
                output.append(f"\n💬 Conversation:")
                output.append(f"   • Session ID: {conv_stats.get('session_id', 'unknown')}")
                output.append(f"   • Total messages: {conv_stats.get('total_messages', 0)}")
                output.append(f"   • Total turns: {conv_stats.get('total_turns', 0)}")
            
            output.append("="*80 + "\n")
            return "\n".join(output)
        
        # Show last sources
        if cmd == 'sources':
            if hasattr(self, 'last_response') and self.last_response.get('sources'):
                output = ["\n" + "="*80, "📚 LAST RETRIEVED SOURCES", "="*80]
                for i, source in enumerate(self.last_response['sources'], 1):
                    output.append(f"\n{i}. {source['source_file']}")
                    output.append(f"   • Page: {source['page_number']}")
                    output.append(f"   • Section: {source['section']}")
                    output.append(f"   • Chunk ID: {source['chunk_id']}")
                output.append("="*80 + "\n")
                return "\n".join(output)
            return "⚠️  No sources available yet. Ask a question first!\n"
        
        # Rate command
        if cmd.startswith('rate '):
            try:
                rating = int(cmd.split()[1])
                if rating < 1 or rating > 5:
                    return "⚠️  Rating must be between 1 and 5.\n"
                
                if hasattr(self, 'last_response') and hasattr(self, 'last_query'):
                    feedback_id = self.feedback_handler.add_rating(
                        query=self.last_query,
                        answer=self.last_response.get('answer', ''),
                        rating=rating,
                        sources_count=len(self.last_response.get('sources', [])),
                        session_id=self.session_id
                    )
                    return f"✅ Thank you for rating! Feedback ID: {feedback_id}\n"
                return "⚠️  No previous response to rate. Ask a question first!\n"
            except (ValueError, IndexError):
                return "⚠️  Invalid rating format. Use: rate <1-5>\n"
        
        # Help command
        if cmd in ['help', '?']:
            return """
Available commands:
  • rate <1-5>  - Rate the last response
  • sources     - Show last retrieved sources
  • history     - View conversation history
  • stats       - Show usage statistics
  • clear       - Start new conversation
  • exit/quit   - End session
  • help/?      - Show this help message

Just type your legal question to get started!
"""
        
        return None  # Not a command
    
    def chat_loop(self):
        """Main interactive chat loop"""
        self.print_welcome()
        
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                cmd_result = self.handle_command(user_input)
                if cmd_result == "EXIT":
                    break
                elif cmd_result:
                    print(cmd_result)
                    continue
                
                # Process query through RAG pipeline
                print("\n⏳ Processing query...\n")
                
                response = self.pipeline.query(
                    user_query=user_input,
                    top_k=5,
                    include_sources=True
                )
                
                # Store for rating/sources commands
                self.last_query = user_input
                self.last_response = response
                
                # Display formatted answer
                formatted = self.format_answer(response)
                print(formatted)
                
            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted by user.")
                break
            except EOFError:
                print("\n\n⚠️  End of input.")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again or type 'exit' to quit.\n")
        
        # Save session on exit
        self.cleanup()
    
    def cleanup(self):
        """Cleanup and save session"""
        print("\n" + "="*80)
        print("💾 SAVING SESSION")
        print("="*80)
        
        # Save conversation
        saved_path = self.pipeline.save_session()
        if saved_path:
            print(f"✅ Conversation saved: {saved_path}")
        
        # Show final stats
        stats = self.pipeline.get_usage_stats()
        llm_stats = stats.get('llm_stats', {})
        
        print(f"\n📊 Session Summary:")
        print(f"   • Total queries: {llm_stats.get('total_calls', 0)}")
        print(f"   • Total tokens: {llm_stats.get('total_tokens', 0)}")
        print(f"   • Estimated cost: ${llm_stats.get('estimated_cost_usd', 0):.4f}")
        
        if 'conversation_stats' in stats:
            conv_stats = stats['conversation_stats']
            print(f"   • Messages exchanged: {conv_stats.get('total_messages', 0)}")
        
        print("\n👋 Thank you for using Legal RAG Chat Agent!")
        print("="*80 + "\n")


def main():
    """Main entry point"""
    try:
        agent = LegalChatAgent()
        agent.chat_loop()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()