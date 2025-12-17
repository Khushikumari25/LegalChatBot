# Milestone 2/src/rag_pipeline.py
"""
Complete RAG Pipeline for Legal Queries
Integrates with Milestone 1's Pinecone setup and uses OpenAI LLM
"""
from pinecone import Pinecone
import os
import sys
from pathlib import Path
from typing import Dict, Optional, List
from dotenv import load_dotenv

# Load environment
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=env_path)

# Pinecone and LangChain imports

from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings

# Import local modules
try:
    from system_prompts import get_system_prompt, get_prompt_with_context
    from llm_integration import LegalLLM
    from conversation_memory import ConversationMemory
    from response_formatter import ResponseFormatter
except ImportError:
    # If running from different location, try adding src to path
    src_path = Path(__file__).parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from system_prompts import get_system_prompt, get_prompt_with_context
    from llm_integration import LegalLLM
    from conversation_memory import ConversationMemory
    from response_formatter import ResponseFormatter


class LegalRAGPipeline:
    """
    Complete RAG pipeline for legal queries
    Retrieves relevant documents and generates accurate responses using OpenAI
    """
    
    def __init__(self, use_conversation_memory: bool = True):
        """
        Initialize RAG pipeline
        
        Args:
            use_conversation_memory: Enable conversation history tracking
        """
        print("\n" + "="*80)
        print("🚀 INITIALIZING LEGAL RAG PIPELINE")
        print("="*80)
        
        # Get index name from environment
        self.index_name = "legal-rag"
        # Initialize components
        self._init_pinecone()
        self._init_vectorstore()
        self._init_retriever()
        self._init_llm()
        
        # Initialize formatters
        self.formatter = ResponseFormatter()
        
        # Initialize conversation memory
        self.use_memory = use_conversation_memory
        self.memory = ConversationMemory() if use_conversation_memory else None
        
        print("\n✅ RAG Pipeline initialized successfully!")
        print("="*80 + "\n")
    
    def _init_pinecone(self):
        """Initialize Pinecone connection"""
        print("\n1️⃣ Connecting to Pinecone...")
        
        api_key = os.getenv('PINECONE_API_KEY')
        if not api_key:
            raise ValueError("❌ PINECONE_API_KEY not found in .env")
        
        # Initialize Pinecone client
        pc = Pinecone(api_key=api_key)
        
        # Find the index (handle timestamped versions)
        available_indexes = [idx['name'] for idx in pc.list_indexes()]
        
        # Look for exact match or timestamped version
        matching_indexes = [
            idx for idx in available_indexes 
            if idx == self.index_name or idx.startswith(self.index_name + "-")
        ]
        
        if not matching_indexes:
            raise ValueError(
                f"❌ No Pinecone index found matching '{self.index_name}'\n"
                f"Available indexes: {available_indexes}\n"
                f"Please run Milestone 1's upsert_to_pinecone.py first!"
            )
        
        # Use the latest matching index (in case of multiple timestamped versions)
        matching_indexes.sort(reverse=True)
        self.index_name = matching_indexes[0]
        
        # Connect to index
        self.pinecone_index = pc.Index(self.index_name)
        
        # Check stats
        stats = self.pinecone_index.describe_index_stats()
        print(f"   ✅ Connected to index: {self.index_name}")
        print(f"   📊 Total vectors: {stats['total_vector_count']:,}")
    
    def _init_vectorstore(self):
        """Initialize LangChain vector store"""
        print("\n2️⃣ Creating LangChain vector store...")
        
        # Load embeddings (must match Milestone 1's embedding model)
        embed_model = os.getenv('EMBED_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
        
        embeddings = HuggingFaceEmbeddings(
            model_name=embed_model
        )
        
        # Create PineconeVectorStore
        self.vectorstore = PineconeVectorStore(
            index_name=self.index_name,
            embedding=embeddings
        )
        
        print(f"   ✅ Vector store created")
    
    def _init_retriever(self):
        """Initialize retriever using as_retriever() method"""
        print("\n3️⃣ Creating retriever with as_retriever()...")
        
        # Create retriever with configuration
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 5  # Top 5 most relevant chunks
            }
        )
        
        print(f"   ✅ Retriever created (search_type='similarity', k=5)")
    
    def _init_llm(self):
        """Initialize OpenAI LLM"""
        print("\n4️⃣ Initializing OpenAI LLM...")
        
        self.llm = LegalLLM()
    
    def query(self, 
             user_query: str,
             top_k: int = 5,
             include_sources: bool = True) -> Dict:
        """
        Main query method - complete RAG pipeline
        
        Args:
            user_query: User's question
            top_k: Number of chunks to retrieve
            include_sources: Include source provenance in response
        
        Returns:
            Complete formatted response
        """
        print(f"\n{'='*80}")
        print(f"🔍 PROCESSING QUERY")
        print(f"{'='*80}")
        print(f"Query: '{user_query}'")
        
        # Step 1: Retrieve relevant documents
        print(f"\n📚 Step 1: Retrieving relevant documents...")
        
        # Update retriever k if needed
        if top_k != 5:
            self.retriever.search_kwargs["k"] = top_k
        
        retrieved_docs = self.retriever.invoke(user_query)
        
        print(f"   ✅ Retrieved {len(retrieved_docs)} documents")
        
        # Step 2: Format context and sources
        print(f"\n📝 Step 2: Formatting context...")
        
        context_string, sources = self.formatter.format_retrieved_context(retrieved_docs)
        
        print(f"   ✅ Context prepared ({len(context_string)} chars)")
        
        # Step 3: Get conversation history if enabled
        conversation_history = None
        if self.use_memory and self.memory:
            conversation_history = self.memory.get_conversation_context(max_chars=500)
            if conversation_history:
                print(f"\n💬 Step 3: Including conversation history...")
                print(f"   ✅ History: {len(conversation_history)} chars")
        
        # Step 4: Build prompt
        print(f"\n🔧 Step 4: Building prompt...")
        
        system_prompt = get_system_prompt(include_conversation_context=bool(conversation_history))
        user_prompt = get_prompt_with_context(
            query=user_query,
            retrieved_context=context_string,
            conversation_history=conversation_history
        )
        
        print(f"   ✅ Prompt built")
        
        # Step 5: Generate response from OpenAI LLM
        print(f"\n🤖 Step 5: Calling OpenAI LLM...")
        
        llm_result = self.llm.generate_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=500
        )
        
        if not llm_result['success']:
            return {
                "success": False,
                "error": llm_result['error'],
                "query": user_query
            }
        
        print(f"   ✅ Response generated ({llm_result['tokens_used']} tokens)")
        
        # Step 6: Format final response
        print(f"\n📄 Step 6: Formatting response...")
        
        formatted_response = self.formatter.format_final_response(
            llm_response=llm_result['answer'],
            sources=sources if include_sources else [],
            query=user_query,
            metadata={
                "tokens_used": llm_result['tokens_used'],
                "model": llm_result['model'],
                "retrieval_count": len(retrieved_docs)
            }
        )
        
        formatted_response['success'] = True
        
        print(f"   ✅ Response formatted")
        
        # Step 7: Update conversation memory
        if self.use_memory and self.memory:
            print(f"\n💾 Step 7: Updating conversation memory...")
            
            self.memory.add_user_message(
                user_query,
                query_metadata={"retrieved_chunks": len(retrieved_docs)}
            )
            
            self.memory.add_assistant_message(
                llm_result['answer'],
                response_metadata={
                    "tokens_used": llm_result['tokens_used'],
                    "sources_count": len(sources)
                }
            )
            
            print(f"   ✅ Memory updated")
        
        print(f"\n{'='*80}")
        print(f"✅ QUERY PROCESSING COMPLETE")
        print(f"{'='*80}\n")
        
        return formatted_response
    
    def batch_query(self, queries: List[str]) -> List[Dict]:
        """
        Process multiple queries in batch
        
        Args:
            queries: List of user queries
        
        Returns:
            List of formatted responses
        """
        results = []
        
        for i, query in enumerate(queries, 1):
            print(f"\n{'='*80}")
            print(f"BATCH QUERY {i}/{len(queries)}")
            print(f"{'='*80}")
            
            result = self.query(query)
            results.append(result)
        
        return results
    
    def save_session(self):
        """Save conversation session"""
        if self.use_memory and self.memory:
            return self.memory.save_conversation()
        return None
    
    def get_usage_stats(self) -> Dict:
        """Get pipeline usage statistics"""
        stats = {
            "llm_stats": self.llm.get_usage_stats()
        }
        
        if self.use_memory and self.memory:
            stats["conversation_stats"] = self.memory.get_summary()
        
        return stats


# Main execution
def main():
    """
    Test the complete RAG pipeline
    """
    print("\n" + "="*80)
    print("🏛️ LEGAL RAG PIPELINE - COMPLETE TEST")
    print("="*80)
    
    # Initialize pipeline
    pipeline = LegalRAGPipeline(use_conversation_memory=True)
    
    # Test queries
    test_queries = [
        "What is murder under IPC Section 300?",
        "What is the punishment for it?",  # Follow-up question
        "What sections of IPC apply to theft?"
    ]
    
    print(f"\n🔍 Testing with {len(test_queries)} queries...\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'#'*80}")
        print(f"TEST QUERY {i}/{len(test_queries)}")
        print(f"{'#'*80}")
        
        # Query pipeline
        result = pipeline.query(query)
        
        if result['success']:
            # Display formatted response
            print(f"\n{'='*80}")
            print("📊 FORMATTED RESPONSE")
            print(f"{'='*80}")
            
            ui_display = pipeline.formatter.format_for_ui(result)
            print(ui_display)
            
            print(f"\n{'='*80}")
        else:
            print(f"\n❌ Error: {result['error']}")
        
        # Pause between queries
        if i < len(test_queries):
            input("\nPress Enter for next query...")
    
    # Show usage stats
    print(f"\n{'='*80}")
    print("📊 PIPELINE USAGE STATISTICS")
    print(f"{'='*80}")
    
    stats = pipeline.get_usage_stats()
    print(f"\nLLM Stats:")
    print(f"   Total calls: {stats['llm_stats']['total_calls']}")
    print(f"   Total tokens: {stats['llm_stats']['total_tokens']}")
    print(f"   Estimated cost: ${stats['llm_stats']['estimated_cost_usd']:.4f}")
    
    if 'conversation_stats' in stats:
        print(f"\nConversation Stats:")
        print(f"   Session ID: {stats['conversation_stats']['session_id']}")
        print(f"   Total messages: {stats['conversation_stats']['total_messages']}")
        print(f"   Total turns: {stats['conversation_stats']['total_turns']}")
    
    # Save session
    print(f"\n💾 Saving conversation session...")
    saved_path = pipeline.save_session()
    if saved_path:
        print(f"   ✅ Saved to: {saved_path}")
    
    print(f"\n{'='*80}")
    print("✅ PIPELINE TEST COMPLETE!")
    print(f"{'='*80}\n")
# =========================
# 🔗 FRONTEND INTEGRATION
# =========================

_pipeline_instance = None

def get_rag_response(query: str) -> str:
    global _pipeline_instance

    if _pipeline_instance is None:
        _pipeline_instance = LegalRAGPipeline(
            use_conversation_memory=True
        )

    result = _pipeline_instance.query(
        user_query=query,
        include_sources=False
    )

    if not result.get("success"):
        return f"ERROR: {result.get('error')}"

    return result["final_answer"]



if __name__ == "__main__":
    main()