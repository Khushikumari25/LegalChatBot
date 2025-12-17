# Milestone 2/src/llm_integration.py
"""
OpenAI LLM Integration for Legal RAG System
Handles all interactions with OpenAI API
"""

import os
from typing import List, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv

# Load environment
load_dotenv()

# OpenAI import
try:
    from openai import OpenAI, AuthenticationError, RateLimitError, BadRequestError
except ImportError:
    raise ImportError(
        "OpenAI package not found. Install with: pip install openai"
    )

class LegalLLM:
    """
    OpenAI LLM wrapper for legal queries
    Configured for accurate, citation-based responses
    """
    
    def __init__(self):
        """Initialize OpenAI client with configuration from .env"""
        
        # Get API key
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError(
                "❌ OPENAI_API_KEY not found in .env file!\n"
                "Please add: OPENAI_API_KEY=your_key_here"
            )
        
        # Get model configuration
        self.model = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
        self.temperature = float(os.getenv('OPENAI_TEMPERATURE', '0.3'))
        
        # Initialize OpenAI client (v1.0+ API)
        self.client = OpenAI(api_key=self.api_key)
        
        # Usage tracking
        self.total_tokens = 0
        self.total_calls = 0
        
        print(f"✅ OpenAI LLM initialized")
        print(f"   Model: {self.model}")
        print(f"   Temperature: {self.temperature}")
    
    def generate_response(self, 
                         system_prompt: str,
                         user_prompt: str,
                         max_tokens: int = 500,
                         temperature: Optional[float] = None) -> Dict:
        """
        Generate response from OpenAI
        
        Args:
            system_prompt: System instructions for the LLM
            user_prompt: User's query with context
            max_tokens: Maximum tokens in response
            temperature: Override default temperature
        
        Returns:
            Dict with response text, tokens used, and metadata
        """
        
        try:
            # Use instance temperature if not overridden
            temp = temperature if temperature is not None else self.temperature
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=temp,
                top_p=0.9,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            # Extract response
            answer = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            
            # Update tracking
            self.total_tokens += tokens_used
            self.total_calls += 1
            
            return {
                "success": True,
                "answer": answer,
                "tokens_used": tokens_used,
                "model": self.model,
                "timestamp": datetime.now().isoformat(),
                "finish_reason": response.choices[0].finish_reason
            }
        
        except AuthenticationError:
            return {
                "success": False,
                "error": "Authentication failed. Check your OpenAI API key in .env file.",
                "error_type": "authentication"
            }
        
        except RateLimitError:
            return {
                "success": False,
                "error": "Rate limit exceeded. Please try again later or upgrade your OpenAI plan.",
                "error_type": "rate_limit"
            }
        
        except BadRequestError as e:
            return {
                "success": False,
                "error": f"Invalid request: {str(e)}",
                "error_type": "invalid_request"
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "error_type": "unknown"
            }
    
    def generate_with_conversation(self,
                                  system_prompt: str,
                                  conversation_history: List[Dict],
                                  max_tokens: int = 500) -> Dict:
        """
        Generate response with full conversation history
        
        Args:
            system_prompt: System instructions
            conversation_history: List of {"role": "user/assistant", "content": "..."}
            max_tokens: Maximum tokens in response
        
        Returns:
            Dict with response
        """
        
        try:
            # Prepare messages
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(conversation_history)
            
            # Call OpenAI
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=self.temperature
            )
            
            answer = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            
            self.total_tokens += tokens_used
            self.total_calls += 1
            
            return {
                "success": True,
                "answer": answer,
                "tokens_used": tokens_used,
                "model": self.model,
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    def get_usage_stats(self) -> Dict:
        """
        Get usage statistics
        
        Returns:
            Dict with usage stats and cost estimate
        """
        # Cost estimates (approximate, as of 2024)
        cost_per_1k = {
            'gpt-3.5-turbo': 0.002,
            'gpt-4': 0.06,
            'gpt-4-turbo': 0.03
        }
        
        cost_rate = cost_per_1k.get(self.model, 0.002)
        estimated_cost = (self.total_tokens / 1000) * cost_rate
        
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": round(estimated_cost, 4),
            "model": self.model
        }


# Test function
def test_llm():
    """Test OpenAI LLM connection and response"""
    
    print("\n" + "="*80)
    print("TESTING OPENAI LLM INTEGRATION")
    print("="*80)
    
    try:
        llm = LegalLLM()
        
        # Test simple query
        system = "You are a helpful legal assistant."
        query = "What is the punishment for theft under IPC?"
        
        print(f"\n🧪 Test Query: '{query}'")
        print(f"⏳ Calling OpenAI API...")
        
        result = llm.generate_response(
            system_prompt=system,
            user_prompt=query,
            max_tokens=150
        )
        
        if result['success']:
            print(f"\n✅ Response received!")
            print(f"📝 Answer: {result['answer'][:200]}...")
            print(f"🔢 Tokens used: {result['tokens_used']}")
            print(f"⏰ Timestamp: {result['timestamp']}")
        else:
            print(f"\n❌ Error: {result['error']}")
            print(f"Error type: {result['error_type']}")
        
        # Show stats
        stats = llm.get_usage_stats()
        print(f"\n📊 Usage Stats:")
        print(f"   Total calls: {stats['total_calls']}")
        print(f"   Total tokens: {stats['total_tokens']}")
        print(f"   Estimated cost: ${stats['estimated_cost_usd']:.4f}")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")


if __name__ == "__main__":
    test_llm()