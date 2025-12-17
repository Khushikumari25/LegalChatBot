# Milestone 2/src/response_formatter.py
"""
Response Formatter
Formats LLM responses with proper citations and provenance
"""

from typing import List, Dict, Optional
from datetime import datetime
import re

class ResponseFormatter:
    """
    Formats LLM responses with citations and source tracking
    """
    
    def __init__(self):
        self.citation_pattern = r'\[Source: (.*?), Page: (.*?), Section: (.*?)\]'
    
    def format_retrieved_context(self, retrieved_docs: List) -> tuple:
        """
        Format retrieved documents into context string and source list
        
        Args:
            retrieved_docs: List of LangChain Document objects from retriever
        
        Returns:
            Tuple of (context_string, sources_list)
        """
        context_parts = []
        sources = []
        
        for i, doc in enumerate(retrieved_docs):
            # Extract content and metadata
            content = doc.page_content
            metadata = doc.metadata
            
            # Build context entry
            source_id = f"Source_{i+1}"
            context_parts.append(f"[{source_id}]\n{content}\n")
            
            # Build source entry
            source_info = {
                "id": source_id,
                "source_file": metadata.get('source_file') or metadata.get('doc_filename', 'Unknown'),
                "page_number": metadata.get('page_number', 'N/A'),
                "section": metadata.get('section', 'N/A'),
                "chunk_id": metadata.get('chunk_id', 'N/A'),
                "text_preview": content[:200] + "..." if len(content) > 200 else content
            }
            sources.append(source_info)
        
        context_string = "\n".join(context_parts)
        
        return context_string, sources
    
    def format_final_response(self, 
                             llm_response: str,
                             sources: List[Dict],
                             query: str,
                             metadata: Optional[Dict] = None) -> Dict:
        """
        Format the complete response with answer, citations, and provenance
        
        Args:
            llm_response: Raw response from LLM
            sources: List of source documents used
            query: Original user query
            metadata: Additional metadata (tokens, timestamp, etc.)
        
        Returns:
            Formatted response dictionary
        """
        # Extract citations from LLM response
        citations = self._extract_citations(llm_response)
        
        # Build formatted response
        formatted_response = {
            "query": query,
            "answer": llm_response,
            "citations": citations,
            "sources": sources,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "sources_count": len(sources),
                "has_citations": len(citations) > 0,
                **(metadata or {})
            }
        }
        
        return formatted_response
    
    def _extract_citations(self, text: str) -> List[Dict]:
        """Extract citations from text"""
        citations = []
        matches = re.finditer(self.citation_pattern, text)
        
        for match in matches:
            citations.append({
                "document": match.group(1),
                "page": match.group(2),
                "section": match.group(3),
                "full_citation": match.group(0)
            })
        
        return citations
    
    def format_sources_display(self, sources: List[Dict]) -> str:
        """
        Format sources for display (provenance)
        
        Args:
            sources: List of source dictionaries
        
        Returns:
            Formatted string for display
        """
        if not sources:
            return "No sources retrieved."
        
        display_parts = ["\n📚 **SOURCES USED:**\n"]
        
        for i, source in enumerate(sources, 1):
            display_parts.append(f"{i}. **{source['source_file']}**")
            display_parts.append(f"   - Page: {source['page_number']}")
            display_parts.append(f"   - Section: {source['section']}")
            display_parts.append(f"   - Preview: {source['text_preview']}")
            display_parts.append("")
        
        return "\n".join(display_parts)
    
    def format_for_ui(self, formatted_response: Dict) -> str:
        """
        Format complete response for UI display
        
        Args:
            formatted_response: Output from format_final_response()
        
        Returns:
            Complete formatted string for display
        """
        parts = []
        
        # Main answer
        parts.append("## 📝 Answer\n")
        parts.append(formatted_response['answer'])
        parts.append("")
        
        # Sources
        parts.append(self.format_sources_display(formatted_response['sources']))
        
        # Metadata
        metadata = formatted_response['metadata']
        parts.append("\n---")
        parts.append(f"*Retrieved {metadata['sources_count']} sources | "
                    f"Generated at {metadata['timestamp'][:19]}*")
        
        return "\n".join(parts)
    
    def create_provenance_summary(self, sources: List[Dict]) -> Dict:
        """
        Create summary of sources for provenance tracking
        
        Args:
            sources: List of source dictionaries
        
        Returns:
            Provenance summary
        """
        # Count by document
        doc_counts = {}
        for source in sources:
            doc = source['source_file']
            doc_counts[doc] = doc_counts.get(doc, 0) + 1
        
        # Extract unique sections
        sections = set()
        for source in sources:
            if source['section'] != 'N/A':
                sections.add(source['section'])
        
        return {
            "total_sources": len(sources),
            "documents_used": list(doc_counts.keys()),
            "document_counts": doc_counts,
            "sections_referenced": sorted(list(sections)),
            "timestamp": datetime.now().isoformat()
        }


# Test function
def test_response_formatter():
    """Test response formatting"""
    
    print("\n" + "="*80)
    print("TESTING RESPONSE FORMATTER")
    print("="*80)
    
    formatter = ResponseFormatter()
    
    # Mock retrieved documents (simulate LangChain Documents)
    class MockDoc:
        def __init__(self, content, metadata):
            self.page_content = content
            self.metadata = metadata
    
    mock_docs = [
        MockDoc(
            "Section 300: Murder. Except in the cases hereinafter excepted, culpable homicide is murder...",
            {"doc_filename": "IPC_1860.pdf", "page_number": 112, "section": "300"}
        ),
        MockDoc(
            "Section 302: Whoever commits murder shall be punished with death, or imprisonment for life...",
            {"doc_filename": "IPC_1860.pdf", "page_number": 113, "section": "302"}
        )
    ]
    
    # Format context
    print("\n1️⃣ Formatting retrieved context...")
    context, sources = formatter.format_retrieved_context(mock_docs)
    print(f"   ✅ Created context with {len(sources)} sources")
    
    # Mock LLM response
    llm_response = """**Answer:**
Murder is defined under Section 300 of the Indian Penal Code [Source: IPC_1860.pdf, Page: 112, Section: 300]. The punishment for murder is specified in Section 302, which prescribes death or life imprisonment [Source: IPC_1860.pdf, Page: 113, Section: 302].

**Sources:**
- IPC Section 300 defines murder
- IPC Section 302 prescribes punishment

**Disclaimer:**
⚠️ Please consult a qualified legal counsel for specific legal advice."""
    
    # Format final response
    print("\n2️⃣ Formatting final response...")
    formatted = formatter.format_final_response(
        llm_response=llm_response,
        sources=sources,
        query="What is murder and its punishment?",
        metadata={"tokens_used": 150}
    )
    
    print(f"   ✅ Response formatted")
    print(f"   📊 Citations found: {len(formatted['citations'])}")
    
    # Show citations
    print("\n3️⃣ Extracted citations:")
    for i, citation in enumerate(formatted['citations'], 1):
        print(f"   {i}. {citation['document']} - Section {citation['section']}")
    
    # Format for UI
    print("\n4️⃣ UI display format:")
    print("-"*80)
    ui_display = formatter.format_for_ui(formatted)
    print(ui_display[:500] + "...")
    print("-"*80)
    
    # Provenance summary
    print("\n5️⃣ Provenance summary:")
    provenance = formatter.create_provenance_summary(sources)
    print(f"   Total sources: {provenance['total_sources']}")
    print(f"   Documents: {provenance['documents_used']}")
    print(f"   Sections: {provenance['sections_referenced']}")
    
    print("\n✅ Test complete!")


if __name__ == "__main__":
    test_response_formatter()