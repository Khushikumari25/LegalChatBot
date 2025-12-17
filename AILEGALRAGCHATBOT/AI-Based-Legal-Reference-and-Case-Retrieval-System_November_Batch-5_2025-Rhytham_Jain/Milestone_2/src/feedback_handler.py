# Milestone 2/src/feedback_handler.py
"""
Feedback & Rating Handler
Captures user trust ratings and feedback
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
import uuid

class FeedbackHandler:
    """
    Handles user feedback and ratings for responses
    """
    
    def __init__(self, output_dir: str = None):
        """
        Initialize feedback handler
        
        Args:
            output_dir: Directory to save feedback logs
        """
        # Fix: Use relative path from src directory to Milestone 2/outputs
        if output_dir is None:
            # Get the directory where this script is located (src folder)
            script_dir = Path(__file__).parent
            # Go up one level to Milestone 2, then into outputs/feedback
            output_dir = script_dir.parent / "outputs" / "feedback"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.feedback_file = self.output_dir / "feedback_log.json"
        self.ratings_file = self.output_dir / "ratings_summary.json"
        
        # Load existing feedback
        self.feedback_log = self._load_feedback_log()
    
    def _load_feedback_log(self) -> List[Dict]:
        """Load existing feedback log"""
        if self.feedback_file.exists():
            with open(self.feedback_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def _save_feedback_log(self):
        """Save feedback log"""
        with open(self.feedback_file, 'w', encoding='utf-8') as f:
            json.dump(self.feedback_log, f, indent=2, ensure_ascii=False)
    
    def add_rating(self,
                  query: str,
                  answer: str,
                  rating: int,
                  sources_count: int = 0,
                  comment: Optional[str] = None,
                  session_id: Optional[str] = None) -> str:
        """
        Add user rating for a response
        
        Args:
            query: User's query
            answer: System's answer
            rating: Rating (1-5 stars, or thumbs up/down: 1=down, 5=up)
            sources_count: Number of sources used
            comment: Optional user comment
            session_id: Optional session identifier
        
        Returns:
            Feedback ID
        """
        feedback_id = str(uuid.uuid4())[:8]
        
        feedback_entry = {
            "feedback_id": feedback_id,
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id or "unknown",
            "query": query,
            "answer": answer[:500],  # Truncate for storage
            "rating": rating,
            "sources_count": sources_count,
            "comment": comment,
        }
        
        self.feedback_log.append(feedback_entry)
        self._save_feedback_log()
        
        print(f"✅ Feedback recorded: {feedback_id} (Rating: {rating}/5)")
        
        return feedback_id
    
    def add_thumbs_feedback(self,
                           query: str,
                           answer: str,
                           is_helpful: bool,
                           sources_count: int = 0,
                           session_id: Optional[str] = None) -> str:
        """
        Add thumbs up/down feedback
        
        Args:
            query: User's query
            answer: System's answer
            is_helpful: True for thumbs up, False for thumbs down
            sources_count: Number of sources used
            session_id: Optional session identifier
        
        Returns:
            Feedback ID
        """
        rating = 5 if is_helpful else 1
        return self.add_rating(
            query=query,
            answer=answer,
            rating=rating,
            sources_count=sources_count,
            comment="Thumbs up" if is_helpful else "Thumbs down",
            session_id=session_id
        )
    
    def get_ratings_summary(self) -> Dict:
        """
        Generate summary statistics of ratings
        
        Returns:
            Summary dict with statistics
        """
        if not self.feedback_log:
            return {
                "total_ratings": 0,
                "average_rating": 0,
                "rating_distribution": {},
                "positive_percentage": 0
            }
        
        ratings = [entry['rating'] for entry in self.feedback_log]
        
        # Calculate distribution
        distribution = {}
        for r in range(1, 6):
            distribution[r] = ratings.count(r)
        
        # Calculate average
        avg_rating = sum(ratings) / len(ratings)
        
        # Calculate positive percentage (4-5 stars)
        positive_count = sum(1 for r in ratings if r >= 4)
        positive_pct = (positive_count / len(ratings)) * 100
        
        summary = {
            "total_ratings": len(ratings),
            "average_rating": round(avg_rating, 2),
            "rating_distribution": distribution,
            "positive_percentage": round(positive_pct, 1),
            "last_updated": datetime.now().isoformat()
        }
        
        # Save summary
        with open(self.ratings_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def get_low_rated_queries(self, threshold: int = 2) -> List[Dict]:
        """
        Get queries with low ratings for improvement
        
        Args:
            threshold: Rating threshold (queries <= this value)
        
        Returns:
            List of low-rated queries
        """
        low_rated = [
            entry for entry in self.feedback_log
            if entry['rating'] <= threshold
        ]
        
        return low_rated
    
    def export_feedback_report(self, filename: Optional[str] = None) -> Path:
        """
        Export detailed feedback report
        
        Args:
            filename: Custom filename (default: feedback_report_YYYYMMDD.json)
        
        Returns:
            Path to exported file
        """
        if not filename:
            filename = f"feedback_report_{datetime.now().strftime('%Y%m%d')}.json"
        
        filepath = self.output_dir / filename
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": self.get_ratings_summary(),
            "total_feedback": len(self.feedback_log),
            "low_rated_queries": self.get_low_rated_queries(),
            "recent_feedback": self.feedback_log[-10:]  # Last 10
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Feedback report exported: {filepath}")
        
        return filepath


# Test function
def test_feedback_handler():
    """Test feedback handling"""
    
    print("\n" + "="*80)
    print("TESTING FEEDBACK HANDLER")
    print("="*80)
    
    handler = FeedbackHandler()
    
    # Simulate some feedback
    print("\n🔍 Adding test feedback...")
    
    # Good feedback
    handler.add_rating(
        query="What is murder?",
        answer="Murder is defined under Section 300...",
        rating=5,
        sources_count=3,
        comment="Very helpful!",
        session_id="test_001"
    )
    
    # Bad feedback
    handler.add_rating(
        query="What is theft?",
        answer="Theft is...",
        rating=2,
        sources_count=1,
        comment="Not enough detail",
        session_id="test_001"
    )
    
    # Thumbs up
    handler.add_thumbs_feedback(
        query="What is Section 302?",
        answer="Section 302 prescribes punishment...",
        is_helpful=True,
        sources_count=2,
        session_id="test_002"
    )
    
    # Thumbs down
    handler.add_thumbs_feedback(
        query="What is bail?",
        answer="Bail provisions...",
        is_helpful=False,
        sources_count=1,
        session_id="test_002"
    )
    
    # Get summary
    print("\n📊 Ratings Summary:")
    summary = handler.get_ratings_summary()
    print(f"   Total ratings: {summary['total_ratings']}")
    print(f"   Average rating: {summary['average_rating']}/5")
    print(f"   Positive percentage: {summary['positive_percentage']}%")
    print(f"   Distribution: {summary['rating_distribution']}")
    
    # Get low-rated
    print("\n⚠️ Low-rated queries:")
    low_rated = handler.get_low_rated_queries(threshold=2)
    for entry in low_rated:
        print(f"   - {entry['query']} (Rating: {entry['rating']}/5)")
        print(f"     Comment: {entry['comment']}")
    
    # Export report
    print("\n📄 Exporting report...")
    report_path = handler.export_feedback_report()
    
    print("\n✅ Test complete!")


if __name__ == "__main__":
    test_feedback_handler()