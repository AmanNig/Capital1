import os
import json
import hashlib
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import PyPDF2
import pdfplumber
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import logging
from pathlib import Path
import time
import requests
from dotenv import load_dotenv
import sys

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PolicySection:
    """Represents a meaningful section from a policy document"""
    title: str
    content: str
    source_file: str
    page_number: int
    section_type: str  # 'definition', 'eligibility', 'benefits', 'procedure', etc.
    keywords: List[str]

@dataclass
class PolicyDocument:
    """Represents a complete policy document with structured sections"""
    filename: str
    title: str
    sections: List[PolicySection]
    metadata: Dict[str, Any]

class ImprovedPolicyProcessor:
    """Advanced PDF processor that extracts meaningful policy information"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.documents: List[PolicyDocument] = []
        self.embeddings: Optional[np.ndarray] = None
        self.section_embeddings: Optional[np.ndarray] = None
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using multiple methods"""
        text = ""
        
        # Try pdfplumber first
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            if text.strip():
                return text
        except Exception as e:
            logger.warning(f"pdfplumber failed for {pdf_path}: {e}")
        
        # Fallback to PyPDF2
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path}: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}]', ' ', text)
        
        # Remove page numbers and headers/footers
        text = re.sub(r'\b\d+\s*of\s*\d+\b', '', text)
        text = re.sub(r'Page\s*\d+', '', text)
        
        # Clean up multiple spaces
        text = ' '.join(text.split())
        
        return text.strip()
    
    def identify_sections(self, text: str) -> List[PolicySection]:
        """Identify and extract meaningful policy sections"""
        sections = []
        
        # If no sections found with patterns, create chunks instead
        if len(text) < 1000:
            # For small texts, create one section
            section = PolicySection(
                title="Main Content",
                content=text,
                source_file="",
                page_number=1,
                section_type="general",
                keywords=self.extract_keywords(text)
            )
            sections.append(section)
            return sections
        
        # Split text into meaningful chunks
        chunk_size = 1500  # Larger chunks for better context
        overlap = 200
        
        start = 0
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                for i in range(end, min(end + 300, len(text))):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk_text = text[start:end].strip()
            if len(chunk_text) > 100:  # Only keep substantial chunks
                section = PolicySection(
                    title=f"Section {len(sections) + 1}",
                    content=chunk_text,
                    source_file="",
                    page_number=1,
                    section_type=self.classify_section("", chunk_text),
                    keywords=self.extract_keywords(chunk_text)
                )
                sections.append(section)
            
            start = end - overlap
            if start >= len(text):
                break
        
        return sections
    
    def classify_section(self, title: str, content: str) -> str:
        """Classify the type of policy section"""
        title_lower = title.lower()
        content_lower = content.lower()
        
        # Define classification patterns
        patterns = {
            'definition': ['definition', 'what is', 'meaning', 'purpose', 'objective'],
            'eligibility': ['eligibility', 'eligible', 'who can', 'qualification', 'criteria'],
            'benefits': ['benefit', 'advantage', 'support', 'assistance', 'subsidy'],
            'procedure': ['procedure', 'process', 'how to', 'application', 'steps'],
            'documents': ['document', 'required', 'certificate', 'proof', 'form'],
            'contact': ['contact', 'address', 'phone', 'email', 'helpline']
        }
        
        for section_type, keywords in patterns.items():
            if any(keyword in title_lower for keyword in keywords) or \
               any(keyword in content_lower for keyword in keywords):
                return section_type
        
        return 'general'
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        # Simple keyword extraction - can be improved with NLP
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        
        keywords = [word for word in words if word not in stop_words and len(word) > 3]
        
        # Count frequency and return top keywords
        from collections import Counter
        keyword_counts = Counter(keywords)
        return [word for word, count in keyword_counts.most_common(10)]
    
    def process_pdf(self, pdf_path: str) -> Optional[PolicyDocument]:
        """Process a single PDF file"""
        try:
            logger.info(f"Processing PDF: {pdf_path}")
            
            # Extract and clean text
            text = self.extract_text_from_pdf(pdf_path)
            if not text.strip():
                logger.error(f"No text extracted from {pdf_path}")
                return None
            
            text = self.clean_text(text)
            
            # Identify sections
            sections = self.identify_sections(text)
            
            # Set source file for all sections
            for section in sections:
                section.source_file = os.path.basename(pdf_path)
            
            # Create document
            document = PolicyDocument(
                filename=os.path.basename(pdf_path),
                title=os.path.basename(pdf_path).replace('.pdf', '').replace('_', ' ').title(),
                sections=sections,
                metadata={
                    'filename': os.path.basename(pdf_path),
                    'total_sections': len(sections),
                    'file_size': os.path.getsize(pdf_path)
                }
            )
            
            logger.info(f"Successfully processed {pdf_path}: {len(sections)} sections")
            return document
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            return None
    
    def process_directory(self, pdf_directory: str) -> List[PolicyDocument]:
        """Process all PDFs in a directory"""
        documents = []
        pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_directory, pdf_file)
            document = self.process_pdf(pdf_path)
            if document:
                documents.append(document)
        
        return documents
    
    def create_embeddings(self, documents: List[PolicyDocument]) -> Tuple[np.ndarray, np.ndarray]:
        """Create embeddings for documents and sections"""
        all_sections = []
        for doc in documents:
            all_sections.extend(doc.sections)
        
        # Create embeddings for sections
        section_texts = [f"{section.title}: {section.content}" for section in all_sections]
        section_embeddings = self.model.encode(section_texts, show_progress_bar=True)
        
        # Create embeddings for full documents
        doc_texts = [f"{doc.title}: {' '.join([s.content for s in doc.sections])}" for doc in documents]
        doc_embeddings = self.model.encode(doc_texts, show_progress_bar=True)
        
        return doc_embeddings, section_embeddings
    
    def build_database(self, pdf_directory: str, output_dir: str = "improved_vector_db"):
        """Build improved vector database"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Process PDFs
        logger.info("Processing PDF documents...")
        documents = self.process_directory(pdf_directory)
        
        if not documents:
            logger.error("No documents processed successfully")
            return False
        
        # Create embeddings
        logger.info("Creating embeddings...")
        doc_embeddings, section_embeddings = self.create_embeddings(documents)
        
        # Save everything
        logger.info("Saving improved vector database...")
        
        with open(os.path.join(output_dir, "documents.pkl"), "wb") as f:
            pickle.dump(documents, f)
        
        np.save(os.path.join(output_dir, "doc_embeddings.npy"), doc_embeddings)
        np.save(os.path.join(output_dir, "section_embeddings.npy"), section_embeddings)
        
        # Save metadata
        metadata = {
            'num_documents': len(documents),
            'num_sections': sum(len(doc.sections) for doc in documents),
            'embedding_dim': doc_embeddings.shape[1],
            'model_name': self.model.get_sentence_embedding_dimension()
        }
        
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        self.documents = documents
        self.embeddings = doc_embeddings
        self.section_embeddings = section_embeddings
        
        logger.info(f"Improved vector database saved to {output_dir}")
        logger.info(f"Documents: {len(documents)}, Sections: {metadata['num_sections']}")
        
        return True
    
    def load_database(self, db_dir: str = "improved_vector_db") -> bool:
        """Load existing database"""
        try:
            with open(os.path.join(db_dir, "documents.pkl"), "rb") as f:
                self.documents = pickle.load(f)
            
            self.embeddings = np.load(os.path.join(db_dir, "doc_embeddings.npy"))
            self.section_embeddings = np.load(os.path.join(db_dir, "section_embeddings.npy"))
            
            with open(os.path.join(db_dir, "metadata.json"), "r") as f:
                metadata = json.load(f)
            
            logger.info(f"Loaded improved database: {metadata['num_documents']} documents, {metadata['num_sections']} sections")
            return True
            
        except Exception as e:
            logger.error(f"Error loading database: {e}")
            return False
    
    def search_and_synthesize(self, query: str, top_k: int = 5) -> str:
        """Search database and synthesize a comprehensive answer"""
        if self.section_embeddings is None or not self.documents:
            return "❌ Database not loaded."
        
        # Encode query
        query_embedding = self.model.encode([query])
        
        # Search in sections
        similarities = np.dot(self.section_embeddings, query_embedding.T).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Collect all sections
        all_sections = []
        for doc in self.documents:
            all_sections.extend(doc.sections)
        
        # Get relevant sections
        relevant_sections = [all_sections[idx] for idx in top_indices]
        
        # Filter out low similarity results
        filtered_sections = []
        for i, section in enumerate(relevant_sections):
            similarity = similarities[top_indices[i]]
            if similarity > 0.3:  # Only include results with decent similarity
                filtered_sections.append((section, similarity))
        
        if not filtered_sections:
            return "❌ No relevant information found for your query. Please try rephrasing your question."
        
        # Sort by similarity
        filtered_sections.sort(key=lambda x: x[1], reverse=True)
        
        # Synthesize answer based on query type
        if any(word in query.lower() for word in ['what is', 'definition', 'meaning']):
            return self._synthesize_definition(query, filtered_sections)
        elif any(word in query.lower() for word in ['how to', 'procedure', 'process', 'apply']):
            return self._synthesize_procedure(query, filtered_sections)
        elif any(word in query.lower() for word in ['benefit', 'advantage', 'support']):
            return self._synthesize_benefits(query, filtered_sections)
        elif any(word in query.lower() for word in ['eligible', 'who can', 'qualification']):
            return self._synthesize_eligibility(query, filtered_sections)
        else:
            return self._synthesize_general(query, filtered_sections)
    
    def _synthesize_definition(self, query: str, sections: List[tuple]) -> str:
        """Synthesize definition answer"""
        answer = f"📋 **Answer to: '{query}'**\n\n"
        
        # Get the best matching section
        best_section, best_score = sections[0]
        
        answer += "**Definition/Overview:**\n"
        answer += f"{best_section.content[:600]}\n\n"
        answer += f"📄 Source: {best_section.source_file}\n"
        answer += f"🎯 Relevance Score: {best_score:.3f}\n\n"
        
        # Add additional context if available
        if len(sections) > 1:
            answer += "**Additional Information:**\n"
            for i, (section, score) in enumerate(sections[1:3], 2):
                answer += f"{i}. {section.content[:300]}\n"
                answer += f"   Source: {section.source_file} (Score: {score:.3f})\n\n"
        
        return answer
    
    def _synthesize_procedure(self, query: str, sections: List[tuple]) -> str:
        """Synthesize procedure answer"""
        answer = f"📋 **Answer to: '{query}'**\n\n"
        
        answer += "**Application Process/Procedure:**\n"
        
        for i, (section, score) in enumerate(sections[:3], 1):
            answer += f"{i}. {section.content[:300]}...\n"
            answer += f"   📄 Source: {section.source_file} (Score: {score:.3f})\n\n"
        
        return answer
    
    def _synthesize_benefits(self, query: str, sections: List[tuple]) -> str:
        """Synthesize benefits answer"""
        answer = f"📋 **Answer to: '{query}'**\n\n"
        
        answer += "**Benefits & Support Available:**\n"
        
        for i, (section, score) in enumerate(sections[:3], 1):
            answer += f"• {section.content[:250]}...\n"
            answer += f"  📄 Source: {section.source_file} (Score: {score:.3f})\n\n"
        
        return answer
    
    def _synthesize_eligibility(self, query: str, sections: List[tuple]) -> str:
        """Synthesize eligibility answer"""
        answer = f"📋 **Answer to: '{query}'**\n\n"
        
        answer += "**Eligibility Criteria:**\n"
        
        for i, (section, score) in enumerate(sections[:2], 1):
            answer += f"{i}. {section.content[:250]}...\n"
            answer += f"   📄 Source: {section.source_file} (Score: {score:.3f})\n\n"
        
        return answer
    
    def _synthesize_general(self, query: str, sections: List[tuple]) -> str:
        """Synthesize general answer"""
        answer = f"📋 **Answer to: '{query}'**\n\n"
        
        answer += "**Relevant Information:**\n"
        for i, (section, score) in enumerate(sections[:3], 1):
            answer += f"{i}. {section.content[:300]}...\n"
            answer += f"   📄 Source: {section.source_file} (Score: {score:.3f})\n\n"
        
        return answer

class GroqClient:
    """Groq API client for generating responses"""
    
    def __init__(self, api_key: str = None):
        # Use provided api_key, or get from environment variable
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def generate_response(self, query: str, context: str, model: str = "llama3-8b-8192") -> str:
        """Generate response using Groq API"""
        if not self.api_key:
            return "❌ GROQ_API_KEY not found. Please set your Groq API key."
        
        try:
            prompt = f"""You are a helpful government policy assistant. Based on the following context from policy documents, answer the user's question accurately and comprehensively.

Context from policy documents:
{context}

User Question: {query}

Instructions:
1. Answer based ONLY on the provided context
2. If the context doesn't contain relevant information, say so clearly
3. Provide specific details, numbers, and procedures when available
4. Structure your response clearly with bullet points or numbered lists when appropriate
5. Cite the source documents mentioned in the context
6. Be concise but comprehensive

Answer:"""

            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a government policy expert assistant. Provide accurate, helpful responses based on the given context."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 1000
            }
            
            response = requests.post(self.base_url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Groq API error: {e}")
            return f"❌ Error connecting to Groq API: {e}"
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"❌ Error generating response: {e}"

class ImprovedPolicyChatbot:
    """Improved chatbot with better understanding and synthesis using Groq"""
    
    def __init__(self, db_dir: str = "improved_vector_db", groq_api_key: str = None):
        self.processor = ImprovedPolicyProcessor()
        self.db_dir = db_dir
        self.is_loaded = False
        self.groq_client = GroqClient(groq_api_key)
        
        if os.path.exists(db_dir):
            self.is_loaded = self.processor.load_database(db_dir)
    
    def build_database(self, pdf_directory: str) -> bool:
        """Build improved database"""
        return self.processor.build_database(pdf_directory, self.db_dir)
    
    def ask_question(self, query: str) -> str:
        """Ask a question and get synthesized answer"""
        if not self.is_loaded:
            return "❌ Database not loaded. Please build the database first."
        
        return self.processor.search_and_synthesize(query)
    
    def ask_question_with_groq(self, query: str, top_k: int = 5) -> str:
        """Ask a question and get AI-generated answer using Groq"""
        if not self.is_loaded:
            return "❌ Database not loaded. Please build the database first."
        
        # Get relevant context from vector database
        context = self._get_context_for_query(query, top_k)
        
        if not context:
            return "❌ No relevant information found for your query. Please try rephrasing your question."
        
        # Generate response using Groq
        return self.groq_client.generate_response(query, context)
    
    def _get_context_for_query(self, query: str, top_k: int = 5) -> str:
        """Get relevant context from vector database for Groq"""
        if self.processor.section_embeddings is None or not self.processor.documents:
            return ""
        
        # Encode query
        query_embedding = self.processor.model.encode([query])
        
        # Search in sections
        similarities = np.dot(self.processor.section_embeddings, query_embedding.T).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Collect all sections
        all_sections = []
        for doc in self.processor.documents:
            all_sections.extend(doc.sections)
        
        # Get relevant sections with similarity scores
        relevant_sections = []
        for i, idx in enumerate(top_indices):
            similarity = similarities[idx]
            if similarity > 0.3:  # Only include results with decent similarity
                section = all_sections[idx]
                relevant_sections.append((section, similarity))
        
        if not relevant_sections:
            return ""
        
        # Sort by similarity
        relevant_sections.sort(key=lambda x: x[1], reverse=True)
        
        # Build context string
        context_parts = []
        for i, (section, score) in enumerate(relevant_sections[:3], 1):
            context_parts.append(f"Source {i} (Relevance: {score:.3f}): {section.source_file}")
            context_parts.append(f"Content: {section.content}")
            context_parts.append("---")
        
        return "\n".join(context_parts)
    
    def get_statistics(self) -> str:
        """Get database statistics"""
        if not self.is_loaded:
            return "❌ Database not loaded."
        
        total_sections = sum(len(doc.sections) for doc in self.processor.documents)
        total_docs = len(self.processor.documents)
        
        stats = f"📊 **Improved Policy Database Statistics**\n\n"
        stats += f"📚 Total Documents: {total_docs}\n"
        stats += f"📄 Total Sections: {total_sections}\n"
        stats += f"🔍 Embedding Dimension: {self.processor.embeddings.shape[1] if self.processor.embeddings is not None else 'N/A'}\n"
        
        # List documents with section counts
        stats += f"\n📋 **Documents in Database:**\n"
        for i, doc in enumerate(self.processor.documents, 1):
            stats += f"{i}. {doc.title} ({len(doc.sections)} sections)\n"
        
        return stats
    
    def run_interactive(self):
        """Run interactive chatbot"""
        print("=" * 70)
        print("🏛️  IMPROVED GOVERNMENT POLICY CHATBOT WITH GROQ 🏛️")
        print("=" * 70)
        
        if not self.is_loaded:
            print("❌ No database loaded!")
            print("💡 To build database: python improved_policy_chatbot.py --build pdfs")
            return
        
        print("✅ Improved database loaded successfully!")
        
        # Check if Groq is available
        if self.groq_client.api_key:
            print("✅ Groq AI integration available!")
            print("🤖 Use 'groq' command to switch to AI-powered responses")
        else:
            print("⚠️  Groq API key not found. Set GROQ_API_KEY environment variable for AI responses")
        
        print("\n💬 Ask questions about your policy documents!")
        print("📝 Commands: 'stats', 'quit', 'help', 'groq', 'vector'")
        print("-" * 70)
        
        # Track response mode
        use_groq = False
        
        while True:
            try:
                user_input = input(f"\n🏛️ You ({'AI' if use_groq else 'Vector'}): ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                elif user_input.lower() == 'stats':
                    print(self.get_statistics())
                
                elif user_input.lower() == 'groq':
                    if self.groq_client.api_key:
                        use_groq = True
                        print("🤖 Switched to AI-powered responses using Groq")
                    else:
                        print("❌ Groq API key not available. Set GROQ_API_KEY environment variable")
                
                elif user_input.lower() == 'vector':
                    use_groq = False
                    print("🔍 Switched to vector-based responses")
                
                elif user_input.lower() == 'help':
                    print("\n📖 **Help**")
                    print("• Ask specific questions about policies")
                    print("• Examples:")
                    print("  - 'What is PM Kisan scheme?'")
                    print("  - 'How to apply for crop insurance?'")
                    print("  - 'What are the benefits of soil health card?'")
                    print("  - 'Who is eligible for PMKSY?'")
                    print("• Commands:")
                    print("  - 'groq': Switch to AI-powered responses")
                    print("  - 'vector': Switch to vector-based responses")
                    print("  - 'stats': See database statistics")
                    print("  - 'quit': Exit")
                
                elif user_input:
                    print("🔄 Processing your question...")
                    start_time = time.time()
                    
                    if use_groq and self.groq_client.api_key:
                        answer = self.ask_question_with_groq(user_input)
                    else:
                        answer = self.ask_question(user_input)
                    
                    end_time = time.time()
                    print(f"\n🤖 Bot: {answer}")
                    print(f"⚡ Response time: {end_time - start_time:.2f} seconds")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

sys.modules["__main__"].PolicyDocument = PolicyDocument
sys.modules["__main__"].PolicySection = PolicySection


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Improved Government Policy Chatbot with Groq")
    parser.add_argument("--build", type=str, help="Build improved database from PDF directory")
    parser.add_argument("--db-dir", type=str, default="improved_vector_db", help="Database directory")
    parser.add_argument("--interactive", action="store_true", help="Run interactive mode")
    parser.add_argument("--groq-key", type=str, help="Groq API key (or set GROQ_API_KEY env var)")
    
    args = parser.parse_args()
    
    chatbot = ImprovedPolicyChatbot(args.db_dir, args.groq_key)
    
    if args.build:
        success = chatbot.build_database(args.build)
        if success:
            print("✅ Improved database built successfully!")
            if args.interactive:
                chatbot.run_interactive()
        else:
            print("❌ Failed to build database")
            return 1
    
    elif args.interactive:
        chatbot.run_interactive()
    
    else:
        if chatbot.is_loaded:
            chatbot.run_interactive()
        else:
            print("❌ No database found!")
            print("💡 To build database: python improved_policy_chatbot.py --build pdfs")
            print("💡 To run interactive: python improved_policy_chatbot.py --interactive")

if __name__ == "__main__":
    main()
