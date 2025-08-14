import os
import json
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import PyPDF2
import pdfplumber
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PDFChunk:
    """Represents a chunk of text from a PDF with metadata"""
    content: str
    page_number: int
    source_file: str
    chunk_id: str
    start_char: int
    end_char: int
    section_title: Optional[str] = None
    keywords: List[str] = None

@dataclass
class PDFDocument:
    """Represents a complete PDF document"""
    filename: str
    title: str
    total_pages: int
    chunks: List[PDFChunk]
    metadata: Dict[str, Any]
    file_hash: str

class PDFVectorProcessor:
    """Processes PDFs and creates vector embeddings for semantic search"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the processor with embedding model"""
        self.model = SentenceTransformer(model_name)
        self.chunks: List[PDFChunk] = []
        self.documents: List[PDFDocument] = []
        self.embeddings: Optional[np.ndarray] = None
        self.chunk_size = 1000  # characters per chunk
        self.chunk_overlap = 200  # overlap between chunks
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using multiple methods"""
        text = ""
        
        # Try pdfplumber first (better for complex layouts)
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            if text.strip():
                logger.info(f"Successfully extracted text using pdfplumber from {pdf_path}")
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
            logger.info(f"Successfully extracted text using PyPDF2 from {pdf_path}")
            return text
        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path}: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        import re
        
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
    
    def split_into_chunks(self, text: str, filename: str) -> List[PDFChunk]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                for i in range(end, min(end + 200, len(text))):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk_text = text[start:end].strip()
            if len(chunk_text) > 50:  # Only keep substantial chunks
                chunk_id = hashlib.md5(f"{filename}_{start}_{end}".encode()).hexdigest()
                
                chunk = PDFChunk(
                    content=chunk_text,
                    page_number=start // self.chunk_size + 1,  # Approximate page
                    source_file=filename,
                    chunk_id=chunk_id,
                    start_char=start,
                    end_char=end
                )
                chunks.append(chunk)
            
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        
        return chunks
    
    def extract_metadata(self, text: str, filename: str) -> Dict[str, Any]:
        """Extract metadata from PDF text"""
        metadata = {
            'filename': filename,
            'title': filename.replace('.pdf', '').replace('_', ' ').title(),
            'total_chunks': 0,
            'file_size': 0,
            'extraction_date': None
        }
        
        # Try to extract title from first few lines
        lines = text.split('\n')[:10]
        for line in lines:
            line = line.strip()
            if len(line) > 10 and len(line) < 200:
                if not any(word in line.lower() for word in ['page', 'chapter', 'section']):
                    metadata['title'] = line
                    break
        
        return metadata
    
    def process_pdf(self, pdf_path: str) -> Optional[PDFDocument]:
        """Process a single PDF file"""
        try:
            logger.info(f"Processing PDF: {pdf_path}")
            
            # Extract text
            text = self.extract_text_from_pdf(pdf_path)
            if not text.strip():
                logger.error(f"No text extracted from {pdf_path}")
                return None
            
            # Clean text
            text = self.clean_text(text)
            
            # Split into chunks
            chunks = self.split_into_chunks(text, os.path.basename(pdf_path))
            
            # Extract metadata
            metadata = self.extract_metadata(text, os.path.basename(pdf_path))
            metadata['total_chunks'] = len(chunks)
            metadata['file_size'] = os.path.getsize(pdf_path)
            
            # Create document
            file_hash = hashlib.md5(text.encode()).hexdigest()
            document = PDFDocument(
                filename=os.path.basename(pdf_path),
                title=metadata['title'],
                total_pages=len(chunks),
                chunks=chunks,
                metadata=metadata,
                file_hash=file_hash
            )
            
            logger.info(f"Successfully processed {pdf_path}: {len(chunks)} chunks")
            return document
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            return None
    
    def process_directory(self, pdf_directory: str) -> List[PDFDocument]:
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
    
    def create_embeddings(self, documents: List[PDFDocument]) -> np.ndarray:
        """Create embeddings for all chunks"""
        all_chunks = []
        for doc in documents:
            all_chunks.extend(doc.chunks)
        
        # Extract text content for embedding
        texts = [chunk.content for chunk in all_chunks]
        
        logger.info(f"Creating embeddings for {len(texts)} chunks")
        
        # Create embeddings
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        return embeddings
    
    def build_vector_database(self, pdf_directory: str, output_dir: str = "vector_db"):
        """Build complete vector database from PDFs"""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Process PDFs
        logger.info("Processing PDF documents...")
        documents = self.process_directory(pdf_directory)
        
        if not documents:
            logger.error("No documents processed successfully")
            return False
        
        # Create embeddings
        logger.info("Creating embeddings...")
        embeddings = self.create_embeddings(documents)
        
        # Save everything
        logger.info("Saving vector database...")
        
        # Save documents
        with open(os.path.join(output_dir, "documents.pkl"), "wb") as f:
            pickle.dump(documents, f)
        
        # Save embeddings
        np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)
        
        # Save metadata
        metadata = {
            'num_documents': len(documents),
            'num_chunks': sum(len(doc.chunks) for doc in documents),
            'embedding_dim': embeddings.shape[1],
            'model_name': self.model.get_sentence_embedding_dimension()
        }
        
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Vector database saved to {output_dir}")
        logger.info(f"Documents: {len(documents)}, Chunks: {metadata['num_chunks']}")
        
        return True
    
    def load_vector_database(self, db_dir: str = "vector_db") -> bool:
        """Load existing vector database"""
        try:
            # Load documents
            with open(os.path.join(db_dir, "documents.pkl"), "rb") as f:
                self.documents = pickle.load(f)
            
            # Load embeddings
            self.embeddings = np.load(os.path.join(db_dir, "embeddings.npy"))
            
            # Load metadata
            with open(os.path.join(db_dir, "metadata.json"), "r") as f:
                metadata = json.load(f)
            
            logger.info(f"Loaded vector database: {metadata['num_documents']} documents, {metadata['num_chunks']} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error loading vector database: {e}")
            return False
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar chunks using vector similarity"""
        if self.embeddings is None or not self.documents:
            logger.error("Vector database not loaded")
            return []
        
        # Encode query
        query_embedding = self.model.encode([query])
        
        # Calculate similarities
        similarities = np.dot(self.embeddings, query_embedding.T).flatten()
        
        # Get top k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        all_chunks = []
        for doc in self.documents:
            all_chunks.extend(doc.chunks)
        
        for idx in top_indices:
            chunk = all_chunks[idx]
            similarity = similarities[idx]
            
            result = {
                'content': chunk.content,
                'source_file': chunk.source_file,
                'page_number': chunk.page_number,
                'similarity_score': float(similarity),
                'chunk_id': chunk.chunk_id
            }
            results.append(result)
        
        return results

if __name__ == "__main__":
    # Example usage
    processor = PDFVectorProcessor()
    
    # Build vector database
    success = processor.build_vector_database("pdfs", "vector_db")
    
    if success:
        print("Vector database built successfully!")
        
        # Test search
        results = processor.search("PM Kisan scheme benefits")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['source_file']} (Score: {result['similarity_score']:.3f})")
            print(f"   {result['content'][:200]}...")
