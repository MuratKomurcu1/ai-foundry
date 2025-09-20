import os
import json
import pickle
from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from datetime import datetime

# Core libraries
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
import chromadb
from chromadb.config import Settings

# Utilities
import requests
from pathlib import Path
import logging

@dataclass
class Document:
    """Doküman sınıfı"""
    content: str
    metadata: Dict
    embedding: Optional[np.ndarray] = None

@dataclass
class RetrievalResult:
    """Arama sonucu sınıfı"""
    document: Document
    score: float
    rank: int

class DocumentEmbedder:
    """
    Dokümanları vector embedding'e çeviren sınıf
    """
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Args:
            model_name: Embedding modeli adı
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        logging.info(f"Embedding model yüklendi: {model_name}")
        logging.info(f"Embedding boyutu: {self.dimension}")
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Metinleri embedding'e çevir"""
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings
    
    def encode_single(self, text: str) -> np.ndarray:
        """Tek metni embedding'e çevir"""
        return self.model.encode([text])[0]

class VectorStore:
    """
    Vector veritabanı yönetimi (FAISS kullanımı)
    """
    
    def __init__(self, dimension: int):
        """
        Args:
            dimension: Embedding boyutu
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product index
        self.documents: List[Document] = []
        
    def add_documents(self, documents: List[Document]):
        """Dokümanları ekle"""
        embeddings = np.array([doc.embedding for doc in documents])
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        self.index.add(embeddings)
        self.documents.extend(documents)
        
        logging.info(f"{len(documents)} doküman eklendi. Toplam: {len(self.documents)}")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[RetrievalResult]:
        """
        Benzer dokümanları ara
        
        Args:
            query_embedding: Sorgu embedding'i
            top_k: Dönülecek doküman sayısı
            
        Returns:
            Benzer dokümanların listesi
        """
        # Normalize query
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1:  # Valid result
                results.append(RetrievalResult(
                    document=self.documents[idx],
                    score=float(score),
                    rank=rank + 1
                ))
        
        return results
    
    def save(self, filepath: str):
        """Vector store'u kaydet"""
        faiss.write_index(self.index, f"{filepath}.faiss")
        
        with open(f"{filepath}.docs", 'wb') as f:
            pickle.dump(self.documents, f)
        
        logging.info(f"Vector store kaydedildi: {filepath}")
    
    def load(self, filepath: str):
        """Vector store'u yükle"""
        self.index = faiss.read_index(f"{filepath}.faiss")
        
        with open(f"{filepath}.docs", 'rb') as f:
            self.documents = pickle.load(f)
        
        logging.info(f"Vector store yüklendi: {filepath}")

class DocumentProcessor:
    """
    Doküman işleme ve chunking
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Args:
            chunk_size: Chunk boyutu
            chunk_overlap: Chunk'lar arası örtüşme
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def process_directory(self, directory_path: str) -> List[Document]:
        """
        Klasördeki tüm dokümanları işle
        
        Args:
            directory_path: Doküman klasörü yolu
            
        Returns:
            İşlenmiş dokümanların listesi
        """
        documents = []
        directory = Path(directory_path)
        
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                try:
                    docs = self.process_file(str(file_path))
                    documents.extend(docs)
                except Exception as e:
                    logging.error(f"Dosya işlenemedi {file_path}: {e}")
        
        return documents
    
    def process_file(self, file_path: str) -> List[Document]:
        """
        Tek dosyayı işle
        
        Args:
            file_path: Dosya yolu
            
        Returns:
            İşlenmiş dokümanların listesi
        """
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_extension in ['.txt', '.md']:
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            raise ValueError(f"Desteklenmeyen dosya türü: {file_extension}")
        
        # Load and split
        raw_documents = loader.load()
        split_docs = self.text_splitter.split_documents(raw_documents)
        
        documents = []
        for i, doc in enumerate(split_docs):
            documents.append(Document(
                content=doc.page_content,
                metadata={
                    'source': file_path,
                    'chunk_id': i,
                    'file_type': file_extension,
                    'created_at': datetime.now().isoformat()
                }
            ))
        
        return documents

class LanguageModel:
    """
    Language Model wrapper (Hugging Face transformers)
    """
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        """
        Args:
            model_name: Model adı
        """
        self.model_name = model_name
        
        # Model türüne göre pipeline oluştur
        if "gpt" in model_name.lower() or "llama" in model_name.lower():
            self.pipeline = pipeline(
                "text-generation",
                model=model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
        else:
            # T5 veya encoder-decoder modeller için
            self.pipeline = pipeline(
                "text2text-generation",
                model=model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
        
        logging.info(f"Language model yüklendi: {model_name}")
    
    def generate(self, prompt: str, max_length: int = 500, temperature: float = 0.7) -> str:
        """
        Metin üret
        
        Args:
            prompt: Giriş metni
            max_length: Maksimum uzunluk
            temperature: Yaratıcılık seviyesi
            
        Returns:
            Üretilen metin
        """
        try:
            result = self.pipeline(
                prompt,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.pipeline.tokenizer.eos_token_id
            )
            
            if isinstance(result, list):
                generated_text = result[0]['generated_text']
            else:
                generated_text = result['generated_text']
            
            # Sadece yeni üretilen kısmı döndür
            if generated_text.startswith(prompt):
                return generated_text[len(prompt):].strip()
            
            return generated_text.strip()
            
        except Exception as e:
            logging.error(f"Metin üretme hatası: {e}")
            return "Üzgünüm, cevap üretirken bir hata oluştu."

class RAGSystem:
    """
    Ana RAG sistemi
    """
    
    def __init__(self, 
                 embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 llm_model: str = "microsoft/DialoGPT-medium",
                 vector_store_path: str = "vector_store"):
        """
        Args:
            embedding_model: Embedding model adı
            llm_model: Language model adı
            vector_store_path: Vector store kayıt yolu
        """
        
        self.vector_store_path = vector_store_path
        
        # Initialize components
        self.embedder = DocumentEmbedder(embedding_model)
        self.vector_store = VectorStore(self.embedder.dimension)
        self.document_processor = DocumentProcessor()
        self.language_model = LanguageModel(llm_model)
        
        # Load existing vector store if available
        if os.path.exists(f"{vector_store_path}.faiss"):
            self.vector_store.load(vector_store_path)
            logging.info("Mevcut vector store yüklendi")
    
    def index_documents(self, document_path: str, save_index: bool = True):
        """
        Dokümanları indexle
        
        Args:
            document_path: Doküman yolu (dosya veya klasör)
            save_index: Index'i kaydet
        """
        logging.info(f"Dokümanlar indexleniyor: {document_path}")
        
        # Process documents
        if os.path.isfile(document_path):
            documents = self.document_processor.process_file(document_path)
        else:
            documents = self.document_processor.process_directory(document_path)
        
        if not documents:
            logging.warning("İşlenecek doküman bulunamadı")
            return
        
        # Generate embeddings
        texts = [doc.content for doc in documents]
        embeddings = self.embedder.encode(texts)
        
        # Add embeddings to documents
        for doc, embedding in zip(documents, embeddings):
            doc.embedding = embedding
        
        # Add to vector store
        self.vector_store.add_documents(documents)
        
        # Save index
        if save_index:
            self.vector_store.save(self.vector_store_path)
        
        logging.info(f"İndexleme tamamlandı: {len(documents)} doküman")
    
    def search_documents(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        Doküman ara
        
        Args:
            query: Sorgu
            top_k: Döndürülecek doküman sayısı
            
        Returns:
            Arama sonuçları
        """
        query_embedding = self.embedder.encode_single(query)
        results = self.vector_store.search(query_embedding, top_k)
        
        logging.info(f"Arama tamamlandı: {len(results)} sonuç bulundu")
        return results
    
    def generate_answer(self, question: str, context: str, max_length: int = 300) -> str:
        """
        Kontekst kullanarak cevap üret
        
        Args:
            question: Soru
            context: Kontekst dokümanları
            max_length: Maksimum cevap uzunluğu
            
        Returns:
            Üretilen cevap
        """
        # Create prompt
        prompt = self._create_prompt(question, context)
        
        # Generate answer
        answer = self.language_model.generate(prompt, max_length=max_length)
        
        return self._clean_answer(answer, question)
    
    def _create_prompt(self, question: str, context: str) -> str:
        """
        Prompt oluştur
        
        Args:
            question: Soru
            context: Kontekst
            
        Returns:
            Hazırlanmış prompt
        """
        prompt = f"""Aşağıdaki bilgileri kullanarak soruyu yanıtlayın. Eğer cevap verilen bilgilerde yoksa, bilmediğinizi belirtin.

Bağlam:
{context}

Soru: {question}

Yanıt:"""
        
        return prompt
    
    def _clean_answer(self, answer: str, question: str) -> str:
        """
        Cevabı temizle ve düzenle
        
        Args:
            answer: Ham cevap
            question: Orijinal soru
            
        Returns:
            Temizlenmiş cevap
        """
        # Remove common artifacts
        answer = answer.strip()
        
        # Remove repeated question
        if answer.lower().startswith(question.lower()):
            answer = answer[len(question):].strip()
        
        # Remove prompt artifacts
        answer = answer.replace("Yanıt:", "").strip()
        
        return answer
    
    def ask(self, question: str, top_k: int = 3) -> Dict:
        """
        Ana soru-cevap fonksiyonu
        
        Args:
            question: Soru
            top_k: Kullanılacak doküman sayısı
            
        Returns:
            Cevap ve metadata
        """
        start_time = datetime.now()
        
        # Search relevant documents
        search_results = self.search_documents(question, top_k)
        
        if not search_results:
            return {
                'answer': 'Üzgünüm, bu soruya cevap verebilecek doküman bulunamadı.',
                'sources': [],
                'confidence': 0.0,
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
        
        # Prepare context
        context = self._prepare_context(search_results)
        
        # Generate answer
        answer = self.generate_answer(question, context)
        
        # Calculate confidence (average of top documents' scores)
        confidence = np.mean([result.score for result in search_results])
        
        # Prepare sources
        sources = []
        for result in search_results:
            sources.append({
                'content': result.document.content[:200] + "...",
                'source': result.document.metadata.get('source', 'Unknown'),
                'score': result.score,
                'rank': result.rank
            })
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            'answer': answer,
            'sources': sources,
            'confidence': float(confidence),
            'processing_time': processing_time,
            'question': question
        }
    
    def _prepare_context(self, search_results: List[RetrievalResult]) -> str:
        """
        Arama sonuçlarından kontekst hazırla
        
        Args:
            search_results: Arama sonuçları
            
        Returns:
            Birleştirilmiş kontekst
        """
        context_parts = []
        
        for i, result in enumerate(search_results, 1):
            context_parts.append(f"Doküman {i}:\n{result.document.content}")
        
        return "\n\n".join(context_parts)
    
    def get_statistics(self) -> Dict:
        """
        Sistem istatistikleri
        
        Returns:
            İstatistikler
        """
        return {
            'total_documents': len(self.vector_store.documents),
            'embedding_dimension': self.embedder.dimension,
            'embedding_model': self.embedder.model_name,
            'language_model': self.language_model.model_name,
            'vector_store_size': self.vector_store.index.ntotal
        }

class RAGEvaluator:
    """
    RAG sistemi değerlendirme araçları
    """
    
    def __init__(self, rag_system: RAGSystem):
        self.rag_system = rag_system
    
    def evaluate_retrieval(self, test_queries: List[str], 
                          ground_truth: List[List[str]], 
                          top_k: int = 5) -> Dict:
        """
        Retrieval performansını değerlendir
        
        Args:
            test_queries: Test soruları
            ground_truth: Her soru için doğru doküman ID'leri
            top_k: Değerlendirme için döküman sayısı
            
        Returns:
            Performans metrikleri
        """
        precision_scores = []
        recall_scores = []
        
        for query, true_docs in zip(test_queries, ground_truth):
            # Get retrieval results
            results = self.rag_system.search_documents(query, top_k)
            retrieved_docs = [r.document.metadata.get('chunk_id', '') for r in results]
            
            # Calculate precision and recall
            relevant_retrieved = set(retrieved_docs) & set(true_docs)
            
            precision = len(relevant_retrieved) / len(retrieved_docs) if retrieved_docs else 0
            recall = len(relevant_retrieved) / len(true_docs) if true_docs else 0
            
            precision_scores.append(precision)
            recall_scores.append(recall)
        
        return {
            'mean_precision': np.mean(precision_scores),
            'mean_recall': np.mean(recall_scores),
            'mean_f1': 2 * np.mean(precision_scores) * np.mean(recall_scores) / 
                      (np.mean(precision_scores) + np.mean(recall_scores))
        }
    
    def evaluate_generation(self, test_qa_pairs: List[Tuple[str, str]]) -> Dict:
        """
        Üretim kalitesini değerlendir
        
        Args:
            test_qa_pairs: (soru, doğru_cevap) çiftleri
            
        Returns:
            Üretim metrikleri
        """
        scores = []
        
        for question, expected_answer in test_qa_pairs:
            result = self.rag_system.ask(question)
            generated_answer = result['answer']
            
            # Simple similarity score (in practice, use BLEU, ROUGE, etc.)
            score = self._calculate_similarity(generated_answer, expected_answer)
            scores.append(score)
        
        return {
            'mean_similarity': np.mean(scores),
            'std_similarity': np.std(scores)
        }
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        İki metin arasındaki benzerlik skoru
        
        Args:
            text1: İlk metin
            text2: İkinci metin
            
        Returns:
            Benzerlik skoru (0-1)
        """
        # Simple word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0


# Demo ve örnek kullanım
def demo_rag_system():
    """
    RAG sistemi demo fonksiyonu
    """
    print("🚀 RAG Sistemi Demo Başlatılıyor...")
    
    # Initialize RAG system
    rag = RAGSystem(
        embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        llm_model="microsoft/DialoGPT-medium"
    )
    
    # Create sample documents
    sample_docs = [
        "Yapay zeka, insan zekasının makineler tarafından sergilenmesidir. Bu alan, makine öğrenmesi, derin öğrenme ve doğal dil işleme gibi teknikleri içerir.",
        "Python, yapay zeka projelerinde en çok kullanılan programlama dillerinden biridir. Scikit-learn, TensorFlow ve PyTorch gibi güçlü kütüphanelere sahiptir.",
        "Transformer mimarisi, 2017 yılında Google tarafından tanıtılmış ve doğal dil işlemede devrim yaratmıştır. BERT, GPT ve T5 gibi modeller bu mimariye dayanır.",
        "RAG (Retrieval-Augmented Generation), büyük dil modellerinin bilgi tabanlarından yararlanarak daha doğru cevaplar üretmesini sağlayan bir tekniktir."
    ]
    
    # Create temporary documents
    temp_dir = "temp_docs"
    os.makedirs(temp_dir, exist_ok=True)
    
    for i, content in enumerate(sample_docs):
        with open(f"{temp_dir}/doc_{i}.txt", "w", encoding="utf-8") as f:
            f.write(content)
    
    print("📚 Örnek dokümanlar oluşturuldu")
    
    # Index documents
    print("🔍 Dokümanlar indexleniyor...")
    rag.index_documents(temp_dir)
    
    # Example queries
    queries = [
        "Yapay zeka nedir?",
        "Python neden AI projelerinde popüler?",
        "Transformer mimarisi ne zaman ortaya çıktı?",
        "RAG teknolojisi nasıl çalışır?"
    ]
    
    print("\n💬 Örnek Sorular ve Cevaplar:")
    print("=" * 60)
    
    for query in queries:
        print(f"\n❓ Soru: {query}")
        result = rag.ask(query)
        
        print(f"✅ Cevap: {result['answer']}")
        print(f"🎯 Güven Skoru: {result['confidence']:.3f}")
        print(f"⏱️  İşlem Süresi: {result['processing_time']:.3f}s")
        
        print("📄 Kaynaklar:")
        for i, source in enumerate(result['sources'], 1):
            print(f"   {i}. {source['content']} (Skor: {source['score']:.3f})")
        
        print("-" * 60)
    
    # Statistics
    stats = rag.get_statistics()
    print(f"\n📊 Sistem İstatistikleri:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    
    print("\n✨ Demo tamamlandı!")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run demo
    demo_rag_system()