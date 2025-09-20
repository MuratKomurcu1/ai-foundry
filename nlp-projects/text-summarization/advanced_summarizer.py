import torch
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, 
    pipeline, BartTokenizer, BartForConditionalGeneration
)
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import re
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass

@dataclass
class SummaryResult:
    """Özetleme sonucu"""
    original_text: str
    summary: str
    method: str
    compression_ratio: float
    key_sentences: List[str]
    keywords: List[str]
    processing_time: float

class TurkishTextProcessor:
    """Türkçe metin ön işleme"""
    
    def __init__(self):
        self.turkish_stopwords = {
            'bir', 'bu', 'da', 'de', 'den', 'dır', 'dir', 'ile', 'için', 'ki', 
            've', 'var', 'yok', 'olan', 'oldu', 'olan', 'ama', 'fakat', 'ancak',
            'çok', 'az', 'fazla', 'biraz', 'şey', 'şu', 'o', 'ben', 'sen', 'biz',
            'siz', 'onlar', 'ne', 'neden', 'nasıl', 'nerede', 'kim', 'hangi'
        }
    
    def clean_text(self, text: str) -> str:
        """Metni temizle"""
        # URL'leri kaldır
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        # E-mail'leri kaldır  
        text = re.sub(r'\S*@\S*\s?', '', text)
        # Özel karakterleri temizle
        text = re.sub(r'[^\w\s.,!?;:]', ' ', text)
        # Çoklu boşlukları tek yap
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """Anahtar kelimeleri çıkar"""
        words = word_tokenize(text.lower())
        words = [w for w in words if w.isalpha() and w not in self.turkish_stopwords]
        
        # Frekans hesapla
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # En sık kullanılanları döndür
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:top_k]]

class ExtractiveSummarizer:
    """Çıkarımsal özetleme (Extractive Summarization)"""
    
    def __init__(self):
        self.sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.text_processor = TurkishTextProcessor()
    
    def textrank_summarize(self, text: str, num_sentences: int = 3) -> Tuple[str, List[str]]:
        """TextRank algoritması ile özetleme"""
        sentences = sent_tokenize(text)
        
        if len(sentences) <= num_sentences:
            return text, sentences
        
        # Cümle embeddingleri
        sentence_embeddings = self.sentence_model.encode(sentences)
        
        # Benzerlik matrisi
        similarity_matrix = cosine_similarity(sentence_embeddings)
        
        # Graf oluştur
        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph)
        
        # En yüksek skorlu cümleleri seç
        ranked_sentences = sorted(
            [(scores[i], s, i) for i, s in enumerate(sentences)], 
            reverse=True
        )
        
        # Orijinal sırayı koru
        selected_indices = sorted([idx for _, _, idx in ranked_sentences[:num_sentences]])
        key_sentences = [sentences[i] for i in selected_indices]
        summary = ' '.join(key_sentences)
        
        return summary, key_sentences
    
    def cluster_based_summarize(self, text: str, num_clusters: int = 3) -> Tuple[str, List[str]]:
        """Kümeleme tabanlı özetleme"""
        sentences = sent_tokenize(text)
        
        if len(sentences) <= num_clusters:
            return text, sentences
        
        # TF-IDF vektörleri
        vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # K-means kümeleme
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(tfidf_matrix)
        
        # Her kümeden en merkeze yakın cümleyi seç
        cluster_centers = kmeans.cluster_centers_
        key_sentences = []
        
        for i in range(num_clusters):
            cluster_sentences = [sentences[j] for j in range(len(sentences)) if cluster_labels[j] == i]
            cluster_vectors = tfidf_matrix[cluster_labels == i]
            
            # Küme merkezine en yakın cümle
            distances = cosine_similarity(cluster_vectors, cluster_centers[i].reshape(1, -1))
            closest_idx = np.argmax(distances)
            key_sentences.append(cluster_sentences[closest_idx])
        
        summary = ' '.join(key_sentences)
        return summary, key_sentences

class AbstractiveSummarizer:
    """Üretimsel özetleme (Abstractive Summarization)"""
    
    def __init__(self):
        # Çok dilli model
        self.model_name = "facebook/mbart-large-50-many-to-many-mmt"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        
        # Pipeline oluştur
        self.summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            tokenizer="facebook/bart-large-cnn"
        )
        
        # Türkçe model (eğer mevcut ise)
        try:
            self.turkish_summarizer = pipeline(
                "summarization",
                model="google/mt5-small"
            )
        except:
            self.turkish_summarizer = None
    
    def bart_summarize(self, text: str, max_length: int = 150, min_length: int = 50) -> str:
        """BART modeli ile özetleme"""
        # Metni parçalara böl (BART'ın token limitine göre)
        max_chunk_length = 1024
        chunks = self._split_text(text, max_chunk_length)
        
        summaries = []
        for chunk in chunks:
            try:
                result = self.summarizer(
                    chunk,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )
                summaries.append(result[0]['summary_text'])
            except Exception as e:
                logging.warning(f"BART summarization failed for chunk: {e}")
                summaries.append(chunk[:200] + "...")
        
        return ' '.join(summaries)
    
    def t5_summarize(self, text: str, max_length: int = 150) -> str:
        """T5 modeli ile özetleme"""
        # T5 için özel prompt
        input_text = f"summarize: {text}"
        
        try:
            inputs = self.tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
            
            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs,
                    max_length=max_length,
                    min_length=30,
                    length_penalty=2.0,
                    num_beams=4,
                    early_stopping=True
                )
            
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return summary
            
        except Exception as e:
            logging.warning(f"T5 summarization failed: {e}")
            return text[:200] + "..."
    
    def _split_text(self, text: str, max_length: int) -> List[str]:
        """Metni parçalara böl"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            current_length += len(word) + 1  # +1 for space
            if current_length > max_length:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

class HybridSummarizer:
    """Hibrit özetleme sistemi"""
    
    def __init__(self):
        self.extractive = ExtractiveSummarizer()
        self.abstractive = AbstractiveSummarizer()
        self.text_processor = TurkishTextProcessor()
    
    def summarize(self, text: str, method: str = "hybrid", 
                 ratio: float = 0.3, max_sentences: int = 5) -> SummaryResult:
        """
        Ana özetleme fonksiyonu
        
        Args:
            text: Özetlenecek metin
            method: Yöntem ("extractive", "abstractive", "hybrid")
            ratio: Özetleme oranı
            max_sentences: Maksimum cümle sayısı
        """
        import time
        start_time = time.time()
        
        # Metni temizle
        cleaned_text = self.text_processor.clean_text(text)
        
        # Anahtar kelimeleri çıkar
        keywords = self.text_processor.extract_keywords(cleaned_text)
        
        if method == "extractive":
            summary, key_sentences = self.extractive.textrank_summarize(
                cleaned_text, max_sentences
            )
        
        elif method == "abstractive":
            summary = self.abstractive.bart_summarize(cleaned_text)
            key_sentences = sent_tokenize(summary)
        
        elif method == "hybrid":
            # İlk extractive ile anahtar cümleleri bul
            extractive_summary, key_sentences = self.extractive.textrank_summarize(
                cleaned_text, max_sentences
            )
            
            # Sonra abstractive ile yeniden yaz
            summary = self.abstractive.bart_summarize(extractive_summary)
        
        else:
            raise ValueError(f"Desteklenmeyen method: {method}")
        
        processing_time = time.time() - start_time
        compression_ratio = len(summary) / len(text)
        
        return SummaryResult(
            original_text=text,
            summary=summary,
            method=method,
            compression_ratio=compression_ratio,
            key_sentences=key_sentences,
            keywords=keywords,
            processing_time=processing_time
        )

class DocumentSummarizer:
    """Doküman özetleme (uzun metinler için)"""
    
    def __init__(self):
        self.hybrid_summarizer = HybridSummarizer()
    
    def summarize_document(self, text: str, target_length: int = 500) -> Dict:
        """
        Uzun dokümanları özetle
        
        Args:
            text: Doküman metni
            target_length: Hedef özet uzunluğu
        """
        # Metni bölümlere ayır
        sections = self._split_into_sections(text)
        
        # Her bölümü özetle
        section_summaries = []
        for i, section in enumerate(sections):
            result = self.hybrid_summarizer.summarize(
                section, method="extractive", max_sentences=3
            )
            section_summaries.append({
                'section_id': i,
                'original_length': len(section),
                'summary': result.summary,
                'keywords': result.keywords
            })
        
        # Bölüm özetlerini birleştir
        combined_summary = ' '.join([s['summary'] for s in section_summaries])
        
        # Gerekirse son bir kez özetle
        if len(combined_summary) > target_length:
            final_result = self.hybrid_summarizer.summarize(
                combined_summary, method="abstractive"
            )
            final_summary = final_result.summary
        else:
            final_summary = combined_summary
        
        # Tüm anahtar kelimeleri topla
        all_keywords = []
        for s in section_summaries:
            all_keywords.extend(s['keywords'])
        
        # Tekrarları kaldır ve frekansa göre sırala
        keyword_freq = {}
        for kw in all_keywords:
            keyword_freq[kw] = keyword_freq.get(kw, 0) + 1
        
        top_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        top_keywords = [kw for kw, freq in top_keywords]
        
        return {
            'original_length': len(text),
            'final_summary': final_summary,
            'compression_ratio': len(final_summary) / len(text),
            'section_summaries': section_summaries,
            'top_keywords': top_keywords,
            'num_sections': len(sections)
        }
    
    def _split_into_sections(self, text: str, max_section_length: int = 2000) -> List[str]:
        """Metni mantıklı bölümlere ayır"""
        # Paragraflara göre böl
        paragraphs = text.split('\n\n')
        
        sections = []
        current_section = []
        current_length = 0
        
        for paragraph in paragraphs:
            para_length = len(paragraph)
            
            if current_length + para_length > max_section_length and current_section:
                sections.append('\n\n'.join(current_section))
                current_section = [paragraph]
                current_length = para_length
            else:
                current_section.append(paragraph)
                current_length += para_length
        
        if current_section:
            sections.append('\n\n'.join(current_section))
        
        return sections

# API ve Web Interface için wrapper
class SummarizerAPI:
    """RESTful API wrapper"""
    
    def __init__(self):
        self.document_summarizer = DocumentSummarizer()
    
    def summarize_text(self, text: str, method: str = "hybrid", 
                      max_length: int = 150) -> Dict:
        """API endpoint için wrapper"""
        try:
            if len(text) > 5000:  # Uzun doküman
                result = self.document_summarizer.summarize_document(text)
                return {
                    'success': True,
                    'summary': result['final_summary'],
                    'method': 'document_summarization',
                    'compression_ratio': result['compression_ratio'],
                    'keywords': result['top_keywords'],
                    'sections': len(result['section_summaries'])
                }
            else:  # Normal metin
                result = self.document_summarizer.hybrid_summarizer.summarize(
                    text, method=method
                )
                return {
                    'success': True,
                    'summary': result.summary,
                    'method': result.method,
                    'compression_ratio': result.compression_ratio,
                    'keywords': result.keywords,
                    'processing_time': result.processing_time
                }
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

# Demo ve örnekler
def demo_summarization():
    """Özetleme sistemi demo"""
    
    # Test metni - Türkçe
    turkish_text = """
    Yapay zeka teknolojisi son yıllarda hızla gelişen bir alan haline geldi. Özellikle makine öğrenmesi 
    ve derin öğrenme algoritmaları sayesinde, bilgisayarlar insan benzeri görevleri yerine getirebilir 
    hale geldi. Bu teknolojiler, sağlık, finans, otomotiv ve eğitim gibi birçok sektörde devrim yaratıyor.
    
    Doğal dil işleme, yapay zekanın en önemli dallarından biridir. ChatGPT, BERT ve GPT-4 gibi modeller, 
    insan dilini anlama ve üretme konusunda çok başarılı sonuçlar elde etti. Bu modeller, metin özetleme, 
    çeviri, soru-cevap ve içerik üretimi gibi görevlerde kullanılıyor.
    
    Bilgisayarlı görü ise görüntü ve video analizi alanında ilerlemeler kaydediyor. Otonom araçlar, 
    tıbbi görüntü analizi, güvenlik sistemleri ve endüstriyel otomasyon gibi alanlarda bu teknoloji 
    aktif olarak kullanılıyor. Yapay zeka algoritmaları, insanlardan daha hızlı ve doğru sonuçlar 
    üretebiliyor.
    
    Gelecekte yapay zeka teknolojilerinin daha da yaygınlaşması bekleniyor. Etik ve güvenlik konuları 
    önemini korurken, bu teknolojilerin insanlığın faydasına kullanılması için çalışmalar devam ediyor.
    """
    
    print("🤖 Gelişmiş Metin Özetleme Sistemi Demo")
    print("=" * 60)
    
    # Summarizer'ı başlat
    api = SummarizerAPI()
    
    # Farklı yöntemleri test et
    methods = ["extractive", "abstractive", "hybrid"]
    
    for method in methods:
        print(f"\n📝 {method.upper()} Özetleme:")
        print("-" * 40)
        
        result = api.summarize_text(turkish_text, method=method)
        
        if result['success']:
            print(f"Özet: {result['summary']}")
            print(f"Sıkıştırma Oranı: {result['compression_ratio']:.2%}")
            print(f"Anahtar Kelimeler: {', '.join(result['keywords'][:5])}")
            if 'processing_time' in result:
                print(f"İşlem Süresi: {result['processing_time']:.2f}s")
        else:
            print(f"Hata: {result['error']}")
    
    # Uzun doküman testi
    long_text = turkish_text * 5  # Metni 5 kez tekrarla
    
    print(f"\n📄 UZUN DOKÜMAN Özetleme:")
    print("-" * 40)
    
    result = api.summarize_text(long_text)
    
    if result['success']:
        print(f"Orijinal Uzunluk: {len(long_text)} karakter")
        print(f"Özet Uzunluk: {len(result['summary'])} karakter")
        print(f"Sıkıştırma Oranı: {result['compression_ratio']:.2%}")
        print(f"Bölüm Sayısı: {result.get('sections', 'N/A')}")
        print(f"\nÖzet: {result['summary']}")
        print(f"\nTop Keywords: {', '.join(result['keywords'])}")

if __name__ == "__main__":
    # NLTK verilerini indir
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
    except:
        pass
    
    # Demo'yu çalıştır
    demo_summarization()