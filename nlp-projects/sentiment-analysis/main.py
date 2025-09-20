import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

class TurkishSentimentAnalyzer:
    """
    Türkçe metinler için sentiment analizi yapan sınıf
    """
    
    def __init__(self, model_name="dbmdz/bert-base-turkish-cased"):
        """
        Modeli yükle ve tokenizer'ı başlat
        
        Args:
            model_name (str): Kullanılacak model adı
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.classifier = None
        self.load_model()
    
    def load_model(self):
        """Pre-trained modeli yükle"""
        try:
            print(f"Model yükleniyor: {self.model_name}")
            
            # Transformer pipeline kullan
            self.classifier = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
                tokenizer="cardiffnlp/twitter-xlm-roberta-base-sentiment"
            )
            
            print("Model başarıyla yüklendi!")
            
        except Exception as e:
            print(f"Model yüklenirken hata: {e}")
            # Fallback olarak basit model kullan
            self.load_fallback_model()
    
    def load_fallback_model(self):
        """Basit fallback modeli"""
        print("Fallback model yükleniyor...")
        
        # Basit kelime tabanlı sentiment analizi
        self.positive_words = [
            'güzel', 'harika', 'mükemmel', 'süper', 'excellent', 'iyi',
            'başarılı', 'etkili', 'kaliteli', 'faydalı', 'yararlı'
        ]
        
        self.negative_words = [
            'kötü', 'berbat', 'rezalet', 'korkunç', 'terrible', 'bad',
            'başarısız', 'etkisiz', 'kalitesiz', 'zararlı', 'gereksiz'
        ]
    
    def analyze_text(self, text):
        """
        Tek bir metin için sentiment analizi
        
        Args:
            text (str): Analiz edilecek metin
            
        Returns:
            dict: Sentiment sonucu
        """
        if self.classifier:
            try:
                result = self.classifier(text)
                return {
                    'text': text,
                    'label': result[0]['label'],
                    'score': result[0]['score'],
                    'sentiment': self._map_label(result[0]['label'])
                }
            except:
                return self._fallback_analysis(text)
        else:
            return self._fallback_analysis(text)
    
    def _map_label(self, label):
        """Model etiketlerini anlamlı isimlere çevir"""
        mapping = {
            'POSITIVE': 'Pozitif',
            'NEGATIVE': 'Negatif',
            'NEUTRAL': 'Nötr',
            'LABEL_0': 'Negatif',
            'LABEL_1': 'Nötr',
            'LABEL_2': 'Pozitif'
        }
        return mapping.get(label, label)
    
    def _fallback_analysis(self, text):
        """Basit kelime tabanlı analiz"""
        text_lower = text.lower()
        
        positive_count = sum(1 for word in self.positive_words if word in text_lower)
        negative_count = sum(1 for word in self.negative_words if word in text_lower)
        
        if positive_count > negative_count:
            sentiment = 'Pozitif'
            score = 0.7
        elif negative_count > positive_count:
            sentiment = 'Negatif'
            score = 0.7
        else:
            sentiment = 'Nötr'
            score = 0.5
        
        return {
            'text': text,
            'label': sentiment,
            'score': score,
            'sentiment': sentiment
        }
    
    def analyze_batch(self, texts):
        """
        Birden fazla metin için batch analiz
        
        Args:
            texts (list): Metin listesi
            
        Returns:
            list: Analiz sonuçları
        """
        results = []
        for text in texts:
            result = self.analyze_text(text)
            results.append(result)
        
        return results
    
    def visualize_results(self, results):
        """
        Analiz sonuçlarını görselleştir
        
        Args:
            results (list): Analiz sonuçları
        """
        # Sentiment dağılımı
        sentiments = [r['sentiment'] for r in results]
        sentiment_counts = pd.Series(sentiments).value_counts()
        
        plt.figure(figsize=(12, 8))
        
        # Pie chart
        plt.subplot(2, 2, 1)
        plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')
        plt.title('Sentiment Dağılımı')
        
        # Bar chart
        plt.subplot(2, 2, 2)
        sentiment_counts.plot(kind='bar', color=['green', 'red', 'gray'])
        plt.title('Sentiment Sayıları')
        plt.xticks(rotation=45)
        
        # Score dağılımı
        plt.subplot(2, 2, 3)
        scores = [r['score'] for r in results]
        plt.hist(scores, bins=20, alpha=0.7)
        plt.title('Güven Skoru Dağılımı')
        plt.xlabel('Güven Skoru')
        plt.ylabel('Frekans')
        
        # Sentiment vs Score
        plt.subplot(2, 2, 4)
        df = pd.DataFrame(results)
        sns.boxplot(data=df, x='sentiment', y='score')
        plt.title('Sentiment vs Güven Skoru')
        
        plt.tight_layout()
        plt.show()
    
    def export_results(self, results, filename='sentiment_results.csv'):
        """
        Sonuçları CSV dosyasına kaydet
        
        Args:
            results (list): Analiz sonuçları
            filename (str): Dosya adı
        """
        df = pd.DataFrame(results)
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"Sonuçlar {filename} dosyasına kaydedildi.")


def main():
    """Ana fonksiyon"""
    parser = argparse.ArgumentParser(description='Türkçe Sentiment Analysis')
    parser.add_argument('--text', type=str, help='Analiz edilecek tek metin')
    parser.add_argument('--file', type=str, help='Analiz edilecek metin dosyası')
    parser.add_argument('--model', type=str, default='cardiffnlp/twitter-xlm-roberta-base-sentiment',
                       help='Kullanılacak model')
    parser.add_argument('--visualize', action='store_true', help='Sonuçları görselleştir')
    parser.add_argument('--export', type=str, help='Sonuçları CSV olarak kaydet')
    
    args = parser.parse_args()
    
    # Analyzer'ı başlat
    analyzer = TurkishSentimentAnalyzer(args.model)
    
    # Test metinleri
    if args.text:
        # Tek metin analizi
        result = analyzer.analyze_text(args.text)
        print(f"\nMetin: {result['text']}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Güven Skoru: {result['score']:.3f}")
        
    elif args.file:
        # Dosyadan okuma
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f.readlines() if line.strip()]
            
            results = analyzer.analyze_batch(texts)
            
            print(f"\n{len(results)} metin analiz edildi:")
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['sentiment']} ({result['score']:.3f}): {result['text'][:50]}...")
            
            # Görselleştirme
            if args.visualize:
                analyzer.visualize_results(results)
            
            # Export
            if args.export:
                analyzer.export_results(results, args.export)
                
        except FileNotFoundError:
            print(f"Dosya bulunamadı: {args.file}")
    
    else:
        # Demo metinler
        demo_texts = [
            "Bu proje gerçekten harika!",
            "Çok kötü bir deneyim yaşadım.",
            "Normal bir gün geçirdim.",
            "Mükemmel bir uygulama, herkese tavsiye ederim.",
            "Berbat bir hizmet, asla kullanmam.",
            "Fiyat performans açısından iyi.",
            "Zaman kaybı oldu.",
            "İlginç bir yaklaşım.",
        ]
        
        print("Demo analiz başlatılıyor...")
        results = analyzer.analyze_batch(demo_texts)
        
        print("\nDemo Sonuçları:")
        print("-" * 80)
        for result in results:
            print(f"Sentiment: {result['sentiment']:>8} | "
                  f"Score: {result['score']:.3f} | "
                  f"Text: {result['text']}")
        
        # Görselleştirme
        if args.visualize:
            analyzer.visualize_results(results)


if __name__ == "__main__":
    main()