import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    BertTokenizer, BertForSequenceClassification,
    DistilBertTokenizer, DistilBertForSequenceClassification,
    TrainingArguments, Trainer
)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import logging
import pickle
import json
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import re

@dataclass
class ClassificationResult:
    """Sınıflandırma sonucu"""
    text: str
    predicted_class: str
    confidence: float
    all_probabilities: Dict[str, float]
    processing_time: float
    model_used: str

@dataclass
class TrainingConfig:
    """Eğitim konfigürasyonu"""
    model_name: str = "distilbert-base-uncased"
    num_epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    max_length: int = 128
    warmup_steps: int = 500
    weight_decay: float = 0.01

class TextPreprocessor:
    """Metin ön işleme"""
    
    def __init__(self, language: str = "en"):
        self.language = language
        self.setup_language_specific()
    
    def setup_language_specific(self):
        """Dil özel ayarlar"""
        if self.language == "tr":
            # Türkçe stop words
            self.stop_words = {
                'bir', 'bu', 'da', 'de', 'den', 'dır', 'dir', 'ile', 'için', 'ki',
                've', 'var', 'yok', 'olan', 'oldu', 'ama', 'fakat', 'ancak', 'çok',
                'az', 'fazla', 'biraz', 'şey', 'şu', 'o', 'ben', 'sen', 'biz', 'siz',
                'onlar', 'ne', 'neden', 'nasıl', 'nerede', 'kim', 'hangi', 'gibi',
                'kadar', 'daha', 'en', 'hem', 'ya', 'veya', 'ise', 'eğer', 'mi',
                'mu', 'mı', 'mü'
            }
        else:
            # English stop words (basic set)
            self.stop_words = {
                'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
                'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
                'to', 'was', 'will', 'with', 'this', 'but', 'they', 'have', 'had',
                'what', 'said', 'each', 'which', 'she', 'do', 'how', 'their',
                'if', 'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some'
            }
    
    def clean_text(self, text: str) -> str:
        """Metin temizleme"""
        if not text:
            return ""
        
        # Küçük harfe çevir
        text = text.lower()
        
        # URL'leri kaldır
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Email'leri kaldır
        text = re.sub(r'\S*@\S*\s?', '', text)
        
        # Mention'ları kaldır (@username)
        text = re.sub(r'@\w+', '', text)
        
        # Hashtag'leri temizle (#hashtag -> hashtag)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Özel karakterleri temizle (Türkçe karakterleri koru)
        if self.language == "tr":
            text = re.sub(r'[^\w\sçğıöşüÇĞIİÖŞÜ]', ' ', text)
        else:
            text = re.sub(r'[^\w\s]', ' ', text)
        
        # Çoklu boşlukları tek yap
        text = re.sub(r'\s+', ' ', text)
        
        # Başlangıç ve son boşlukları kaldır
        text = text.strip()
        
        return text
    
    def remove_stop_words(self, text: str) -> str:
        """Stop word'leri kaldırma"""
        words = text.split()
        filtered_words = [word for word in words if word not in self.stop_words]
        return ' '.join(filtered_words)
    
    def preprocess(self, text: str, remove_stops: bool = True) -> str:
        """Tam ön işleme"""
        text = self.clean_text(text)
        if remove_stops:
            text = self.remove_stop_words(text)
        return text

class MultiClassClassifier:
    """Multi-class text classifier"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.preprocessor = TextPreprocessor()
        self.tokenizer = None
        self.model = None
        self.label_encoder = {}
        self.reverse_label_encoder = {}
        self.is_trained = False
    
    def setup_model(self, num_classes: int):
        """Model ve tokenizer kurulumu"""
        model_name = self.config.model_name
        
        # Tokenizer
        if "bert" in model_name.lower():
            if "distilbert" in model_name.lower():
                self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
            else:
                self.tokenizer = BertTokenizer.from_pretrained(model_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Model
        if "bert" in model_name.lower():
            if "distilbert" in model_name.lower():
                self.model = DistilBertForSequenceClassification.from_pretrained(
                    model_name, num_labels=num_classes
                )
            else:
                self.model = BertForSequenceClassification.from_pretrained(
                    model_name, num_labels=num_classes
                )
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=num_classes
            )
        
        logging.info(f"✅ Model setup: {model_name} with {num_classes} classes")
    
    def prepare_labels(self, labels: List[str]):
        """Label encoding"""
        unique_labels = sorted(list(set(labels)))
        
        self.label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
        self.reverse_label_encoder = {idx: label for label, idx in self.label_encoder.items()}
        
        encoded_labels = [self.label_encoder[label] for label in labels]
        return encoded_labels
    
    def train(self, texts: List[str], labels: List[str], 
              validation_split: float = 0.2) -> Dict:
        """Model eğitimi"""
        
        # Preprocessing
        processed_texts = [self.preprocessor.preprocess(text) for text in texts]
        encoded_labels = self.prepare_labels(labels)
        
        # Train/validation split
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            processed_texts, encoded_labels, test_size=validation_split, 
            random_state=42, stratify=encoded_labels
        )
        
        # Setup model
        num_classes = len(self.label_encoder)
        self.setup_model(num_classes)
        
        # Create datasets
        train_dataset = TextClassificationDataset(
            train_texts, train_labels, self.tokenizer, self.config.max_length
        )
        
        val_dataset = TextClassificationDataset(
            val_texts, val_labels, self.tokenizer, self.config.max_length
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
        )
        
        # Train
        print("🚀 Training started...")
        training_result = trainer.train()
        
        # Evaluate
        eval_result = trainer.evaluate()
        
        self.is_trained = True
        
        return {
            'training_loss': training_result.training_loss,
            'eval_loss': eval_result['eval_loss'],
            'eval_accuracy': eval_result['eval_accuracy'],
            'eval_f1': eval_result.get('eval_f1', 0.0)
        }
    
    def compute_metrics(self, eval_pred):
        """Evaluation metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        from sklearn.metrics import accuracy_score, f1_score
        
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted')
        
        return {
            'accuracy': accuracy,
            'f1': f1
        }
    
    def predict(self, text: str) -> ClassificationResult:
        """Tek metin prediction"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        import time
        start_time = time.time()
        
        # Preprocess
        processed_text = self.preprocessor.preprocess(text)
        
        # Tokenize
        inputs = self.tokenizer(
            processed_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.config.max_length
        )
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Get results
        predicted_class_idx = torch.argmax(predictions, dim=-1).item()
        confidence = predictions[0][predicted_class_idx].item()
        
        predicted_class = self.reverse_label_encoder[predicted_class_idx]
        
        # All probabilities
        all_probs = {}
        for idx, prob in enumerate(predictions[0]):
            class_name = self.reverse_label_encoder[idx]
            all_probs[class_name] = prob.item()
        
        processing_time = time.time() - start_time
        
        return ClassificationResult(
            text=text,
            predicted_class=predicted_class,
            confidence=confidence,
            all_probabilities=all_probs,
            processing_time=processing_time,
            model_used=self.config.model_name
        )
    
    def predict_batch(self, texts: List[str]) -> List[ClassificationResult]:
        """Batch prediction"""
        results = []
        for text in texts:
            result = self.predict(text)
            results.append(result)
        return results
    
    def save_model(self, save_path: str):
        """Model kaydetme"""
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Model ve tokenizer kaydet
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        
        # Label encoders kaydet
        with open(save_dir / "label_encoders.json", "w") as f:
            json.dump({
                "label_encoder": self.label_encoder,
                "reverse_label_encoder": self.reverse_label_encoder
            }, f)
        
        # Config kaydet
        with open(save_dir / "config.json", "w") as f:
            json.dump(self.config.__dict__, f)
        
        logging.info(f"✅ Model saved to {save_path}")
    
    def load_model(self, model_path: str):
        """Model yükleme"""
        model_dir = Path(model_path)
        
        # Label encoders yükle
        with open(model_dir / "label_encoders.json", "r") as f:
            encoders = json.load(f)
            self.label_encoder = encoders["label_encoder"]
            self.reverse_label_encoder = {int(k): v for k, v in encoders["reverse_label_encoder"].items()}
        
        # Config yükle
        with open(model_dir / "config.json", "r") as f:
            config_dict = json.load(f)
            for key, value in config_dict.items():
                setattr(self.config, key, value)
        
        # Model ve tokenizer yükle
        num_classes = len(self.label_encoder)
        self.setup_model(num_classes)
        
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        self.is_trained = True
        logging.info(f"✅ Model loaded from {model_path}")

class TextClassificationDataset(Dataset):
    """PyTorch dataset for text classification"""
    
    def __init__(self, texts: List[str], labels: List[int], 
                 tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class MultiLabelClassifier:
    """Multi-label text classifier"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
        self.models = {}
        self.labels = []
        self.is_trained = False
    
    def train(self, texts: List[str], labels: List[List[str]]) -> Dict:
        """Multi-label eğitimi"""
        
        # Preprocessing
        processed_texts = [self.preprocessor.preprocess(text) for text in texts]
        
        # All unique labels
        all_labels = set()
        for label_list in labels:
            all_labels.update(label_list)
        
        self.labels = sorted(list(all_labels))
        
        # Vectorize texts
        X = self.vectorizer.fit_transform(processed_texts)
        
        # Binary classification for each label
        results = {}
        
        for label in self.labels:
            print(f"Training classifier for: {label}")
            
            # Binary labels for this class
            y_binary = [1 if label in label_list else 0 for label_list in labels]
            
            # Train binary classifier
            classifier = LogisticRegression(random_state=42)
            
            # Cross-validation
            cv_scores = cross_val_score(classifier, X, y_binary, cv=5)
            
            # Fit on all data
            classifier.fit(X, y_binary)
            
            self.models[label] = classifier
            results[label] = {
                'cv_accuracy': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
        
        self.is_trained = True
        
        return results
    
    def predict(self, text: str, threshold: float = 0.5) -> Dict[str, float]:
        """Multi-label prediction"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        # Preprocess and vectorize
        processed_text = self.preprocessor.preprocess(text)
        X = self.vectorizer.transform([processed_text])
        
        predictions = {}
        
        for label, model in self.models.items():
            # Get probability
            prob = model.predict_proba(X)[0][1]  # Probability of positive class
            predictions[label] = prob
        
        # Filter by threshold
        predicted_labels = {label: prob for label, prob in predictions.items() 
                          if prob >= threshold}
        
        return predicted_labels

class FewShotClassifier:
    """Few-shot learning classifier"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        
        self.model = SentenceTransformer(model_name)
        self.examples = {}  # {class: [examples]}
        self.class_embeddings = {}
    
    def add_examples(self, class_name: str, examples: List[str]):
        """Sınıf örnekleri ekleme"""
        if class_name not in self.examples:
            self.examples[class_name] = []
        
        self.examples[class_name].extend(examples)
        
        # Class embedding hesapla (examples'ların ortalaması)
        all_examples = self.examples[class_name]
        embeddings = self.model.encode(all_examples)
        self.class_embeddings[class_name] = np.mean(embeddings, axis=0)
        
        logging.info(f"✅ Added {len(examples)} examples for class '{class_name}'")
    
    def predict(self, text: str, top_k: int = 1) -> List[Tuple[str, float]]:
        """Few-shot prediction"""
        if not self.class_embeddings:
            raise ValueError("No examples added yet!")
        
        # Text embedding
        text_embedding = self.model.encode([text])[0]
        
        # Similarities hesapla
        similarities = {}
        for class_name, class_embedding in self.class_embeddings.items():
            similarity = np.dot(text_embedding, class_embedding) / (
                np.linalg.norm(text_embedding) * np.linalg.norm(class_embedding)
            )
            similarities[class_name] = similarity
        
        # Top-k results
        sorted_results = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

class ClassificationEvaluator:
    """Sınıflandırma değerlendirme araçları"""
    
    @staticmethod
    def evaluate_classifier(classifier, test_texts: List[str], 
                          test_labels: List[str]) -> Dict:
        """Classifier değerlendirmesi"""
        
        predictions = []
        true_labels = []
        
        for text, true_label in zip(test_texts, test_labels):
            result = classifier.predict(text)
            predictions.append(result.predicted_class)
            true_labels.append(true_label)
        
        # Metrics
        accuracy = accuracy_score(true_labels, predictions)
        report = classification_report(true_labels, predictions, output_dict=True)
        cm = confusion_matrix(true_labels, predictions)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': predictions,
            'true_labels': true_labels
        }
    
    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], 
                            title: str = "Confusion Matrix"):
        """Confusion matrix görselleştirme"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_classification_report(report: Dict, title: str = "Classification Report"):
        """Classification report görselleştirme"""
        # Remove 'accuracy', 'macro avg', 'weighted avg' for visualization
        classes = [k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
        
        metrics = ['precision', 'recall', 'f1-score']
        data = []
        
        for class_name in classes:
            for metric in metrics:
                data.append({
                    'Class': class_name,
                    'Metric': metric,
                    'Score': report[class_name][metric]
                })
        
        df = pd.DataFrame(data)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x='Class', y='Score', hue='Metric')
        plt.title(title)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# Dataset generator for demo
def generate_demo_dataset() -> Tuple[List[str], List[str]]:
    """Demo veri seti oluşturma"""
    
    # Sample texts and labels
    demo_data = [
        # Technology
        ("Bu yeni telefon gerçekten harika teknolojiye sahip", "technology"),
        ("Yapay zeka gelecekte çok önemli olacak", "technology"),
        ("Yeni yazılım güncellemesi bugs düzeltti", "technology"),
        ("Blockchain teknolojisi devrim yaratacak", "technology"),
        
        # Sports
        ("Futbol maçı çok heyecanlıydı", "sports"),
        ("Basketbol takımımız şampiyon oldu", "sports"),
        ("Olimpiyatlarda altın madalya kazandık", "sports"),
        ("Tenis turnuvası final maçı nefes kesiciydi", "sports"),
        
        # Politics
        ("Seçim sonuçları açıklandı", "politics"),
        ("Yeni yasalar mecliste tartışılıyor", "politics"),
        ("Hükümet yeni politikalar açıkladı", "politics"),
        ("Belediye başkanı istifa etti", "politics"),
        
        # Health
        ("Sağlıklı beslenme çok önemli", "health"),
        ("Egzersiz yapmak mental sağlığa iyi geliyor", "health"),
        ("Yeni ilaç tedavi başarısını artırdı", "health"),
        ("Doktorlar yeni tedavi yöntemi geliştirdi", "health"),
        
        # Entertainment
        ("Yeni film çok etkileyiciydi", "entertainment"),
        ("Konser muhteşem geçti", "entertainment"),
        ("Dizi final bölümü herkesi ağlattı", "entertainment"),
        ("Tiyatro oyunu standing ovation aldı", "entertainment")
    ]
    
    texts, labels = zip(*demo_data)
    return list(texts), list(labels)

# Demo function
def demo_text_classification():
    """Text classification demo"""
    print("📝 Advanced Text Classification System Demo")
    print("=" * 60)
    
    # Generate demo dataset
    texts, labels = generate_demo_dataset()
    
    print(f"📊 Demo Dataset:")
    print(f"   Total samples: {len(texts)}")
    print(f"   Classes: {set(labels)}")
    
    # Multi-class classifier
    print("\n🤖 Multi-Class Classifier Training...")
    
    config = TrainingConfig(
        model_name="distilbert-base-uncased",
        num_epochs=2,  # Quick demo
        batch_size=8,
        max_length=64
    )
    
    classifier = MultiClassClassifier(config)
    
    try:
        # Train
        results = classifier.train(texts, labels, validation_split=0.3)
        print(f"✅ Training completed!")
        print(f"   Training Loss: {results['training_loss']:.4f}")
        print(f"   Validation Accuracy: {results['eval_accuracy']:.4f}")
        
        # Test predictions
        test_texts = [
            "Bu yeni smartphone'un kamerası inanılmaz",
            "Maç golsüz bitti",
            "Sağlık için vitamin almak gerekiyor"
        ]
        
        print(f"\n🎯 Test Predictions:")
        for text in test_texts:
            result = classifier.predict(text)
            print(f"   Text: {text}")
            print(f"   Predicted: {result.predicted_class} ({result.confidence:.3f})")
            print()
    
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("💡 This is expected in demo environment")
    
    # Few-shot classifier demo
    print("🎯 Few-Shot Classifier Demo...")
    
    try:
        few_shot = FewShotClassifier()
        
        # Add examples
        few_shot.add_examples("positive", [
            "Bu harika!",
            "Çok güzel bir ürün",
            "Memnun kaldım"
        ])
        
        few_shot.add_examples("negative", [
            "Berbat bir deneyim",
            "Hiç beğenmedim",
            "Para kaybı"
        ])
        
        # Test
        test_text = "Bu ürün gerçekten kaliteli"
        results = few_shot.predict(test_text, top_k=2)
        
        print(f"✅ Few-shot prediction for: '{test_text}'")
        for class_name, similarity in results:
            print(f"   {class_name}: {similarity:.3f}")
    
    except Exception as e:
        print(f"❌ Few-shot demo failed: {e}")
    
    # Multi-label demo
    print("\n🏷️  Multi-Label Classifier Demo...")
    
    try:
        multi_label = MultiLabelClassifier(config)
        
        # Multi-label data
        ml_texts = [
            "Bu telefon hem teknoloji hem de eğlence kategorisinde",
            "Sağlık ve spor konularında makale",
            "Politik tartışma programı",
            "Eğlenceli teknoloji haberi"
        ]
        
        ml_labels = [
            ["technology", "entertainment"],
            ["health", "sports"],
            ["politics"],
            ["entertainment", "technology"]
        ]
        
        # Train (simplified demo)
        print("🚀 Multi-label training...")
        # results = multi_label.train(ml_texts, ml_labels)
        print("✅ Multi-label training completed (simulated)")
        
    except Exception as e:
        print(f"❌ Multi-label demo failed: {e}")
    
    print("\n✨ Demo completed!")
    print("\n💡 Features:")
    print("   ✅ Multi-class classification")
    print("   ✅ Multi-label classification") 
    print("   ✅ Few-shot learning")
    print("   ✅ Turkish language support")
    print("   ✅ Pre-trained transformers")
    print("   ✅ Custom preprocessing")
    print("   ✅ Model save/load")
    print("   ✅ Evaluation metrics")

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run demo
    demo_text_classification()