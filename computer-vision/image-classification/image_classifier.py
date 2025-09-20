import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import logging
from datetime import datetime

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, models
from PIL import Image

# Metrics and utilities
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import wandb  # For experiment tracking (optional)

@dataclass
class ClassificationResult:
    """Sınıflandırma sonucu"""
    predicted_class: str
    confidence: float
    all_probabilities: Dict[str, float]
    processing_time: float

class ImageDataset(Dataset):
    """
    Custom PyTorch Dataset sınıfı
    """
    
    def __init__(self, image_paths: List[str], labels: List[int], 
                 transform=None, class_names: List[str] = None):
        """
        Args:
            image_paths: Görüntü dosya yolları
            labels: Sınıf etiketleri
            transform: Görüntü dönüşümleri
            class_names: Sınıf isimleri
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.class_names = class_names or [f"class_{i}" for i in range(max(labels) + 1)]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class DataAugmentation:
    """
    Veri artırma (Data Augmentation) araçları
    """
    
    @staticmethod
    def get_train_transforms(input_size: int = 224):
        """Eğitim için veri artırma dönüşümleri"""
        return transforms.Compose([
            transforms.Resize((input_size + 32, input_size + 32)),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    @staticmethod
    def get_val_transforms(input_size: int = 224):
        """Validation için dönüşümler"""
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

class CustomCNN(nn.Module):
    """
    Custom CNN mimarisi
    """
    
    def __init__(self, num_classes: int, input_channels: int = 3):
        super(CustomCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        # Conv blocks
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Global average pooling
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x

class ImageClassifier:
    """
    Ana görüntü sınıflandırma sınıfı
    """
    
    def __init__(self, model_name: str = 'resnet50', num_classes: int = 2, 
                 input_size: int = 224, device: str = None):
        """
        Args:
            model_name: Model adı ('resnet50', 'efficientnet', 'custom')
            num_classes: Sınıf sayısı
            input_size: Giriş görüntü boyutu
            device: Cihaz (cuda/cpu)
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.input_size = input_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = None
        self.class_names = []
        self.training_history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        
        # Initialize model
        self._build_model()
        
        logging.info(f"ImageClassifier initialized: {model_name} on {self.device}")
    
    def _build_model(self):
        """Model oluştur"""
        if self.model_name == 'resnet50':
            self.model = models.resnet50(pretrained=True)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, self.num_classes)
            
        elif self.model_name == 'resnet18':
            self.model = models.resnet18(pretrained=True)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, self.num_classes)
            
        elif self.model_name == 'efficientnet':
            self.model = models.efficientnet_b0(pretrained=True)
            num_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_features, self.num_classes)
            
        elif self.model_name == 'custom':
            self.model = CustomCNN(self.num_classes)
            
        else:
            raise ValueError(f"Desteklenmeyen model: {self.model_name}")
        
        self.model.to(self.device)
    
    def load_dataset_from_directory(self, data_dir: str, test_split: float = 0.2, 
                                   val_split: float = 0.1) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Klasör yapısından veri seti yükle
        data_dir/
        ├── class1/
        │   ├── image1.jpg
        │   └── image2.jpg
        └── class2/
            ├── image3.jpg
            └── image4.jpg
        
        Args:
            data_dir: Veri klasörü
            test_split: Test veri oranı
            val_split: Validation veri oranı
            
        Returns:
            train_loader, val_loader, test_loader
        """
        data_path = Path(data_dir)
        
        # Sınıf isimlerini al
        self.class_names = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
        
        # Görüntü yolları ve etiketleri topla
        image_paths = []
        labels = []
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = data_path / class_name
            class_images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            
            image_paths.extend([str(img) for img in class_images])
            labels.extend([class_idx] * len(class_images))
        
        # Train/test split
        train_paths, test_paths, train_labels, test_labels = train_test_split(
            image_paths, labels, test_size=test_split, stratify=labels, random_state=42
        )
        
        # Train/validation split
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_paths, train_labels, test_size=val_split/(1-test_split), 
            stratify=train_labels, random_state=42
        )
        
        # Create datasets
        train_dataset = ImageDataset(
            train_paths, train_labels, 
            transform=DataAugmentation.get_train_transforms(self.input_size),
            class_names=self.class_names
        )
        
        val_dataset = ImageDataset(
            val_paths, val_labels,
            transform=DataAugmentation.get_val_transforms(self.input_size),
            class_names=self.class_names
        )
        
        test_dataset = ImageDataset(
            test_paths, test_labels,
            transform=DataAugmentation.get_val_transforms(self.input_size),
            class_names=self.class_names
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
        
        logging.info(f"Dataset loaded: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
        logging.info(f"Classes: {self.class_names}")
        
        return train_loader, val_loader, test_loader
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int = 10, learning_rate: float = 0.001, 
              save_best: bool = True, model_save_path: str = "best_model.pth"):
        """
        Model eğitimi
        
        Args:
            train_loader: Eğitim veri yükleyici
            val_loader: Validation veri yükleyici
            epochs: Epoch sayısı
            learning_rate: Öğrenme oranı
            save_best: En iyi modeli kaydet
            model_save_path: Model kayıt yolu
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_corrects = 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                train_corrects += torch.sum(preds == labels.data)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_corrects = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, preds = torch.max(outputs, 1)
                    val_corrects += torch.sum(preds == labels.data)
            
            # Calculate metrics
            train_loss = train_loss / len(train_loader)
            train_acc = train_corrects.double() / len(train_loader.dataset)
            val_loss = val_loss / len(val_loader)
            val_acc = val_corrects.double() / len(val_loader.dataset)
            
            # Update learning rate
            scheduler.step()
            
            # Save history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc.item())
            
            # Save best model
            if save_best and val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'class_names': self.class_names,
                    'model_name': self.model_name,
                    'num_classes': self.num_classes,
                    'input_size': self.input_size
                }, model_save_path)
            
            # Print progress
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            print('-' * 50)
        
        logging.info(f"Training completed. Best val accuracy: {best_val_acc:.4f}")
    
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """
        Model değerlendirmesi
        
        Args:
            test_loader: Test veri yükleyici
            
        Returns:
            Değerlendirme metrikleri
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        class_report = classification_report(all_labels, all_preds, 
                                           target_names=self.class_names, 
                                           output_dict=True)
        conf_matrix = confusion_matrix(all_labels, all_preds)
        
        return {
            'accuracy': accuracy,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'predictions': all_preds,
            'true_labels': all_labels
        }
    
    def predict_single_image(self, image_path: str) -> ClassificationResult:
        """
        Tek görüntü tahmin etme
        
        Args:
            image_path: Görüntü yolu
            
        Returns:
            Tahmin sonucu
        """
        start_time = datetime.now()
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        transform = DataAugmentation.get_val_transforms(self.input_size)
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # Prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            # Get top prediction
            confidence, predicted_idx = torch.max(probabilities, 1)
            predicted_class = self.class_names[predicted_idx.item()]
            
            # Get all probabilities
            all_probs = {
                self.class_names[i]: prob.item() 
                for i, prob in enumerate(probabilities[0])
            }
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return ClassificationResult(
            predicted_class=predicted_class,
            confidence=confidence.item(),
            all_probabilities=all_probs,
            processing_time=processing_time
        )
    
    def visualize_training_history(self):
        """Eğitim geçmişini görselleştir"""
        plt.figure(figsize=(15, 5))
        
        # Loss plot
        plt.subplot(1, 3, 1)
        plt.plot(self.training_history['train_loss'], label='Train Loss')
        plt.plot(self.training_history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Accuracy plot
        plt.subplot(1, 3, 2)
        plt.plot(self.training_history['val_acc'], label='Validation Accuracy')
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Learning rate (if available)
        plt.subplot(1, 3, 3)
        plt.text(0.5, 0.5, f'Final Val Acc: {self.training_history["val_acc"][-1]:.4f}', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
        plt.title('Training Summary')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_predictions(self, test_loader: DataLoader, num_images: int = 16):
        """Tahminleri görselleştir"""
        self.model.eval()
        
        # Get a batch of test images
        data_iter = iter(test_loader)
        images, labels = next(data_iter)
        images, labels = images.to(self.device), labels.to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(images)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
        
        # Denormalize images for visualization
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        
        # Plot images
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        
        for i in range(min(num_images, len(images))):
            ax = axes[i // 4, i % 4]
            
            # Denormalize image
            img = images[i].cpu()
            for t, m, s in zip(img, mean, std):
                t.mul_(s).add_(m)
            img = torch.clamp(img, 0, 1)
            
            # Convert to numpy and transpose
            img_np = img.permute(1, 2, 0).numpy()
            
            ax.imshow(img_np)
            
            # Title with prediction info
            true_class = self.class_names[labels[i]]
            pred_class = self.class_names[predicted[i]]
            confidence = probabilities[i][predicted[i]].item()
            
            color = 'green' if true_class == pred_class else 'red'
            ax.set_title(f'True: {true_class}\nPred: {pred_class}\nConf: {confidence:.3f}', 
                        color=color, fontsize=10)
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, save_path: str):
        """Modeli kaydet"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'class_names': self.class_names,
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'input_size': self.input_size,
            'training_history': self.training_history
        }, save_path)
        
        logging.info(f"Model saved: {save_path}")
    
    def load_model(self, model_path: str):
        """Modeli yükle"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.class_names = checkpoint['class_names']
        self.num_classes = checkpoint['num_classes']
        self.input_size = checkpoint['input_size']
        
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
        
        logging.info(f"Model loaded: {model_path}")


# Demo fonksiyonu
def demo_image_classification():
    """
    Görüntü sınıflandırma demo fonksiyonu
    """
    print("🖼️  Image Classification Demo Başlatılıyor...")
    
    # Create sample dataset structure
    demo_dir = "demo_dataset"
    classes = ["cats", "dogs"]
    
    os.makedirs(demo_dir, exist_ok=True)
    for class_name in classes:
        os.makedirs(f"{demo_dir}/{class_name}", exist_ok=True)
    
    print("📁 Demo dataset yapısı oluşturuldu")
    print("🔄 Gerçek bir dataset ile eğitim yapmak için:")
    print(f"   {demo_dir}/cats/ klasörüne kedi resimlerini")
    print(f"   {demo_dir}/dogs/ klasörüne köpek resimlerini yerleştirin")
    
    # Initialize classifier
    classifier = ImageClassifier(model_name='resnet18', num_classes=2)
    
    print(f"🚀 Model yüklendi: {classifier.model_name}")
    print(f"🎯 Sınıf sayısı: {classifier.num_classes}")
    print(f"💻 Cihaz: {classifier.device}")
    
    # Model özeti
    total_params = sum(p.numel() for p in classifier.model.parameters())
    trainable_params = sum(p.numel() for p in classifier.model.parameters() if p.requires_grad)
    
    print(f"📊 Model parametreleri:")
    print(f"   Toplam: {total_params:,}")
    print(f"   Eğitilebilir: {trainable_params:,}")
    
    # Cleanup
    import shutil
    shutil.rmtree(demo_dir)
    
    print("\n✨ Demo tamamlandı!")
    print("💡 Gerçek bir proje için:")
    print("   1. Veri setinizi hazırlayın")
    print("   2. load_dataset_from_directory() ile yükleyin")
    print("   3. train() ile modeli eğitin")
    print("   4. evaluate() ile test edin")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run demo
    demo_image_classification()