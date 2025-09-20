import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional, Callable
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import tensorflow as tf
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import plotly.graph_objects as go
import plotly.express as px
from dataclasses import dataclass
import time
import json

@dataclass
class NetworkConfig:
    """Neural network konfigürasyonu"""
    input_size: int
    hidden_sizes: List[int]
    output_size: int
    activation: str = "relu"
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 32
    dropout_rate: float = 0.0

class NeuralNetworkFromScratch:
    """Sıfırdan Neural Network implementasyonu"""
    
    def __init__(self, layers: List[int], activation: str = "relu", learning_rate: float = 0.001):
        """
        Args:
            layers: Her layer'daki neuron sayısı [input, hidden1, hidden2, ..., output]
            activation: Aktivasyon fonksiyonu
            learning_rate: Öğrenme oranı
        """
        self.layers = layers
        self.activation = activation
        self.learning_rate = learning_rate
        
        # Weights ve biases'ı initialize et
        self.weights = []
        self.biases = []
        
        for i in range(len(layers) - 1):
            # Xavier initialization
            w = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i])
            b = np.zeros((1, layers[i+1]))
            
            self.weights.append(w)
            self.biases.append(b)
        
        # Training history
        self.history = {'loss': [], 'accuracy': []}
    
    def _activation_function(self, x: np.ndarray, derivative: bool = False) -> np.ndarray:
        """Aktivasyon fonksiyonları"""
        if self.activation == "relu":
            if derivative:
                return (x > 0).astype(float)
            return np.maximum(0, x)
        
        elif self.activation == "sigmoid":
            sig = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
            if derivative:
                return sig * (1 - sig)
            return sig
        
        elif self.activation == "tanh":
            if derivative:
                return 1 - np.tanh(x) ** 2
            return np.tanh(x)
        
        elif self.activation == "leaky_relu":
            if derivative:
                return np.where(x > 0, 1, 0.01)
            return np.where(x > 0, x, 0.01 * x)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax fonksiyonu"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Forward propagation"""
        activations = [X]
        
        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            
            if i == len(self.weights) - 1:  # Output layer
                if self.layers[-1] == 1:  # Regression
                    a = z
                else:  # Classification
                    a = self._softmax(z)
            else:  # Hidden layers
                a = self._activation_function(z)
            
            activations.append(a)
        
        return activations[-1], activations
    
    def backward(self, X: np.ndarray, y: np.ndarray, activations: List[np.ndarray]):
        """Backward propagation"""
        m = X.shape[0]
        
        # Output layer error
        if self.layers[-1] == 1:  # Regression
            dZ = activations[-1] - y.reshape(-1, 1)
        else:  # Classification
            y_one_hot = np.eye(self.layers[-1])[y]
            dZ = activations[-1] - y_one_hot
        
        # Gradients
        dW = []
        dB = []
        
        for i in range(len(self.weights) - 1, -1, -1):
            dW_i = (1/m) * np.dot(activations[i].T, dZ)
            dB_i = (1/m) * np.sum(dZ, axis=0, keepdims=True)
            
            dW.insert(0, dW_i)
            dB.insert(0, dB_i)
            
            if i > 0:  # Not input layer
                dA = np.dot(dZ, self.weights[i].T)
                dZ = dA * self._activation_function(activations[i], derivative=True)
        
        return dW, dB
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, verbose: bool = True):
        """Training loop"""
        for epoch in range(epochs):
            # Forward pass
            predictions, activations = self.forward(X)
            
            # Loss calculation
            if self.layers[-1] == 1:  # Regression
                loss = np.mean((predictions - y.reshape(-1, 1)) ** 2)
                accuracy = 0  # MSE for regression
            else:  # Classification
                # Cross-entropy loss
                y_one_hot = np.eye(self.layers[-1])[y]
                loss = -np.mean(np.sum(y_one_hot * np.log(predictions + 1e-15), axis=1))
                
                # Accuracy
                pred_classes = np.argmax(predictions, axis=1)
                accuracy = np.mean(pred_classes == y)
            
            # Backward pass
            dW, dB = self.backward(X, y, activations)
            
            # Update weights
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * dW[i]
                self.biases[i] -= self.learning_rate * dB[i]
            
            # Save history
            self.history['loss'].append(loss)
            self.history['accuracy'].append(accuracy)
            
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prediction"""
        predictions, _ = self.forward(X)
        
        if self.layers[-1] == 1:  # Regression
            return predictions.flatten()
        else:  # Classification
            return np.argmax(predictions, axis=1)

class PyTorchNetwork(nn.Module):
    """PyTorch Neural Network"""
    
    def __init__(self, config: NetworkConfig):
        super(PyTorchNetwork, self).__init__()
        
        self.config = config
        layers = []
        
        # Input layer
        layers.append(nn.Linear(config.input_size, config.hidden_sizes[0]))
        layers.append(self._get_activation())
        
        # Hidden layers
        for i in range(len(config.hidden_sizes) - 1):
            layers.append(nn.Linear(config.hidden_sizes[i], config.hidden_sizes[i+1]))
            layers.append(self._get_activation())
            
            if config.dropout_rate > 0:
                layers.append(nn.Dropout(config.dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(config.hidden_sizes[-1], config.output_size))
        
        self.network = nn.Sequential(*layers)
        
        # Training history
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    def _get_activation(self) -> nn.Module:
        """Aktivasyon fonksiyonu seç"""
        activations = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU()
        }
        return activations.get(self.config.activation, nn.ReLU())
    
    def forward(self, x):
        return self.network(x)
    
    def train_model(self, train_loader: DataLoader, val_loader: DataLoader = None, 
                   task_type: str = "classification"):
        """Model eğitimi"""
        
        # Loss function
        if task_type == "classification":
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()
        
        # Optimizer
        optimizer = optim.Adam(self.parameters(), lr=self.config.learning_rate)
        
        # Training loop
        for epoch in range(self.config.epochs):
            # Training phase
            self.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                
                outputs = self(batch_X)
                loss = criterion(outputs, batch_y)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                if task_type == "classification":
                    _, predicted = torch.max(outputs.data, 1)
                    train_total += batch_y.size(0)
                    train_correct += (predicted == batch_y).sum().item()
            
            # Validation phase
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            if val_loader:
                self.eval()
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = self(batch_X)
                        loss = criterion(outputs, batch_y)
                        
                        val_loss += loss.item()
                        
                        if task_type == "classification":
                            _, predicted = torch.max(outputs.data, 1)
                            val_total += batch_y.size(0)
                            val_correct += (predicted == batch_y).sum().item()
            
            # History güncelle
            self.history['train_loss'].append(train_loss / len(train_loader))
            
            if task_type == "classification":
                self.history['train_acc'].append(train_correct / train_total)
            
            if val_loader:
                self.history['val_loss'].append(val_loss / len(val_loader))
                if task_type == "classification":
                    self.history['val_acc'].append(val_correct / val_total)
            
            # Progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss/len(train_loader):.4f}")

class TensorFlowNetwork:
    """TensorFlow Neural Network"""
    
    def __init__(self, config: NetworkConfig):
        self.config = config
        self.model = self._build_model()
        self.history = None
    
    def _build_model(self) -> tf.keras.Model:
        """Model oluştur"""
        model = tf.keras.Sequential()
        
        # Input layer
        model.add(tf.keras.layers.Dense(
            self.config.hidden_sizes[0], 
            input_shape=(self.config.input_size,),
            activation=self.config.activation
        ))
        
        # Hidden layers
        for hidden_size in self.config.hidden_sizes[1:]:
            model.add(tf.keras.layers.Dense(hidden_size, activation=self.config.activation))
            
            if self.config.dropout_rate > 0:
                model.add(tf.keras.layers.Dropout(self.config.dropout_rate))
        
        # Output layer
        if self.config.output_size == 1:
            model.add(tf.keras.layers.Dense(1))  # Regression
        else:
            model.add(tf.keras.layers.Dense(self.config.output_size, activation='softmax'))
        
        return model
    
    def compile_model(self, task_type: str = "classification"):
        """Model compile"""
        if task_type == "classification":
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
        else:
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
                loss='mse',
                metrics=['mae']
            )
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray = None, y_val: np.ndarray = None):
        """Model eğitimi"""
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_data=validation_data,
            verbose=0
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prediction"""
        return self.model.predict(X)

class DeepLearningVisualizer:
    """Deep Learning görselleştirme araçları"""
    
    @staticmethod
    def plot_training_history(history: Dict, title: str = "Training History"):
        """Training history görselleştirme"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)
        
        # Loss
        axes[0, 0].plot(history.get('loss', []), label='Loss', color='blue')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy (if available)
        if 'accuracy' in history:
            axes[0, 1].plot(history['accuracy'], label='Accuracy', color='green')
            axes[0, 1].set_title('Training Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Validation metrics (if available)
        if 'val_loss' in history:
            axes[1, 0].plot(history['loss'], label='Train Loss', color='blue')
            axes[1, 0].plot(history['val_loss'], label='Val Loss', color='red')
            axes[1, 0].set_title('Train vs Validation Loss')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        if 'val_acc' in history:
            axes[1, 1].plot(history['train_acc'], label='Train Acc', color='green')
            axes[1, 1].plot(history['val_acc'], label='Val Acc', color='orange')
            axes[1, 1].set_title('Train vs Validation Accuracy')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_decision_boundary(model, X: np.ndarray, y: np.ndarray, 
                             title: str = "Decision Boundary"):
        """Decision boundary görselleştirme (2D için)"""
        if X.shape[1] != 2:
            print("Decision boundary sadece 2D data için görselleştirilebilir.")
            return
        
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))
        
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        
        # Model prediction
        if hasattr(model, 'predict'):
            Z = model.predict(grid_points)
        else:
            Z = model(torch.FloatTensor(grid_points)).detach().numpy()
            Z = np.argmax(Z, axis=1)
        
        Z = Z.reshape(xx.shape)
        
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
        
        # Data points
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
        plt.colorbar(scatter)
        plt.title(title)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()
    
    @staticmethod
    def plot_weight_distributions(model, title: str = "Weight Distributions"):
        """Ağırlık dağılımları"""
        if hasattr(model, 'weights'):  # Scratch implementation
            weights = model.weights
        elif hasattr(model, 'parameters'):  # PyTorch
            weights = [p.data.numpy() for p in model.parameters() if len(p.shape) == 2]
        else:
            print("Weight visualization desteklenmiyor.")
            return
        
        num_layers = len(weights)
        fig, axes = plt.subplots(1, num_layers, figsize=(5*num_layers, 4))
        
        if num_layers == 1:
            axes = [axes]
        
        for i, w in enumerate(weights):
            axes[i].hist(w.flatten(), bins=50, alpha=0.7)
            axes[i].set_title(f'Layer {i+1} Weights')
            axes[i].set_xlabel('Weight Value')
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

class InteractiveNeuralNetworkPlayground:
    """Interactive neural network playground"""
    
    def __init__(self):
        self.datasets = {}
        self.models = {}
        self.current_dataset = None
        self.current_model = None
    
    def generate_dataset(self, dataset_type: str, n_samples: int = 1000, **kwargs):
        """Dataset oluştur"""
        if dataset_type == "classification":
            X, y = make_classification(
                n_samples=n_samples,
                n_features=kwargs.get('n_features', 2),
                n_redundant=0,
                n_informative=kwargs.get('n_features', 2),
                n_clusters_per_class=1,
                random_state=42
            )
        
        elif dataset_type == "regression":
            X, y = make_regression(
                n_samples=n_samples,
                n_features=kwargs.get('n_features', 2),
                noise=kwargs.get('noise', 0.1),
                random_state=42
            )
        
        elif dataset_type == "circles":
            from sklearn.datasets import make_circles
            X, y = make_circles(
                n_samples=n_samples,
                noise=kwargs.get('noise', 0.1),
                factor=kwargs.get('factor', 0.3),
                random_state=42
            )
        
        elif dataset_type == "moons":
            from sklearn.datasets import make_moons
            X, y = make_moons(
                n_samples=n_samples,
                noise=kwargs.get('noise', 0.1),
                random_state=42
            )
        
        elif dataset_type == "blobs":
            X, y = make_blobs(
                n_samples=n_samples,
                centers=kwargs.get('centers', 3),
                n_features=kwargs.get('n_features', 2),
                random_state=42
            )
        
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        # Normalize
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        self.datasets[dataset_type] = {
            'X': X,
            'y': y,
            'scaler': scaler,
            'type': 'classification' if dataset_type != 'regression' else 'regression'
        }
        
        self.current_dataset = dataset_type
        
        return X, y
    
    def create_model(self, framework: str, config: NetworkConfig, model_name: str):
        """Model oluştur"""
        if framework == "scratch":
            layers = [config.input_size] + config.hidden_sizes + [config.output_size]
            model = NeuralNetworkFromScratch(layers, config.activation, config.learning_rate)
        
        elif framework == "pytorch":
            model = PyTorchNetwork(config)
        
        elif framework == "tensorflow":
            model = TensorFlowNetwork(config)
            model.compile_model(self.datasets[self.current_dataset]['type'])
        
        else:
            raise ValueError(f"Unknown framework: {framework}")
        
        self.models[model_name] = {
            'model': model,
            'framework': framework,
            'config': config
        }
        
        self.current_model = model_name
        
        return model
    
    def train_current_model(self, test_size: float = 0.2):
        """Mevcut modeli eğit"""
        if not self.current_dataset or not self.current_model:
            raise ValueError("Dataset ve model seçilmeli")
        
        # Data hazırla
        dataset = self.datasets[self.current_dataset]
        X, y = dataset['X'], dataset['y']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Model ve framework al
        model_info = self.models[self.current_model]
        model = model_info['model']
        framework = model_info['framework']
        
        print(f"🚀 Training {framework} model on {self.current_dataset} dataset...")
        
        start_time = time.time()
        
        if framework == "scratch":
            model.train(X_train, y_train, epochs=model_info['config'].epochs)
        
        elif framework == "pytorch":
            # PyTorch DataLoader
            train_dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(X_train), 
                torch.LongTensor(y_train) if dataset['type'] == 'classification' else torch.FloatTensor(y_train)
            )
            train_loader = DataLoader(train_dataset, batch_size=model_info['config'].batch_size, shuffle=True)
            
            val_dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(X_test),
                torch.LongTensor(y_test) if dataset['type'] == 'classification' else torch.FloatTensor(y_test)
            )
            val_loader = DataLoader(val_dataset, batch_size=model_info['config'].batch_size)
            
            model.train_model(train_loader, val_loader, dataset['type'])
        
        elif framework == "tensorflow":
            model.train_model(X_train, y_train, X_test, y_test)
        
        training_time = time.time() - start_time
        
        # Evaluation
        if framework == "scratch":
            predictions = model.predict(X_test)
        elif framework == "pytorch":
            model.eval()
            with torch.no_grad():
                predictions = model(torch.FloatTensor(X_test))
                if dataset['type'] == 'classification':
                    predictions = torch.argmax(predictions, dim=1).numpy()
                else:
                    predictions = predictions.numpy().flatten()
        else:  # tensorflow
            predictions = model.predict(X_test)
            if dataset['type'] == 'classification':
                predictions = np.argmax(predictions, axis=1)
            else:
                predictions = predictions.flatten()
        
        # Metrics
        if dataset['type'] == 'classification':
            accuracy = accuracy_score(y_test, predictions)
            print(f"✅ Test Accuracy: {accuracy:.4f}")
            print(f"⏱️  Training Time: {training_time:.2f}s")
        else:
            mse = mean_squared_error(y_test, predictions)
            print(f"✅ Test MSE: {mse:.4f}")
            print(f"⏱️  Training Time: {training_time:.2f}s")
        
        return model
    
    def visualize_current_model(self):
        """Mevcut modeli görselleştir"""
        if not self.current_model:
            print("Model seçilmedi")
            return
        
        model_info = self.models[self.current_model]
        model = model_info['model']
        
        # Training history
        if hasattr(model, 'history'):
            if model_info['framework'] == "tensorflow":
                history = model.history.history
            else:
                history = model.history
            
            DeepLearningVisualizer.plot_training_history(history, f"{self.current_model} Training")
        
        # Decision boundary (2D data için)
        if self.current_dataset and self.datasets[self.current_dataset]['X'].shape[1] == 2:
            X = self.datasets[self.current_dataset]['X']
            y = self.datasets[self.current_dataset]['y']
            
            DeepLearningVisualizer.plot_decision_boundary(
                model, X, y, f"{self.current_model} Decision Boundary"
            )
        
        # Weight distributions
        DeepLearningVisualizer.plot_weight_distributions(model, f"{self.current_model} Weights")
    
    def compare_models(self, model_names: List[str]):
        """Modelleri karşılaştır"""
        print("🔍 Model Comparison")
        print("=" * 50)
        
        for name in model_names:
            if name in self.models:
                model_info = self.models[name]
                print(f"📊 {name} ({model_info['framework']}):")
                
                # Config bilgisi
                config = model_info['config']
                print(f"   Architecture: {config.input_size} -> {config.hidden_sizes} -> {config.output_size}")
                print(f"   Activation: {config.activation}")
                print(f"   Learning Rate: {config.learning_rate}")
                print(f"   Epochs: {config.epochs}")
                print("-" * 30)

# Demo fonksiyonu
def demo_deep_learning_playground():
    """Deep Learning Playground Demo"""
    print("🧠 Deep Learning Playground Demo")
    print("=" * 50)
    
    # Playground oluştur
    playground = InteractiveNeuralNetworkPlayground()
    
    # Farklı datasetler oluştur
    datasets = [
        ("classification", {"n_features": 2, "n_samples": 800}),
        ("circles", {"noise": 0.1, "n_samples": 600}),
        ("moons", {"noise": 0.15, "n_samples": 600})
    ]
    
    print("📊 Datasets oluşturuluyor...")
    for dataset_type, kwargs in datasets:
        X, y = playground.generate_dataset(dataset_type, **kwargs)
        print(f"✅ {dataset_type}: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Model konfigürasyonları
    configs = {
        "simple": NetworkConfig(
            input_size=2,
            hidden_sizes=[10],
            output_size=2,
            activation="relu",
            learning_rate=0.01,
            epochs=50
        ),
        "deep": NetworkConfig(
            input_size=2,
            hidden_sizes=[20, 15, 10],
            output_size=2,
            activation="relu",
            learning_rate=0.001,
            epochs=100
        )
    }
    
    # Farklı frameworkler test et
    frameworks = ["scratch", "pytorch"]  # tensorflow excluded for demo
    
    print("\n🤖 Models oluşturuluyor...")
    for framework in frameworks:
        for config_name, config in configs.items():
            model_name = f"{framework}_{config_name}"
            try:
                model = playground.create_model(framework, config, model_name)
                print(f"✅ {model_name} created")
            except Exception as e:
                print(f"❌ {model_name} failed: {e}")
    
    # Test eğitimi
    print("\n🚀 Test training başlatılıyor...")
    
    # Circles dataset seç
    playground.current_dataset = "circles"
    
    # Simple model ile test
    playground.current_model = "scratch_simple"
    try:
        model = playground.train_current_model()
        print("✅ Training completed!")
        
        # Görselleştirme
        print("📈 Visualizing results...")
        playground.visualize_current_model()
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
    
    # Model karşılaştırması
    print("\n📊 Model comparison:")
    available_models = list(playground.models.keys())
    playground.compare_models(available_models)
    
    print("\n✨ Demo completed!")
    print("\n💡 Bu playground ile:")
    print("   - Farklı dataset türleri oluşturabilirsiniz")
    print("   - Multiple framework'leri karşılaştırabilirsiniz")  
    print("   - Neural network mimarilerini deneyebilirsiniz")
    print("   - Training sürecini görselleştirebilirsiniz")

if __name__ == "__main__":
    demo_deep_learning_playground()