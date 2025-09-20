"""
AI Lab Kurulum Script'i
Tüm projeleri ve bağımlılıkları kurar
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, description):
    """Komut çalıştır ve sonucu kontrol et"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} başarılı!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} başarısız!")
        print(f"Hata: {e.stderr}")
        return False

def check_python_version():
    """Python versiyonunu kontrol et"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ gerekli!")
        print(f"Mevcut versiyon: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python versiyonu uygun: {version.major}.{version.minor}.{version.micro}")
    return True

def create_directory_structure():
    """Proje klasör yapısını oluştur"""
    directories = [
        "nlp-projects/sentiment-analysis",
        "nlp-projects/text-classification", 
        "nlp-projects/named-entity-recognition",
        "nlp-projects/text-summarization",
        "nlp-projects/bert-fine-tuning",
        
        "llm-projects/fine-tuning-guide",
        "llm-projects/prompt-engineering",
        "llm-projects/rag-implementation", 
        "llm-projects/custom-chatbot",
        "llm-projects/local-llm-setup",
        
        "mcp-projects/basic-mcp-server",
        "mcp-projects/database-connector",
        "mcp-projects/api-integration",
        "mcp-projects/claude-integration",
        
        "computer-vision/image-classification",
        "computer-vision/object-detection",
        "computer-vision/face-recognition",
        "computer-vision/style-transfer",
        
        "ml-fundamentals/regression-models",
        "ml-fundamentals/clustering",
        "ml-fundamentals/neural-networks",
        "ml-fundamentals/feature-engineering",
        
        "utils/data-preprocessing",
        "utils/model-evaluation", 
        "utils/deployment-tools",
        
        "data/datasets",
        "data/models",
        "data/outputs",
        
        "docs/tutorials",
        "docs/api-reference",
        "docs/examples",
        
        "tests/unit",
        "tests/integration",
        "tests/data"
    ]
    
    print("📁 Klasör yapısı oluşturuluyor...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Her klasöre README.md ekle
        readme_path = Path(directory) / "README.md"
        if not readme_path.exists():
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(f"# {directory.replace('/', ' ').title()}\n\n")
                f.write("Bu bölüm henüz geliştirilme aşamasında.\n\n")
                f.write("## TODO\n")
                f.write("- [ ] Proje implementasyonu\n")
                f.write("- [ ] Dokümantasyon\n")
                f.write("- [ ] Test yazma\n")
    
    print("✅ Klasör yapısı oluşturuldu!")

def install_requirements():
    """Gereksinimleri yükle"""
    print("📦 Python paketleri yükleniyor...")
    
    # Base requirements
    base_packages = [
        "torch torchvision",
        "transformers sentence-transformers",
        "scikit-learn numpy pandas",
        "matplotlib seaborn plotly",
        "fastapi uvicorn streamlit",
        "jupyter notebook",
        "pytest black flake8"
    ]
    
    for package_group in base_packages:
        success = run_command(f"pip install {package_group}", f"Installing {package_group}")
        if not success:
            print(f"⚠️  {package_group} yüklenemedi, devam ediliyor...")

def setup_git_hooks():
    """Git hooks kurulumu"""
    if not Path(".git").exists():
        run_command("git init", "Git repository initialize")
    
    # Pre-commit hook
    hook_content = """#!/bin/bash
# Pre-commit hook for code quality
echo "🔍 Code quality check..."

# Run black formatter
black --check . || {
    echo "❌ Code formatting issues found. Run: black ."
    exit 1
}

# Run flake8 linter  
flake8 . || {
    echo "❌ Linting issues found. Fix them before commit."
    exit 1
}

echo "✅ Code quality check passed!"
"""
    
    hooks_dir = Path(".git/hooks")
    if hooks_dir.exists():
        hook_path = hooks_dir / "pre-commit"
        with open(hook_path, 'w') as f:
            f.write(hook_content)
        os.chmod(hook_path, 0o755)
        print("✅ Git hooks kuruldu!")

def create_config_files():
    """Konfigürasyon dosyaları oluştur"""
    
    # .env template
    env_template = """# AI Lab Environment Variables
# Database
DATABASE_URL=sqlite:///ai_lab.db

# API Keys (fill with your keys)
OPENAI_API_KEY=your_openai_key_here
HUGGINGFACE_API_KEY=your_hf_key_here
WANDB_API_KEY=your_wandb_key_here

# Model Paths
MODELS_DIR=./data/models
DATASETS_DIR=./data/datasets

# Server Settings
HOST=localhost
PORT=8000
DEBUG=True
"""
    
    with open('.env.template', 'w') as f:
        f.write(env_template)
    
    # config.yaml
    config_yaml = """# AI Lab Configuration
project:
  name: "AI Lab"
  version: "1.0.0"
  description: "Modern AI Projects Collection"

models:
  embedding:
    default: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    cache_dir: "./data/models"
  
  llm:
    default: "microsoft/DialoGPT-medium"
    max_length: 512

data:
  batch_size: 32
  test_split: 0.2
  val_split: 0.1

training:
  epochs: 10
  learning_rate: 0.001
  device: "auto"  # auto, cuda, cpu

logging:
  level: "INFO"
  format: "%(asctime)s - %(levelname)s - %(message)s"
"""
    
    with open('config.yaml', 'w') as f:
        f.write(config_yaml)
    
    # .gitignore
    gitignore_content = """# AI Lab .gitignore

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# Data files
*.csv
*.json
*.parquet
*.h5
*.hdf5
*.pkl
*.pickle

# Model files
*.pth
*.pt
*.bin
*.safetensors
*.onnx

# Logs
*.log
logs/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Environment
.env
.env.local

# Jupyter
.ipynb_checkpoints/

# Cache
.cache/
.pytest_cache/

# Weights & Biases
wandb/

# Data directories (keep structure, ignore content)
data/datasets/*
!data/datasets/.gitkeep
data/models/*
!data/models/.gitkeep
data/outputs/*
!data/outputs/.gitkeep
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    
    print("✅ Konfigürasyon dosyaları oluşturuldu!")

def create_demo_scripts():
    """Demo scriptleri oluştur"""
    
    # Quick start script
    quickstart = """#!/usr/bin/env python3
\"\"\"
AI Lab Quick Start Script
Hızlı demo ve test için
\"\"\"

import sys
import os
sys.path.append(os.getcwd())

def main():
    print("🚀 AI Lab Quick Start")
    print("=" * 50)
    
    # Import check
    try:
        import torch
        import transformers
        import sklearn
        print("✅ Core libraries imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return
    
    # Available demos
    demos = {
        "1": ("Sentiment Analysis", "nlp-projects/sentiment-analysis/main.py"),
        "2": ("Image Classification", "computer-vision/image-classification/image_classifier.py"),
        "3": ("RAG System", "llm-projects/rag-implementation/rag_system.py"),
        "4": ("MCP Server", "mcp-projects/basic-mcp-server/mcp_server.py")
    }
    
    print("\\nAvailable Demos:")
    for key, (name, path) in demos.items():
        status = "✅" if os.path.exists(path) else "🚧"
        print(f"  {key}. {status} {name}")
    
    print("\\n💡 To run a demo:")
    print("   python quick_start.py")
    print("\\n📚 For detailed tutorials, check docs/")

if __name__ == "__main__":
    main()
"""
    
    with open('quick_start.py', 'w') as f:
        f.write(quickstart)
    
    # Make executable
    os.chmod('quick_start.py', 0o755)
    
    print("✅ Demo scriptleri oluşturuldu!")

def create_documentation():
    """Temel dokümantasyonu oluştur"""
    
    # CONTRIBUTING.md
    contributing = """# Contributing to AI Lab

AI Lab projesine katkıda bulunduğunuz için teşekkürler! 🎉

## Katkı Türleri

- 🐛 **Bug Reports**: Hata raporları
- 💡 **Feature Requests**: Yeni özellik önerileri  
- 📝 **Documentation**: Dokümantasyon iyileştirmeleri
- 🔧 **Code**: Kod katkıları
- 🌍 **Translation**: Çeviri katkıları

## Development Setup

1. Repository'yi fork edin
2. Local development kurulumu:
   ```bash
   git clone https://github.com/yourusername/ai-lab.git
   cd ai-lab
   python setup.py
   ```

## Code Style

- **Python**: PEP 8 standardı
- **Formatter**: Black
- **Linter**: Flake8
- **Type Hints**: Mümkün olduğunca kullanın

## Commit Messages

```
type(scope): description

feat(nlp): add new sentiment analysis model
fix(rag): resolve embedding dimension mismatch
docs(readme): update installation instructions
```

## Pull Request Process

1. Feature branch oluşturun: `git checkout -b feature/amazing-feature`
2. Değişikliklerinizi commit edin: `git commit -m 'Add amazing feature'`
3. Branch'i push edin: `git push origin feature/amazing-feature`
4. Pull Request açın

## Testing

```bash
# Unit tests
pytest tests/unit/

# Integration tests  
pytest tests/integration/

# Coverage
pytest --cov=src/
```

## Documentation

- Yeni özellikler için dokümantasyon ekleyin
- Code comments yazın
- README'leri güncel tutun

## Community

- Respectful ve inclusive davranın
- Constructive feedback verin
- Yardım edin ve yardım isteyin

Teşekkürler! 🙏
"""
    
    with open('CONTRIBUTING.md', 'w', encoding='utf-8') as f:
        f.write(contributing)
    
    # LICENSE
    license_content = """MIT License

Copyright (c) 2024 AI Lab Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
    
    with open('LICENSE', 'w') as f:
        f.write(license_content)
    
    print("✅ Dokümantasyon dosyaları oluşturuldu!")

def main():
    """Ana kurulum fonksiyonu"""
    print("🧠 AI Lab Setup Script")
    print("=" * 50)
    
    # System checks
    if not check_python_version():
        sys.exit(1)
    
    print(f"💻 İşletim Sistemi: {platform.system()} {platform.release()}")
    print(f"🐍 Python: {sys.executable}")
    
    # Setup steps
    steps = [
        ("Klasör yapısı oluşturma", create_directory_structure),
        ("Paket yükleme", install_requirements),
        ("Git hooks kurulumu", setup_git_hooks),
        ("Konfigürasyon dosyaları", create_config_files),
        ("Demo scriptleri", create_demo_scripts),
        ("Dokümantasyon", create_documentation)
    ]
    
    for step_name, step_func in steps:
        try:
            step_func()
        except Exception as e:
            print(f"❌ {step_name} sırasında hata: {e}")
            continue
    
    print("\n🎉 AI Lab kurulumu tamamlandı!")
    print("\n📋 Sonraki adımlar:")
    print("   1. .env.template dosyasını .env olarak kopyalayın ve API key'leri ekleyin")
    print("   2. python quick_start.py ile demo'ları deneyin")
    print("   3. docs/ klasöründeki tutorialleri inceleyin")
    print("   4. İlk projenizi seçin ve geliştirmeye başlayın!")
    print("\n🤝 Katkıda bulunmak için CONTRIBUTING.md dosyasını okuyun")
    print("⭐ Projeyi beğendiyseniz GitHub'da yıldız vermeyi unutmayın!")

if __name__ == "__main__":
    main()
