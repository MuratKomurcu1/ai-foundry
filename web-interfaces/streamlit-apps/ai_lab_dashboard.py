import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import cv2
from PIL import Image
import io
import base64
import requests
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

# AI Libraries (conditional imports for demo)
try:
    import torch
    import torch.nn as nn
    from transformers import pipeline
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Configure Streamlit page
st.set_page_config(
    page_title="AI Lab Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class AILabDashboard:
    """Main dashboard class"""
    
    def __init__(self):
        self.models = self.load_models()
        self.setup_session_state()
    
    def setup_session_state(self):
        """Initialize session state variables"""
        if 'page' not in st.session_state:
            st.session_state.page = 'Home'
        
        if 'model_cache' not in st.session_state:
            st.session_state.model_cache = {}
        
        if 'processing_history' not in st.session_state:
            st.session_state.processing_history = []
    
    def load_models(self):
        """Load available models"""
        models = {
            'nlp': {
                'sentiment_analysis': {
                    'name': 'Sentiment Analysis',
                    'description': 'Analyze text sentiment (positive/negative/neutral)',
                    'status': 'available' if TORCH_AVAILABLE else 'unavailable'
                },
                'text_classification': {
                    'name': 'Text Classification',
                    'description': 'Classify texts into predefined categories',
                    'status': 'available' if TORCH_AVAILABLE else 'unavailable'
                },
                'text_summarization': {
                    'name': 'Text Summarization',
                    'description': 'Generate concise summaries of long texts',
                    'status': 'available' if TORCH_AVAILABLE else 'unavailable'
                }
            },
            'computer_vision': {
                'image_classification': {
                    'name': 'Image Classification',
                    'description': 'Classify images into categories',
                    'status': 'available'
                },
                'object_detection': {
                    'name': 'Object Detection',
                    'description': 'Detect and locate objects in images',
                    'status': 'available'
                },
                'face_recognition': {
                    'name': 'Face Recognition',
                    'description': 'Detect and recognize faces',
                    'status': 'available'
                }
            },
            'llm': {
                'chatbot': {
                    'name': 'AI Chatbot',
                    'description': 'Intelligent conversational AI',
                    'status': 'available' if TORCH_AVAILABLE else 'unavailable'
                },
                'rag_system': {
                    'name': 'RAG System',
                    'description': 'Retrieval-Augmented Generation',
                    'status': 'available' if SENTENCE_TRANSFORMERS_AVAILABLE else 'unavailable'
                }
            }
        }
        
        return models

    def render_sidebar(self):
        """Render sidebar navigation"""
        st.sidebar.markdown("# 🧠 AI Lab")
        st.sidebar.markdown("*Modern AI/ML Toolkit*")
        
        # Navigation
        pages = [
            "🏠 Home",
            "📝 NLP Models", 
            "👁️ Computer Vision",
            "🤖 LLM & Chatbots",
            "📊 Data Explorer",
            "🔧 Model Training",
            "📈 Analytics",
            "⚙️ Settings"
        ]
        
        selected_page = st.sidebar.selectbox("Navigation", pages)
        st.session_state.page = selected_page.split(" ", 1)[1]  # Remove emoji
        
        # Model status
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🔧 System Status")
        
        status_indicators = {
            "PyTorch": "🟢" if TORCH_AVAILABLE else "🔴",
            "Transformers": "🟢" if TORCH_AVAILABLE else "🔴",
            "SentenceTransformers": "🟢" if SENTENCE_TRANSFORMERS_AVAILABLE else "🔴",
            "OpenCV": "🟢",
            "Streamlit": "🟢"
        }
        
        for lib, status in status_indicators.items():
            st.sidebar.write(f"{status} {lib}")
        
        # Quick stats
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Quick Stats")
        
        total_models = sum(len(category) for category in self.models.values())
        available_models = sum(
            1 for category in self.models.values() 
            for model in category.values() 
            if model['status'] == 'available'
        )
        
        st.sidebar.metric("Total Models", total_models)
        st.sidebar.metric("Available Models", available_models)
        st.sidebar.metric("Processing History", len(st.session_state.processing_history))

    def render_home_page(self):
        """Render home page"""
        st.markdown('<h1 class="main-header">🧠 AI Lab Dashboard</h1>', unsafe_allow_html=True)
        
        st.markdown("""
        Welcome to the AI Lab Dashboard! This interactive platform provides access to a comprehensive 
        suite of AI/ML models and tools. Explore different categories using the sidebar navigation.
        """)
        
        # Feature overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 📝 Natural Language Processing
            - **Sentiment Analysis**: Detect emotions in text
            - **Text Classification**: Categorize documents
            - **Summarization**: Generate concise summaries
            - **Turkish Language Support**: Native Turkish models
            """)
        
        with col2:
            st.markdown("""
            ### 👁️ Computer Vision
            - **Image Classification**: Identify objects and scenes
            - **Object Detection**: Locate objects in images
            - **Face Recognition**: Detect and identify faces
            - **Real-time Processing**: Webcam integration
            """)
        
        with col3:
            st.markdown("""
            ### 🤖 LLM & Advanced AI
            - **Intelligent Chatbot**: Conversational AI
            - **RAG System**: Knowledge-based Q&A
            - **Function Calling**: Tool integration
            - **Memory System**: Persistent conversations
            """)
        
        # Recent activity
        st.markdown("---")
        st.markdown("### 📊 System Overview")
        
        # Create sample metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Models Available",
                value=f"{sum(1 for cat in self.models.values() for m in cat.values() if m['status'] == 'available')}",
                delta="2 new this week"
            )
        
        with col2:
            st.metric(
                label="Processing Speed",
                value="< 2s",
                delta="-0.3s faster"
            )
        
        with col3:
            st.metric(
                label="Accuracy",
                value="94.2%",
                delta="1.2%"
            )
        
        with col4:
            st.metric(
                label="Uptime",
                value="99.9%",
                delta="0.1%"
            )
        
        # Model categories visualization
        st.markdown("### 🎯 Model Categories")
        
        # Create pie chart of model categories
        categories = list(self.models.keys())
        counts = [len(self.models[cat]) for cat in categories]
        
        fig = px.pie(
            values=counts, 
            names=categories,
            title="Distribution of AI Models by Category",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def render_nlp_page(self):
        """Render NLP models page"""
        st.markdown("# 📝 Natural Language Processing")
        
        # Model selection
        nlp_models = list(self.models['nlp'].keys())
        selected_model = st.selectbox("Select NLP Model", nlp_models)
        
        model_info = self.models['nlp'][selected_model]
        
        st.markdown(f"## {model_info['name']}")
        st.markdown(f"*{model_info['description']}*")
        
        if model_info['status'] != 'available':
            st.error("⚠️ This model is currently unavailable. Please check system requirements.")
            return
        
        # Model-specific interfaces
        if selected_model == 'sentiment_analysis':
            self.render_sentiment_analysis()
        elif selected_model == 'text_classification':
            self.render_text_classification()
        elif selected_model == 'text_summarization':
            self.render_text_summarization()

    def render_sentiment_analysis(self):
        """Render sentiment analysis interface"""
        st.markdown("### 🎭 Sentiment Analysis")
        
        # Input options
        input_method = st.radio("Input Method", ["Text Input", "File Upload", "URL"])
        
        text_to_analyze = ""
        
        if input_method == "Text Input":
            text_to_analyze = st.text_area(
                "Enter text to analyze:",
                placeholder="Type your text here...",
                height=150
            )
        
        elif input_method == "File Upload":
            uploaded_file = st.file_uploader(
                "Upload text file",
                type=['txt', 'csv']
            )
            
            if uploaded_file:
                content = uploaded_file.read().decode('utf-8')
                text_to_analyze = content
                st.text_area("File content:", content, height=150, disabled=True)
        
        elif input_method == "URL":
            url = st.text_input("Enter URL:")
            if url and st.button("Fetch Content"):
                try:
                    # Simulated URL content fetching
                    text_to_analyze = "Sample content from URL"
                    st.success("Content fetched successfully!")
                except Exception as e:
                    st.error(f"Error fetching content: {e}")
        
        # Analysis options
        col1, col2 = st.columns(2)
        with col1:
            confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)
        
        with col2:
            language = st.selectbox("Language", ["Auto-detect", "English", "Turkish"])
        
        # Analyze button
        if st.button("🔍 Analyze Sentiment", type="primary") and text_to_analyze:
            with st.spinner("Analyzing sentiment..."):
                # Simulate sentiment analysis
                time.sleep(1)  # Simulate processing time
                
                # Mock results
                sentiment_result = {
                    'sentiment': np.random.choice(['Positive', 'Negative', 'Neutral']),
                    'confidence': np.random.uniform(0.7, 0.99),
                    'scores': {
                        'Positive': np.random.uniform(0.1, 0.9),
                        'Negative': np.random.uniform(0.1, 0.9),
                        'Neutral': np.random.uniform(0.1, 0.9)
                    }
                }
                
                # Normalize scores
                total = sum(sentiment_result['scores'].values())
                sentiment_result['scores'] = {
                    k: v/total for k, v in sentiment_result['scores'].items()
                }
                
                # Display results
                st.markdown("### 📊 Analysis Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    sentiment_color = {
                        'Positive': 'green',
                        'Negative': 'red', 
                        'Neutral': 'gray'
                    }[sentiment_result['sentiment']]
                    
                    st.markdown(f"""
                    <div class="metric-container">
                        <h3 style="color: {sentiment_color};">{sentiment_result['sentiment']}</h3>
                        <p>Primary Sentiment</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.metric(
                        "Confidence",
                        f"{sentiment_result['confidence']:.1%}",
                        delta=f"{(sentiment_result['confidence']-0.5)*100:+.1f}%"
                    )
                
                with col3:
                    st.metric(
                        "Text Length",
                        f"{len(text_to_analyze)} chars",
                        delta=f"{len(text_to_analyze.split())} words"
                    )
                
                # Probability distribution
                st.markdown("### 📈 Probability Distribution")
                
                fig = px.bar(
                    x=list(sentiment_result['scores'].keys()),
                    y=list(sentiment_result['scores'].values()),
                    title="Sentiment Probabilities",
                    color=list(sentiment_result['scores'].values()),
                    color_continuous_scale="RdYlGn"
                )
                
                fig.update_layout(
                    xaxis_title="Sentiment",
                    yaxis_title="Probability",
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Save to history
                st.session_state.processing_history.append({
                    'timestamp': datetime.now(),
                    'type': 'Sentiment Analysis',
                    'input_length': len(text_to_analyze),
                    'result': sentiment_result['sentiment'],
                    'confidence': sentiment_result['confidence']
                })

    def render_text_classification(self):
        """Render text classification interface"""
        st.markdown("### 🏷️ Text Classification")
        
        # Predefined categories
        default_categories = [
            "Technology", "Sports", "Politics", "Health", 
            "Entertainment", "Business", "Science", "Education"
        ]
        
        # Category selection
        use_custom_categories = st.checkbox("Use custom categories")
        
        if use_custom_categories:
            categories = st.text_input(
                "Enter categories (comma-separated):",
                "Tech, Sports, Politics"
            ).split(',')
            categories = [cat.strip() for cat in categories if cat.strip()]
        else:
            categories = default_categories
        
        st.write(f"**Categories:** {', '.join(categories)}")
        
        # Text input
        text_to_classify = st.text_area(
            "Text to classify:",
            placeholder="Enter text for classification...",
            height=100
        )
        
        # Classification settings
        col1, col2 = st.columns(2)
        with col1:
            multi_label = st.checkbox("Multi-label classification")
        with col2:
            top_k = st.slider("Show top K predictions", 1, min(5, len(categories)), 3)
        
        if st.button("🎯 Classify Text", type="primary") and text_to_classify:
            with st.spinner("Classifying text..."):
                time.sleep(1)
                
                # Mock classification results
                if multi_label:
                    # Multi-label: multiple categories can be true
                    results = {}
                    for category in categories:
                        results[category] = np.random.uniform(0.1, 0.9)
                    
                    # Filter by threshold
                    threshold = 0.5
                    predicted_categories = [
                        cat for cat, score in results.items() 
                        if score >= threshold
                    ]
                    
                    st.markdown("### 🏷️ Multi-label Results")
                    
                    if predicted_categories:
                        for category in predicted_categories:
                            st.markdown(f"✅ **{category}**: {results[category]:.1%}")
                    else:
                        st.warning("No categories above threshold")
                    
                    # All scores
                    st.markdown("### 📊 All Category Scores")
                    df = pd.DataFrame({
                        'Category': list(results.keys()),
                        'Score': list(results.values())
                    }).sort_values('Score', ascending=False)
                    
                    fig = px.bar(df, x='Category', y='Score', 
                               title="Classification Scores")
                    st.plotly_chart(fig, use_container_width=True)
                
                else:
                    # Single-label: one category prediction
                    scores = {cat: np.random.uniform(0.1, 0.9) for cat in categories}
                    # Normalize to probabilities
                    total = sum(scores.values())
                    probabilities = {cat: score/total for cat, score in scores.items()}
                    
                    # Sort by probability
                    sorted_results = sorted(
                        probabilities.items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )
                    
                    st.markdown("### 🎯 Classification Results")
                    
                    # Top prediction
                    top_category, top_prob = sorted_results[0]
                    st.success(f"**Predicted Category:** {top_category} ({top_prob:.1%})")
                    
                    # Top K results
                    st.markdown(f"### 📈 Top {top_k} Predictions")
                    
                    for i, (category, prob) in enumerate(sorted_results[:top_k]):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"{i+1}. **{category}**")
                        with col2:
                            st.write(f"{prob:.1%}")
                        
                        # Progress bar
                        st.progress(prob)

    def render_text_summarization(self):
        """Render text summarization interface"""
        st.markdown("### 📄 Text Summarization")
        
        # Input text
        text_to_summarize = st.text_area(
            "Text to summarize:",
            placeholder="Paste your long text here...",
            height=200
        )
        
        if text_to_summarize:
            # Text statistics
            word_count = len(text_to_summarize.split())
            char_count = len(text_to_summarize)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Words", word_count)
            with col2:
                st.metric("Characters", char_count)
            with col3:
                est_reading_time = max(1, word_count // 200)
                st.metric("Est. Reading Time", f"{est_reading_time} min")
        
        # Summarization options
        st.markdown("### ⚙️ Summarization Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            summary_method = st.selectbox(
                "Method",
                ["Extractive", "Abstractive", "Hybrid"]
            )
            
            summary_length = st.selectbox(
                "Summary Length",
                ["Short (1-2 sentences)", "Medium (3-5 sentences)", "Long (6+ sentences)"]
            )
        
        with col2:
            compression_ratio = st.slider(
                "Compression Ratio",
                0.1, 0.8, 0.3,
                help="Ratio of summary length to original length"
            )
            
            preserve_keywords = st.checkbox(
                "Preserve Keywords",
                value=True,
                help="Ensure important keywords are included"
            )
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            focus_sentences = st.number_input(
                "Focus on first N sentences",
                min_value=0, max_value=50, value=0,
                help="0 means use entire text"
            )
            
            avoid_redundancy = st.checkbox(
                "Avoid redundancy",
                value=True
            )
            
            maintain_order = st.checkbox(
                "Maintain sentence order",
                value=True
            )
        
        # Summarize button
        if st.button("📝 Generate Summary", type="primary") and text_to_summarize:
            with st.spinner("Generating summary..."):
                time.sleep(2)  # Simulate processing
                
                # Mock summary generation
                sentences = text_to_summarize.split('.')
                num_sentences = max(1, int(len(sentences) * compression_ratio))
                
                # Select sentences (mock extractive summarization)
                selected_sentences = sentences[:num_sentences]
                summary = '. '.join(selected_sentences).strip()
                
                if summary and not summary.endswith('.'):
                    summary += '.'
                
                # Display results
                st.markdown("### 📑 Summary Results")
                
                # Summary metrics
                summary_words = len(summary.split())
                original_words = len(text_to_summarize.split())
                actual_compression = summary_words / original_words if original_words > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Summary Words", summary_words)
                with col2:
                    st.metric("Compression", f"{actual_compression:.1%}")
                with col3:
                    st.metric("Method", summary_method)
                
                # Summary text
                st.markdown("### 📝 Generated Summary")
                st.markdown(f"""
                <div class="success-box">
                    {summary}
                </div>
                """, unsafe_allow_html=True)
                
                # Quality metrics (mock)
                st.markdown("### 📊 Quality Metrics")
                
                quality_metrics = {
                    'Coherence': np.random.uniform(0.7, 0.95),
                    'Relevance': np.random.uniform(0.8, 0.98),
                    'Fluency': np.random.uniform(0.75, 0.92),
                    'Coverage': np.random.uniform(0.6, 0.85)
                }
                
                for metric, score in quality_metrics.items():
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        st.write(f"**{metric}:**")
                    with col2:
                        st.progress(score)
                        st.write(f"{score:.1%}")

    def render_computer_vision_page(self):
        """Render computer vision page"""
        st.markdown("# 👁️ Computer Vision")
        
        cv_models = list(self.models['computer_vision'].keys())
        selected_model = st.selectbox("Select CV Model", cv_models)
        
        model_info = self.models['computer_vision'][selected_model]
        
        st.markdown(f"## {model_info['name']}")
        st.markdown(f"*{model_info['description']}*")
        
        if selected_model == 'image_classification':
            self.render_image_classification()
        elif selected_model == 'object_detection':
            self.render_object_detection()
        elif selected_model == 'face_recognition':
            self.render_face_recognition()

    def render_image_classification(self):
        """Render image classification interface"""
        st.markdown("### 🖼️ Image Classification")
        
        # Image upload
        uploaded_image = st.file_uploader(
            "Upload an image",
            type=['png', 'jpg', 'jpeg'],
            help="Supported formats: PNG, JPG, JPEG"
        )
        
        # Camera input
        camera_image = st.camera_input("Or take a photo")
        
        # Use uploaded or camera image
        image_source = None
        if uploaded_image:
            image_source = uploaded_image
        elif camera_image:
            image_source = camera_image
        
        if image_source:
            # Display image
            image = Image.open(image_source)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.image(image, caption="Input Image", use_container_width=True)
            
            with col2:
                # Image info
                st.markdown("**Image Info:**")
                st.write(f"Size: {image.size}")
                st.write(f"Mode: {image.mode}")
                
                # Preprocessing options
                st.markdown("**Preprocessing:**")
                resize_image = st.checkbox("Resize to 224x224", value=True)
                normalize = st.checkbox("Normalize", value=True)
                augment = st.checkbox("Apply augmentation", value=False)
            
            # Classification button
            if st.button("🔍 Classify Image", type="primary"):
                with st.spinner("Classifying image..."):
                    time.sleep(1)
                    
                    # Mock classification results
                    mock_results = [
                        {"class": "Cat", "confidence": 0.92},
                        {"class": "Dog", "confidence": 0.05},
                        {"class": "Bird", "confidence": 0.02},
                        {"class": "Car", "confidence": 0.01}
                    ]
                    
                    st.markdown("### 🎯 Classification Results")
                    
                    # Top prediction
                    top_result = mock_results[0]
                    st.success(f"**Predicted Class:** {top_result['class']} ({top_result['confidence']:.1%})")
                    
                    # All predictions
                    st.markdown("### 📊 All Predictions")
                    
                    df = pd.DataFrame(mock_results)
                    
                    fig = px.bar(
                        df, x='class', y='confidence',
                        title="Classification Confidence",
                        color='confidence',
                        color_continuous_scale="viridis"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed results table
                    st.dataframe(df, use_container_width=True)

    def render_object_detection(self):
        """Render object detection interface"""
        st.markdown("### 🎯 Object Detection")
        
        st.info("Object detection identifies and locates multiple objects within an image.")
        
        # Model selection
        detection_models = ["YOLO v8", "R-CNN", "SSD MobileNet"]
        selected_detection_model = st.selectbox("Detection Model", detection_models)
        
        # Confidence threshold
        confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5)
        
        # Image upload
        uploaded_image = st.file_uploader(
            "Upload image for object detection",
            type=['png', 'jpg', 'jpeg']
        )
        
        if uploaded_image:
            image = Image.open(uploaded_image)
            st.image(image, caption="Input Image", use_container_width=True)
            
            if st.button("🔍 Detect Objects", type="primary"):
                with st.spinner("Detecting objects..."):
                    time.sleep(2)
                    
                    # Mock detection results
                    detections = [
                        {"object": "person", "confidence": 0.95, "bbox": [100, 50, 200, 300]},
                        {"object": "car", "confidence": 0.87, "bbox": [250, 150, 400, 250]},
                        {"object": "dog", "confidence": 0.78, "bbox": [50, 200, 150, 280]}
                    ]
                    
                    # Filter by confidence
                    filtered_detections = [
                        d for d in detections 
                        if d["confidence"] >= confidence_threshold
                    ]
                    
                    st.markdown(f"### 🎯 Detected Objects ({len(filtered_detections)})")
                    
                    if filtered_detections:
                        # Results table
                        df = pd.DataFrame(filtered_detections)
                        st.dataframe(df[['object', 'confidence']], use_container_width=True)
                        
                        # Visualization
                        st.markdown("### 📊 Detection Confidence")
                        
                        fig = px.bar(
                            df, x='object', y='confidence',
                            title="Object Detection Confidence",
                            color='confidence'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Object count summary
                        object_counts = df['object'].value_counts()
                        
                        st.markdown("### 📈 Object Count Summary")
                        st.bar_chart(object_counts)
                    
                    else:
                        st.warning("No objects detected above the confidence threshold.")

    def render_face_recognition(self):
        """Render face recognition interface"""
        st.markdown("### 👤 Face Recognition")
        
        tab1, tab2 = st.tabs(["Face Detection", "Face Recognition"])
        
        with tab1:
            st.markdown("#### Face Detection & Analysis")
            
            uploaded_image = st.file_uploader(
                "Upload image with faces",
                type=['png', 'jpg', 'jpeg'],
                key="face_detection"
            )
            
            if uploaded_image:
                image = Image.open(uploaded_image)
                st.image(image, caption="Input Image", use_container_width=True)
                
                # Detection options
                col1, col2 = st.columns(2)
                with col1:
                    detect_emotions = st.checkbox("Detect Emotions", value=True)
                    detect_age_gender = st.checkbox("Estimate Age & Gender", value=True)
                
                with col2:
                    draw_landmarks = st.checkbox("Draw Facial Landmarks", value=False)
                    min_face_size = st.slider("Minimum Face Size", 20, 200, 50)
                
                if st.button("👁️ Analyze Faces", type="primary"):
                    with st.spinner("Analyzing faces..."):
                        time.sleep(2)
                        
                        # Mock face analysis results
                        face_results = [
                            {
                                "face_id": 1,
                                "confidence": 0.98,
                                "bbox": [120, 80, 200, 160],
                                "emotion": "Happy",
                                "emotion_confidence": 0.92,
                                "age": "25-32",
                                "gender": "Female",
                                "gender_confidence": 0.89
                            },
                            {
                                "face_id": 2,
                                "confidence": 0.95,
                                "bbox": [300, 100, 380, 180],
                                "emotion": "Neutral",
                                "emotion_confidence": 0.78,
                                "age": "35-43",
                                "gender": "Male", 
                                "gender_confidence": 0.94
                            }
                        ]
                        
                        st.markdown(f"### 👥 Detected {len(face_results)} faces")
                        
                        for i, face in enumerate(face_results):
                            with st.expander(f"Face {face['face_id']} - {face['emotion']}"):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.write(f"**Detection Confidence:** {face['confidence']:.1%}")
                                    if detect_emotions:
                                        st.write(f"**Emotion:** {face['emotion']} ({face['emotion_confidence']:.1%})")
                                
                                with col2:
                                    if detect_age_gender:
                                        st.write(f"**Age:** {face['age']}")
                                        st.write(f"**Gender:** {face['gender']} ({face['gender_confidence']:.1%})")
                        
                        # Emotion distribution
                        if detect_emotions:
                            emotions = [face['emotion'] for face in face_results]
                            emotion_counts = pd.Series(emotions).value_counts()
                            
                            st.markdown("### 😊 Emotion Distribution")
                            fig = px.pie(
                                values=emotion_counts.values,
                                names=emotion_counts.index,
                                title="Detected Emotions"
                            )
                            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown("#### Face Recognition Database")
            st.info("Add faces to the database and recognize them in new images.")
            
            # Add new face
            st.markdown("##### Add New Face")
            
            new_face_image = st.file_uploader(
                "Upload face image",
                type=['png', 'jpg', 'jpeg'],
                key="new_face"
            )
            
            new_face_name = st.text_input("Person's name")
            
            if st.button("➕ Add to Database") and new_face_image and new_face_name:
                st.success(f"✅ Added {new_face_name} to the face database!")
            
            # Recognition
            st.markdown("##### Recognize Faces")
            
            recognition_image = st.file_uploader(
                "Upload image for recognition",
                type=['png', 'jpg', 'jpeg'],
                key="recognition"
            )
            
            if recognition_image:
                st.image(Image.open(recognition_image), caption="Recognition Image", width=300)
                
                if st.button("🔍 Recognize Faces"):
                    with st.spinner("Recognizing faces..."):
                        time.sleep(1)
                        
                        # Mock recognition results
                        recognition_results = [
                            {"name": "John Doe", "confidence": 0.94, "status": "Known"},
                            {"name": "Unknown", "confidence": 0.0, "status": "Unknown"}
                        ]
                        
                        st.markdown("### 🎯 Recognition Results")
                        
                        for result in recognition_results:
                            if result["status"] == "Known":
                                st.success(f"✅ Recognized: **{result['name']}** ({result['confidence']:.1%})")
                            else:
                                st.warning("❓ Unknown person detected")

    def render_llm_page(self):
        """Render LLM and chatbot page"""
        st.markdown("# 🤖 LLM & Chatbots")
        
        llm_models = list(self.models['llm'].keys())
        selected_model = st.selectbox("Select LLM Model", llm_models)
        
        model_info = self.models['llm'][selected_model]
        
        st.markdown(f"## {model_info['name']}")
        st.markdown(f"*{model_info['description']}*")
        
        if model_info['status'] != 'available':
            st.error("⚠️ This model is currently unavailable.")
            return
        
        if selected_model == 'chatbot':
            self.render_chatbot_interface()
        elif selected_model == 'rag_system':
            self.render_rag_interface()

    def render_chatbot_interface(self):
        """Render chatbot interface"""
        st.markdown("### 💬 AI Chatbot")
        
        # Chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Chatbot settings
        with st.sidebar:
            st.markdown("### 🎛️ Chat Settings")
            
            personality = st.selectbox(
                "Personality",
                ["Professional", "Friendly", "Creative", "Technical"]
            )
            
            response_length = st.selectbox(
                "Response Length",
                ["Short", "Medium", "Long"]
            )
            
            temperature = st.slider("Creativity", 0.0, 1.0, 0.7)
            
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()
        
        # Display chat history
        st.markdown("### 💭 Conversation")
        
        # Create chat container
        chat_container = st.container()
        
        with chat_container:
            for i, message in enumerate(st.session_state.chat_history):
                if message['role'] == 'user':
                    st.markdown(f"""
                    <div style="background-color: #e3f2fd; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0;">
                        <strong>You:</strong> {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background-color: #f3e5f5; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0;">
                        <strong>AI:</strong> {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Chat input
        user_input = st.text_input(
            "Type your message:",
            placeholder="Ask me anything...",
            key="chat_input"
        )
        
        col1, col2 = st.columns([6, 1])
        
        with col2:
            send_button = st.button("📤 Send", type="primary")
        
        if send_button and user_input:
            # Add user message
            st.session_state.chat_history.append({
                'role': 'user',
                'content': user_input,
                'timestamp': datetime.now()
            })
            
            # Generate AI response (mock)
            with st.spinner("AI is thinking..."):
                time.sleep(1)
                
                # Mock response based on personality
                responses = {
                    "Professional": f"Thank you for your question about '{user_input}'. Based on my analysis, I can provide you with a comprehensive response.",
                    "Friendly": f"Hey there! That's a great question about '{user_input}'. I'd be happy to help you with that!",
                    "Creative": f"What an interesting question about '{user_input}'! Let me think creatively about this...",
                    "Technical": f"Analyzing your query regarding '{user_input}'. Here's a detailed technical breakdown:"
                }
                
                ai_response = responses[personality]
                
                # Add AI response
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': ai_response,
                    'timestamp': datetime.now()
                })
            
            st.rerun()
        
        # Chat statistics
        if st.session_state.chat_history:
            st.markdown("---")
            st.markdown("### 📊 Chat Statistics")
            
            total_messages = len(st.session_state.chat_history)
            user_messages = len([m for m in st.session_state.chat_history if m['role'] == 'user'])
            ai_messages = len([m for m in st.session_state.chat_history if m['role'] == 'assistant'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Messages", total_messages)
            with col2:
                st.metric("Your Messages", user_messages)
            with col3:
                st.metric("AI Messages", ai_messages)

    def render_rag_interface(self):
        """Render RAG system interface"""
        st.markdown("### 🔍 RAG System - Knowledge-Based Q&A")
        
        st.info("RAG (Retrieval-Augmented Generation) combines information retrieval with text generation for accurate, knowledge-based responses.")
        
        tab1, tab2 = st.tabs(["Query Documents", "Manage Knowledge Base"])
        
        with tab1:
            # Knowledge base selection
            knowledge_bases = ["General Knowledge", "Technical Documentation", "Company Policies", "Product Manuals"]
            selected_kb = st.selectbox("Select Knowledge Base", knowledge_bases)
            
            # Query input
            query = st.text_area(
                "Enter your question:",
                placeholder="Ask a question about the selected knowledge base...",
                height=100
            )
            
            # RAG settings
            col1, col2 = st.columns(2)
            with col1:
                retrieval_k = st.slider("Retrieve top K documents", 1, 10, 5)
                similarity_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.7)
            
            with col2:
                include_sources = st.checkbox("Include source citations", value=True)
                rerank_results = st.checkbox("Re-rank results", value=True)
            
            if st.button("🔍 Query Knowledge Base", type="primary") and query:
                with st.spinner("Searching knowledge base and generating response..."):
                    time.sleep(2)
                    
                    # Mock RAG response
                    rag_response = {
                        'answer': f"Based on the available documentation in {selected_kb}, here's what I found regarding your question: '{query}'. The system retrieved relevant information from multiple sources to provide you with this comprehensive answer.",
                        'confidence': np.random.uniform(0.7, 0.95),
                        'sources': [
                            {
                                'title': 'Document 1: Technical Overview',
                                'content': 'Relevant excerpt from the first document...',
                                'similarity': 0.92,
                                'source': 'tech_docs/overview.pdf'
                            },
                            {
                                'title': 'Document 2: Implementation Guide',
                                'content': 'Another relevant excerpt...',
                                'similarity': 0.87,
                                'source': 'guides/implementation.md'
                            }
                        ]
                    }
                    
                    # Display answer
                    st.markdown("### 💡 Generated Answer")
                    st.markdown(f"""
                    <div class="success-box">
                        {rag_response['answer']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Answer metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Confidence", f"{rag_response['confidence']:.1%}")
                    with col2:
                        st.metric("Sources Used", len(rag_response['sources']))
                    with col3:
                        st.metric("Knowledge Base", selected_kb)
                    
                    # Source citations
                    if include_sources:
                        st.markdown("### 📚 Source Documents")
                        
                        for i, source in enumerate(rag_response['sources']):
                            with st.expander(f"Source {i+1}: {source['title']} (Similarity: {source['similarity']:.1%})"):
                                st.markdown(f"**Content:** {source['content']}")
                                st.markdown(f"**Source:** `{source['source']}`")
                    
                    # Relevance visualization
                    st.markdown("### 📊 Document Relevance")
                    
                    source_df = pd.DataFrame(rag_response['sources'])
                    
                    fig = px.bar(
                        source_df, 
                        x='title', 
                        y='similarity',
                        title="Document Similarity Scores",
                        color='similarity',
                        color_continuous_scale="viridis"
                    )
                    
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown("#### 📁 Knowledge Base Management")
            
            # Upload documents
            st.markdown("##### Upload New Documents")
            
            uploaded_docs = st.file_uploader(
                "Upload documents to knowledge base",
                type=['pdf', 'txt', 'md', 'docx'],
                accept_multiple_files=True
            )
            
            if uploaded_docs:
                st.write(f"Selected {len(uploaded_docs)} files:")
                for doc in uploaded_docs:
                    st.write(f"- {doc.name}")
                
                if st.button("📤 Upload to Knowledge Base"):
                    with st.spinner("Processing and indexing documents..."):
                        time.sleep(2)
                        st.success(f"✅ Successfully uploaded and indexed {len(uploaded_docs)} documents!")
            
            # Knowledge base statistics
            st.markdown("##### 📊 Knowledge Base Statistics")
            
            kb_stats = {
                'Total Documents': 1247,
                'Total Chunks': 8934,
                'Average Similarity': 0.847,
                'Last Updated': '2024-01-15',
                'Index Size': '2.3 GB'
            }
            
            for stat, value in kb_stats.items():
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.write(f"**{stat}:**")
                with col2:
                    st.write(str(value))

    def render_data_explorer_page(self):
        """Render data exploration page"""
        st.markdown("# 📊 Data Explorer")
        
        st.markdown("Explore and analyze your datasets with interactive visualizations.")
        
        # Sample data or file upload
        data_source = st.radio("Data Source", ["Sample Data", "Upload File", "Connect Database"])
        
        df = None
        
        if data_source == "Sample Data":
            # Generate sample data
            sample_datasets = {
                "Sales Data": pd.DataFrame({
                    'date': pd.date_range('2024-01-01', periods=100, freq='D'),
                    'sales': np.random.normal(1000, 200, 100),
                    'category': np.random.choice(['A', 'B', 'C'], 100),
                    'region': np.random.choice(['North', 'South', 'East', 'West'], 100)
                }),
                "Customer Data": pd.DataFrame({
                    'age': np.random.normal(35, 10, 500),
                    'income': np.random.normal(50000, 15000, 500),
                    'satisfaction': np.random.uniform(1, 5, 500),
                    'segment': np.random.choice(['Premium', 'Standard', 'Basic'], 500)
                })
            }
            
            selected_dataset = st.selectbox("Select Sample Dataset", list(sample_datasets.keys()))
            df = sample_datasets[selected_dataset]
        
        elif data_source == "Upload File":
            uploaded_file = st.file_uploader(
                "Upload CSV or Excel file",
                type=['csv', 'xlsx', 'xls']
            )
            
            if uploaded_file:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                except Exception as e:
                    st.error(f"Error loading file: {e}")
        
        elif data_source == "Connect Database":
            st.info("Database connection feature coming soon!")
        
        if df is not None:
            # Dataset overview
            st.markdown("### 📋 Dataset Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Rows", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            with col4:
                st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
            
            # Data preview
            st.markdown("### 👀 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Column analysis
            st.markdown("### 📊 Column Analysis")
            
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
            
            if numeric_columns:
                st.markdown("#### 🔢 Numeric Columns")
                
                selected_numeric = st.multiselect(
                    "Select numeric columns to analyze",
                    numeric_columns,
                    default=numeric_columns[:3]
                )
                
                if selected_numeric:
                    # Statistics
                    st.markdown("##### Descriptive Statistics")
                    st.dataframe(df[selected_numeric].describe(), use_container_width=True)
                    
                    # Distribution plots
                    st.markdown("##### Distribution Plots")
                    
                    for col in selected_numeric:
                        fig = px.histogram(
                            df, x=col,
                            title=f"Distribution of {col}",
                            marginal="box"
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            if categorical_columns:
                st.markdown("#### 📝 Categorical Columns")
                
                selected_categorical = st.selectbox(
                    "Select categorical column to analyze",
                    categorical_columns
                )
                
                if selected_categorical:
                    # Value counts
                    value_counts = df[selected_categorical].value_counts()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("##### Value Counts")
                        st.dataframe(value_counts.reset_index(), use_container_width=True)
                    
                    with col2:
                        st.markdown("##### Distribution")
                        fig = px.pie(
                            values=value_counts.values,
                            names=value_counts.index,
                            title=f"Distribution of {selected_categorical}"
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            # Correlation analysis
            if len(numeric_columns) > 1:
                st.markdown("### 🔗 Correlation Analysis")
                
                correlation_matrix = df[numeric_columns].corr()
                
                fig = px.imshow(
                    correlation_matrix,
                    title="Correlation Matrix",
                    color_continuous_scale="RdBu",
                    aspect="auto"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Custom visualization
            st.markdown("### 🎨 Custom Visualization")
            
            viz_type = st.selectbox(
                "Visualization Type",
                ["Scatter Plot", "Line Chart", "Bar Chart", "Box Plot"]
            )
            
            if viz_type == "Scatter Plot" and len(numeric_columns) >= 2:
                col1, col2 = st.columns(2)
                with col1:
                    x_axis = st.selectbox("X-axis", numeric_columns)
                with col2:
                    y_axis = st.selectbox("Y-axis", [col for col in numeric_columns if col != x_axis])
                
                color_by = st.selectbox("Color by", ["None"] + categorical_columns)
                
                if x_axis and y_axis:
                    fig = px.scatter(
                        df, x=x_axis, y=y_axis,
                        color=color_by if color_by != "None" else None,
                        title=f"{y_axis} vs {x_axis}"
                    )
                    st.plotly_chart(fig, use_container_width=True)

    def render_analytics_page(self):
        """Render analytics page"""
        st.markdown("# 📈 Analytics")
        
        st.markdown("Monitor system performance, model accuracy, and usage statistics.")
        
        # Time range selector
        time_range = st.selectbox(
            "Time Range",
            ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "Last 90 Days"]
        )
        
        # Generate mock analytics data
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        analytics_data = pd.DataFrame({
            'date': dates,
            'requests': np.random.poisson(100, 30),
            'accuracy': np.random.normal(0.92, 0.02, 30),
            'response_time': np.random.normal(1.5, 0.3, 30),
            'errors': np.random.poisson(2, 30)
        })
        
        # Key metrics
        st.markdown("### 📊 Key Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_requests = analytics_data['requests'].sum()
            st.metric(
                "Total Requests",
                f"{total_requests:,}",
                delta=f"+{np.random.randint(50, 200)}"
            )
        
        with col2:
            avg_accuracy = analytics_data['accuracy'].mean()
            st.metric(
                "Average Accuracy",
                f"{avg_accuracy:.1%}",
                delta=f"+{np.random.uniform(0.5, 2.0):.1f}%"
            )
        
        with col3:
            avg_response_time = analytics_data['response_time'].mean()
            st.metric(
                "Avg Response Time",
                f"{avg_response_time:.2f}s",
                delta=f"-{np.random.uniform(0.1, 0.3):.2f}s"
            )
        
        with col4:
            total_errors = analytics_data['errors'].sum()
            error_rate = total_errors / total_requests * 100
            st.metric(
                "Error Rate",
                f"{error_rate:.2f}%",
                delta=f"-{np.random.uniform(0.1, 0.5):.1f}%"
            )
        
        # Charts
        st.markdown("### 📈 Trends")
        
        tab1, tab2, tab3 = st.tabs(["Usage", "Performance", "Errors"])
        
        with tab1:
            fig = px.line(
                analytics_data, x='date', y='requests',
                title="Daily Request Volume",
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Performance metrics
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Model Accuracy", "Response Time"),
                vertical_spacing=0.1
            )
            
            fig.add_trace(
                go.Scatter(x=analytics_data['date'], y=analytics_data['accuracy'],
                          mode='lines+markers', name='Accuracy'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=analytics_data['date'], y=analytics_data['response_time'],
                          mode='lines+markers', name='Response Time'),
                row=2, col=1
            )
            
            fig.update_layout(height=500, title="Performance Metrics Over Time")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            fig = px.bar(
                analytics_data, x='date', y='errors',
                title="Daily Error Count"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Model performance breakdown
        st.markdown("### 🎯 Model Performance")
        
        model_performance = pd.DataFrame({
            'Model': ['Sentiment Analysis', 'Image Classification', 'Object Detection', 'Face Recognition'],
            'Accuracy': [0.94, 0.91, 0.88, 0.96],
            'Usage': [450, 320, 180, 240],
            'Avg Response Time': [0.8, 2.1, 3.2, 1.4]
        })
        
        fig = px.scatter(
            model_performance, 
            x='Usage', y='Accuracy',
            size='Avg Response Time',
            color='Model',
            title="Model Performance vs Usage"
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def run(self):
        """Run the dashboard application"""
        self.render_sidebar()
        
        # Route to appropriate page
        if st.session_state.page == 'Home':
            self.render_home_page()
        elif st.session_state.page == 'NLP Models':
            self.render_nlp_page()
        elif st.session_state.page == 'Computer Vision':
            self.render_computer_vision_page()
        elif st.session_state.page == 'LLM & Chatbots':
            self.render_llm_page()
        elif st.session_state.page == 'Data Explorer':
            self.render_data_explorer_page()
        elif st.session_state.page == 'Analytics':
            self.render_analytics_page()
        elif st.session_state.page == 'Settings':
            st.markdown("# ⚙️ Settings")
            st.info("Settings page - Configure system preferences, API keys, and model parameters.")
        else:
            st.markdown("# 🔧 Model Training")
            st.info("Model training interface - Train custom models with your data.")

# Run the dashboard
if __name__ == "__main__":
    dashboard = AILabDashboard()
    dashboard.run()
