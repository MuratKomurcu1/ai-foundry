import json
import asyncio
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import sqlite3
from pathlib import Path

# Core AI libraries
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    pipeline, AutoProcessor, AutoModel
)
from sentence_transformers import SentenceTransformer
import openai
from anthropic import Anthropic

# RAG components
import chromadb
from chromadb.config import Settings
import faiss
import numpy as np

# Web and API
import requests
from bs4 import BeautifulSoup
import wikipedia

# Utilities
import re
from PIL import Image
import base64
import io

@dataclass
class Message:
    """Chat mesajı"""
    role: str  # user, assistant, system, function
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = None
    function_call: Dict[str, Any] = None
    
    def to_dict(self):
        return {
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata or {},
            'function_call': self.function_call
        }

@dataclass
class ChatSession:
    """Chat oturumu"""
    session_id: str
    user_id: str
    messages: List[Message]
    context: Dict[str, Any]
    created_at: datetime
    last_activity: datetime

class Memory:
    """Chatbot bellek sistemi"""
    
    def __init__(self, db_path: str = "chatbot_memory.db"):
        self.db_path = db_path
        self.setup_database()
        
        # Embedding modeli
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Vector database
        self.chroma_client = chromadb.Client(Settings(
            persist_directory="./chroma_db"
        ))
        
        try:
            self.memory_collection = self.chroma_client.get_collection("memory")
        except:
            self.memory_collection = self.chroma_client.create_collection(
                name="memory",
                metadata={"description": "Chatbot long-term memory"}
            )
    
    def setup_database(self):
        """SQLite veritabanını kurma"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Chat sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT,
                context TEXT,
                created_at TIMESTAMP,
                last_activity TIMESTAMP
            )
        ''')
        
        # Messages table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                timestamp TIMESTAMP,
                metadata TEXT,
                FOREIGN KEY (session_id) REFERENCES chat_sessions (session_id)
            )
        ''')
        
        # User preferences
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id TEXT PRIMARY KEY,
                preferences TEXT,
                updated_at TIMESTAMP
            )
        ''')
        
        # Long-term memory
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS long_term_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                memory_type TEXT,
                content TEXT,
                importance_score REAL,
                created_at TIMESTAMP,
                last_accessed TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_session(self, session: ChatSession):
        """Chat session'ı kaydetme"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Session kaydet
        cursor.execute('''
            INSERT OR REPLACE INTO chat_sessions 
            (session_id, user_id, context, created_at, last_activity)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            session.session_id,
            session.user_id,
            json.dumps(session.context),
            session.created_at,
            session.last_activity
        ))
        
        # Messages kaydet
        for message in session.messages:
            cursor.execute('''
                INSERT INTO messages 
                (session_id, role, content, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                session.session_id,
                message.role,
                message.content,
                message.timestamp,
                json.dumps(message.metadata or {})
            ))
        
        conn.commit()
        conn.close()
    
    def load_session(self, session_id: str) -> Optional[ChatSession]:
        """Chat session'ı yükleme"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Session bilgisi
        cursor.execute('''
            SELECT user_id, context, created_at, last_activity 
            FROM chat_sessions WHERE session_id = ?
        ''', (session_id,))
        
        session_data = cursor.fetchone()
        if not session_data:
            conn.close()
            return None
        
        user_id, context_json, created_at, last_activity = session_data
        
        # Messages
        cursor.execute('''
            SELECT role, content, timestamp, metadata 
            FROM messages WHERE session_id = ?
            ORDER BY timestamp
        ''', (session_id,))
        
        messages = []
        for role, content, timestamp, metadata_json in cursor.fetchall():
            messages.append(Message(
                role=role,
                content=content,
                timestamp=datetime.fromisoformat(timestamp),
                metadata=json.loads(metadata_json)
            ))
        
        conn.close()
        
        return ChatSession(
            session_id=session_id,
            user_id=user_id,
            messages=messages,
            context=json.loads(context_json),
            created_at=datetime.fromisoformat(created_at),
            last_activity=datetime.fromisoformat(last_activity)
        )
    
    def add_to_long_term_memory(self, user_id: str, content: str, 
                               memory_type: str = "conversation", 
                               importance_score: float = 0.5):
        """Uzun dönem belleğe ekleme"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO long_term_memory 
            (user_id, memory_type, content, importance_score, created_at, last_accessed)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            user_id, memory_type, content, importance_score,
            datetime.now(), datetime.now()
        ))
        
        conn.commit()
        conn.close()
        
        # Vector store'a da ekle
        embedding = self.embedder.encode([content])
        
        self.memory_collection.add(
            embeddings=embedding.tolist(),
            documents=[content],
            metadatas=[{
                "user_id": user_id,
                "memory_type": memory_type,
                "importance_score": importance_score,
                "timestamp": datetime.now().isoformat()
            }],
            ids=[f"{user_id}_{datetime.now().timestamp()}"]
        )
    
    def retrieve_relevant_memories(self, query: str, user_id: str, 
                                  limit: int = 5) -> List[Dict]:
        """İlgili anıları getirme"""
        query_embedding = self.embedder.encode([query])
        
        results = self.memory_collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=limit,
            where={"user_id": user_id}
        )
        
        memories = []
        if results['documents']:
            for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                memories.append({
                    'content': doc,
                    'metadata': metadata
                })
        
        return memories

class FunctionRegistry:
    """Function calling sistemi"""
    
    def __init__(self):
        self.functions: Dict[str, Callable] = {}
        self.function_schemas: Dict[str, Dict] = {}
        
        # Varsayılan fonksiyonları kaydet
        self.register_default_functions()
    
    def register_function(self, name: str, func: Callable, schema: Dict):
        """Fonksiyon kaydetme"""
        self.functions[name] = func
        self.function_schemas[name] = schema
    
    def register_default_functions(self):
        """Varsayılan fonksiyonları kaydetme"""
        
        # Web arama
        self.register_function(
            "web_search",
            self.web_search,
            {
                "name": "web_search",
                "description": "İnternette arama yapar",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Arama sorgusu"}
                    },
                    "required": ["query"]
                }
            }
        )
        
        # Wikipedia arama
        self.register_function(
            "wikipedia_search",
            self.wikipedia_search,
            {
                "name": "wikipedia_search",
                "description": "Wikipedia'da arama yapar",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Arama terimi"},
                        "lang": {"type": "string", "description": "Dil kodu", "default": "tr"}
                    },
                    "required": ["query"]
                }
            }
        )
        
        # Hesap makinesi
        self.register_function(
            "calculator",
            self.calculator,
            {
                "name": "calculator",
                "description": "Matematiksel hesaplamalar yapar",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Matematik ifadesi"}
                    },
                    "required": ["expression"]
                }
            }
        )
        
        # Zaman bilgisi
        self.register_function(
            "get_time",
            self.get_time,
            {
                "name": "get_time",
                "description": "Şu anki zamanı verir",
                "parameters": {"type": "object", "properties": {}}
            }
        )
    
    async def web_search(self, query: str) -> str:
        """Web arama fonksiyonu"""
        try:
            # Basit web arama simülasyonu
            search_url = f"https://www.google.com/search?q={query}"
            response = requests.get(search_url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # Arama sonuçlarını çek (basit versiyon)
            for result in soup.find_all('div', class_='BVG0Nb')[:3]:
                text = result.get_text()
                if text:
                    results.append(text[:200])
            
            return f"Arama sonuçları '{query}' için:\n" + "\n".join(results)
            
        except Exception as e:
            return f"Arama sırasında hata oluştu: {str(e)}"
    
    async def wikipedia_search(self, query: str, lang: str = "tr") -> str:
        """Wikipedia arama fonksiyonu"""
        try:
            wikipedia.set_lang(lang)
            summary = wikipedia.summary(query, sentences=3)
            return f"Wikipedia '{query}' hakkında:\n{summary}"
        except wikipedia.exceptions.DisambiguationError as e:
            return f"Belirsiz terim. Seçenekler: {', '.join(e.options[:5])}"
        except wikipedia.exceptions.PageError:
            return f"'{query}' için Wikipedia sayfası bulunamadı."
        except Exception as e:
            return f"Wikipedia aramasında hata: {str(e)}"
    
    async def calculator(self, expression: str) -> str:
        """Hesap makinesi fonksiyonu"""
        try:
            # Güvenli matematik değerlendirmesi
            allowed_chars = set("0123456789+-*/.() ")
            if not all(c in allowed_chars for c in expression):
                return "Hata: Sadece matematik operasyonlarına izin verilir."
            
            result = eval(expression, {"__builtins__": {}}, {})
            return f"{expression} = {result}"
        except Exception as e:
            return f"Hesaplama hatası: {str(e)}"
    
    async def get_time(self) -> str:
        """Zaman bilgisi fonksiyonu"""
        now = datetime.now()
        return f"Şu anki zaman: {now.strftime('%Y-%m-%d %H:%M:%S')}"
    
    async def call_function(self, function_name: str, parameters: Dict) -> str:
        """Fonksiyon çağırma"""
        if function_name not in self.functions:
            return f"Bilinmeyen fonksiyon: {function_name}"
        
        try:
            func = self.functions[function_name]
            result = await func(**parameters)
            return str(result)
        except Exception as e:
            return f"Fonksiyon çağırma hatası: {str(e)}"

class LLMProvider:
    """LLM sağlayıcıları için abstract sınıf"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    async def generate_response(self, messages: List[Message], 
                              functions: List[Dict] = None) -> Message:
        raise NotImplementedError

class HuggingFaceLLM(LLMProvider):
    """Hugging Face LLM wrapper"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        super().__init__(model_name)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Pad token ayarla
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    async def generate_response(self, messages: List[Message], 
                              functions: List[Dict] = None) -> Message:
        """Response üretme"""
        
        # Conversation history oluştur
        conversation = self._format_conversation(messages)
        
        # Tokenize
        inputs = self.tokenizer.encode(conversation, return_tensors='pt', max_length=512, truncation=True)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + 100,
                num_beams=4,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Sadece yeni kısmı al
        new_response = response[len(conversation):].strip()
        
        return Message(
            role="assistant",
            content=new_response,
            timestamp=datetime.now()
        )
    
    def _format_conversation(self, messages: List[Message]) -> str:
        """Konuşmayı formatla"""
        formatted = ""
        for msg in messages[-5:]:  # Son 5 mesaj
            if msg.role == "user":
                formatted += f"User: {msg.content}\n"
            elif msg.role == "assistant":
                formatted += f"Bot: {msg.content}\n"
        
        formatted += "Bot: "
        return formatted

class OpenAILLM(LLMProvider):
    """OpenAI LLM wrapper"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", api_key: str = None):
        super().__init__(model_name)
        
        if api_key:
            openai.api_key = api_key
        
        self.client = openai.OpenAI(api_key=api_key)
    
    async def generate_response(self, messages: List[Message], 
                              functions: List[Dict] = None) -> Message:
        """OpenAI ile response üretme"""
        
        # Messages formatla
        formatted_messages = []
        for msg in messages[-10:]:  # Son 10 mesaj
            formatted_messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        try:
            # Function calling ile
            if functions:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=formatted_messages,
                    functions=functions,
                    function_call="auto"
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=formatted_messages
                )
            
            message = response.choices[0].message
            
            # Function call kontrolü
            if hasattr(message, 'function_call') and message.function_call:
                return Message(
                    role="assistant",
                    content=message.content or "",
                    timestamp=datetime.now(),
                    function_call={
                        "name": message.function_call.name,
                        "arguments": json.loads(message.function_call.arguments)
                    }
                )
            else:
                return Message(
                    role="assistant",
                    content=message.content,
                    timestamp=datetime.now()
                )
                
        except Exception as e:
            return Message(
                role="assistant",
                content=f"Üzgünüm, bir hata oluştu: {str(e)}",
                timestamp=datetime.now()
            )

class MultimodalProcessor:
    """Çok modlu (text, image, audio) işleme"""
    
    def __init__(self):
        # Image processing
        try:
            self.image_processor = AutoProcessor.from_pretrained("microsoft/git-base")
            self.image_model = AutoModel.from_pretrained("microsoft/git-base")
            self.image_available = True
        except:
            self.image_available = False
            logging.warning("Image processing model not available")
    
    async def process_image(self, image_path: str) -> str:
        """Görüntü analizi"""
        if not self.image_available:
            return "Görüntü işleme mevcut değil."
        
        try:
            image = Image.open(image_path)
            
            # Image captioning
            inputs = self.image_processor(images=image, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.image_model.generate(**inputs, max_length=50)
            
            caption = self.image_processor.decode(outputs[0], skip_special_tokens=True)
            
            return f"Görüntü açıklaması: {caption}"
            
        except Exception as e:
            return f"Görüntü işleme hatası: {str(e)}"
    
    def encode_image_base64(self, image_path: str) -> str:
        """Görüntüyü base64'e çevirme"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

class IntelligentChatbot:
    """Ana chatbot sınıfı"""
    
    def __init__(self, llm_provider: LLMProvider, config: Dict = None):
        """
        Args:
            llm_provider: LLM sağlayıcısı
            config: Konfigürasyon
        """
        self.llm_provider = llm_provider
        self.config = config or {}
        
        # Components
        self.memory = Memory()
        self.function_registry = FunctionRegistry()
        self.multimodal = MultimodalProcessor()
        
        # Current session
        self.current_session: Optional[ChatSession] = None
        
        # Personality and behavior
        self.system_prompt = self.config.get('system_prompt', 
            "Sen yardımksever, bilgili ve dostane bir AI asistanısın. "
            "Kullanıcılara en iyi şekilde yardım etmeye çalışırsın.")
        
        logging.info(f"Chatbot initialized with {llm_provider.model_name}")
    
    def start_session(self, user_id: str, session_id: str = None) -> str:
        """Yeni chat session başlatma"""
        if not session_id:
            session_id = f"{user_id}_{datetime.now().timestamp()}"
        
        # Existing session kontrolü
        existing_session = self.memory.load_session(session_id)
        
        if existing_session:
            self.current_session = existing_session
            self.current_session.last_activity = datetime.now()
        else:
            # Yeni session
            self.current_session = ChatSession(
                session_id=session_id,
                user_id=user_id,
                messages=[],
                context={},
                created_at=datetime.now(),
                last_activity=datetime.now()
            )
            
            # System message ekle
            system_message = Message(
                role="system",
                content=self.system_prompt,
                timestamp=datetime.now()
            )
            self.current_session.messages.append(system_message)
        
        return session_id
    
    async def chat(self, user_input: str, image_path: str = None) -> str:
        """Ana chat fonksiyonu"""
        if not self.current_session:
            raise ValueError("Session başlatılmamış. start_session() çağırın.")
        
        # Görüntü varsa işle
        if image_path:
            image_description = await self.multimodal.process_image(image_path)
            user_input = f"{user_input}\n\n[Görüntü: {image_description}]"
        
        # User message ekle
        user_message = Message(
            role="user",
            content=user_input,
            timestamp=datetime.now()
        )
        self.current_session.messages.append(user_message)
        
        # Relevant memories getir
        relevant_memories = self.memory.retrieve_relevant_memories(
            user_input, self.current_session.user_id
        )
        
        # Context oluştur
        context_info = ""
        if relevant_memories:
            context_info = "\n\nİlgili anılar:\n"
            for memory in relevant_memories:
                context_info += f"- {memory['content']}\n"
        
        # Enhanced user input
        enhanced_input = user_input + context_info
        
        # LLM'den response al
        functions = list(self.function_registry.function_schemas.values())
        
        # Son mesajları al (memory için)
        recent_messages = self.current_session.messages[-10:]
        
        try:
            response = await self.llm_provider.generate_response(
                recent_messages, functions
            )
            
            # Function call kontrolü
            if response.function_call:
                # Function çağır
                function_result = await self.function_registry.call_function(
                    response.function_call["name"],
                    response.function_call["arguments"]
                )
                
                # Function result ile yeni response al
                function_message = Message(
                    role="function",
                    content=function_result,
                    timestamp=datetime.now()
                )
                self.current_session.messages.append(function_message)
                
                # Final response
                final_response = await self.llm_provider.generate_response(
                    self.current_session.messages[-10:]
                )
                
                response_content = final_response.content
            else:
                response_content = response.content
            
            # Assistant message ekle
            assistant_message = Message(
                role="assistant",
                content=response_content,
                timestamp=datetime.now()
            )
            self.current_session.messages.append(assistant_message)
            
            # Memory'ye önemli bilgileri ekle
            await self._update_memory(user_input, response_content)
            
            # Session kaydet
            self.current_session.last_activity = datetime.now()
            self.memory.save_session(self.current_session)
            
            return response_content
            
        except Exception as e:
            error_message = f"Üzgünüm, bir hata oluştu: {str(e)}"
            
            error_msg = Message(
                role="assistant",
                content=error_message,
                timestamp=datetime.now()
            )
            self.current_session.messages.append(error_msg)
            
            return error_message
    
    async def _update_memory(self, user_input: str, assistant_response: str):
        """Memory güncelleme"""
        # Önemli bilgileri tespit et ve memory'ye ekle
        conversation = f"User: {user_input}\nAssistant: {assistant_response}"
        
        # Basit importance scoring
        importance_keywords = ['önemli', 'unutma', 'hatırla', 'kaydet', 'not al']
        importance_score = 0.5
        
        for keyword in importance_keywords:
            if keyword in user_input.lower():
                importance_score = 0.8
                break
        
        # Memory'ye ekle
        self.memory.add_to_long_term_memory(
            self.current_session.user_id,
            conversation,
            "conversation",
            importance_score
        )
    
    def get_conversation_history(self, limit: int = 10) -> List[Dict]:
        """Konuşma geçmişini getirme"""
        if not self.current_session:
            return []
        
        history = []
        for message in self.current_session.messages[-limit:]:
            if message.role in ['user', 'assistant']:
                history.append(message.to_dict())
        
        return history
    
    def get_session_stats(self) -> Dict:
        """Session istatistikleri"""
        if not self.current_session:
            return {}
        
        user_messages = sum(1 for msg in self.current_session.messages if msg.role == 'user')
        assistant_messages = sum(1 for msg in self.current_session.messages if msg.role == 'assistant')
        
        return {
            'session_id': self.current_session.session_id,
            'user_id': self.current_session.user_id,
            'total_messages': len(self.current_session.messages),
            'user_messages': user_messages,
            'assistant_messages': assistant_messages,
            'session_duration': (datetime.now() - self.current_session.created_at).total_seconds(),
            'last_activity': self.current_session.last_activity.isoformat()
        }

class ChatbotAPI:
    """REST API wrapper"""
    
    def __init__(self, chatbot: IntelligentChatbot):
        self.chatbot = chatbot
        self.active_sessions: Dict[str, str] = {}  # user_id -> session_id
    
    async def start_chat(self, user_id: str) -> Dict:
        """Chat başlatma endpoint"""
        try:
            session_id = self.chatbot.start_session(user_id)
            self.active_sessions[user_id] = session_id
            
            return {
                'success': True,
                'session_id': session_id,
                'message': 'Chat başlatıldı! Nasıl yardımcı olabilirim?'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def send_message(self, user_id: str, message: str, 
                          image_path: str = None) -> Dict:
        """Mesaj gönderme endpoint"""
        try:
            # Session kontrolü
            if user_id not in self.active_sessions:
                await self.start_chat(user_id)
            
            response = await self.chatbot.chat(message, image_path)
            
            return {
                'success': True,
                'response': response,
                'session_stats': self.chatbot.get_session_stats()
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def get_history(self, user_id: str, limit: int = 10) -> Dict:
        """Geçmiş getirme endpoint"""
        try:
            if user_id not in self.active_sessions:
                return {'success': False, 'error': 'Aktif session bulunamadı'}
            
            history = self.chatbot.get_conversation_history(limit)
            
            return {
                'success': True,
                'history': history
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

# Demo ve test fonksiyonları
async def demo_intelligent_chatbot():
    """Chatbot demo"""
    print("🤖 Intelligent Chatbot Demo")
    print("=" * 50)
    
    # LLM provider seç
    print("LLM Provider seçiliyor...")
    
    try:
        # HuggingFace provider (local)
        llm_provider = HuggingFaceLLM("microsoft/DialoGPT-medium")
        print("✅ HuggingFace LLM yüklendi")
    except Exception as e:
        print(f"❌ HuggingFace LLM yüklenemedi: {e}")
        return
    
    # Chatbot oluştur
    config = {
        'system_prompt': (
            "Sen Türkçe konuşan, yardımsever bir AI asistanısın. "
            "Kullanıcılara bilgili ve dostane bir şekilde yardım ediyorsun. "
            "Gerektiğinde fonksiyonları kullanarak güncel bilgi sağlayabilirsin."
        )
    }
    
    chatbot = IntelligentChatbot(llm_provider, config)
    
    # API wrapper
    api = ChatbotAPI(chatbot)
    
    # Demo conversation
    user_id = "demo_user"
    
    print(f"\n👤 User: {user_id}")
    print("-" * 30)
    
    # Session başlat
    start_result = await api.start_chat(user_id)
    print(f"🟢 Session: {start_result}")
    
    # Test mesajları
    test_messages = [
        "Merhaba! Ben Kim?",
        "Yapay zeka hakkında bilgi verebilir misin?",
        "2+2 kaç eder?",
        "Wikipedia'dan Python programlama dili hakkında bilgi ara",
        "Şu anki saat kaç?"
    ]
    
    for message in test_messages:
        print(f"\n👤 User: {message}")
        
        response = await api.send_message(user_id, message)
        
        if response['success']:
            print(f"🤖 Bot: {response['response']}")
            
            # Stats göster
            stats = response['session_stats']
            print(f"📊 Messages: {stats['user_messages']}↔️{stats['assistant_messages']}")
        else:
            print(f"❌ Error: {response['error']}")
        
        print("-" * 50)
    
    # History göster
    print("\n📚 Conversation History:")
    history_result = await api.get_history(user_id, 5)
    
    if history_result['success']:
        for msg in history_result['history']:
            role = "👤" if msg['role'] == 'user' else "🤖"
            print(f"{role} {msg['content'][:100]}...")
    
    print("\n✨ Demo tamamlandı!")

async def demo_custom_functions():
    """Custom function demo"""
    print("\n🔧 Custom Functions Demo")
    print("=" * 30)
    
    # Function registry
    registry = FunctionRegistry()
    
    # Custom function ekle
    async def get_weather(city: str) -> str:
        """Hava durumu fonksiyonu (demo)"""
        weather_data = {
            "istanbul": "15°C, Parçalı bulutlu",
            "ankara": "8°C, Karlı", 
            "izmir": "18°C, Güneşli"
        }
        
        return weather_data.get(city.lower(), f"{city} için hava durumu bilgisi bulunamadı")
    
    registry.register_function(
        "get_weather",
        get_weather,
        {
            "name": "get_weather",
            "description": "Şehir için hava durumu bilgisi getirir",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "Şehir adı"}
                },
                "required": ["city"]
            }
        }
    )
    
    # Test function calls
    test_calls = [
        ("calculator", {"expression": "15 * 3 + 7"}),
        ("get_time", {}),
        ("get_weather", {"city": "Istanbul"}),
        ("wikipedia_search", {"query": "Python", "lang": "tr"})
    ]
    
    for func_name, params in test_calls:
        print(f"\n🔧 Calling: {func_name}({params})")
        result = await registry.call_function(func_name, params)
        print(f"📤 Result: {result}")

if __name__ == "__main__":
    import asyncio
    
    # Logging setup
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run demos
    asyncio.run(demo_intelligent_chatbot())
    asyncio.run(demo_custom_functions())