import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import sqlite3
import os

# MCP Types and Structures
@dataclass
class Tool:
    """MCP Tool definition"""
    name: str
    description: str
    inputSchema: Dict[str, Any]

@dataclass
class Resource:
    """MCP Resource definition"""
    uri: str
    name: str
    description: str
    mimeType: str

class MCPServer:
    """
    Basic MCP Server Implementation
    Claude AI ile iletişim kurmak için MCP protokolünü uygular
    """
    
    def __init__(self, name: str = "AI-Lab Basic MCP Server"):
        self.name = name
        self.version = "1.0.0"
        self.tools: Dict[str, Tool] = {}
        self.resources: Dict[str, Resource] = {}
        self.db_path = "mcp_data.db"
        self.setup_database()
        self.register_default_tools()
        
        # Logging setup
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def setup_database(self):
        """SQLite veritabanını ayarla"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Notes tablosu
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tasks tablosu
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT,
                status TEXT DEFAULT 'pending',
                priority TEXT DEFAULT 'medium',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                due_date TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def register_default_tools(self):
        """Varsayılan araçları kaydet"""
        
        # Note management tools
        self.register_tool(Tool(
            name="create_note",
            description="Yeni bir not oluştur",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Not başlığı"},
                    "content": {"type": "string", "description": "Not içeriği"}
                },
                "required": ["title", "content"]
            }
        ))
        
        self.register_tool(Tool(
            name="list_notes",
            description="Tüm notları listele",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Maksimum not sayısı", "default": 10}
                }
            }
        ))
        
        self.register_tool(Tool(
            name="search_notes",
            description="Notlarda arama yap",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Arama terimi"}
                },
                "required": ["query"]
            }
        ))
        
        # Task management tools
        self.register_tool(Tool(
            name="create_task",
            description="Yeni bir görev oluştur",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Görev başlığı"},
                    "description": {"type": "string", "description": "Görev açıklaması"},
                    "priority": {"type": "string", "enum": ["low", "medium", "high"], "default": "medium"},
                    "due_date": {"type": "string", "format": "date-time", "description": "Son tarih"}
                },
                "required": ["title"]
            }
        ))
        
        self.register_tool(Tool(
            name="list_tasks",
            description="Görevleri listele",
            inputSchema={
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["pending", "completed", "cancelled"]},
                    "priority": {"type": "string", "enum": ["low", "medium", "high"]}
                }
            }
        ))
        
        self.register_tool(Tool(
            name="update_task_status",
            description="Görev durumunu güncelle",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "integer", "description": "Görev ID"},
                    "status": {"type": "string", "enum": ["pending", "completed", "cancelled"]}
                },
                "required": ["task_id", "status"]
            }
        ))
        
        # Utility tools
        self.register_tool(Tool(
            name="get_current_time",
            description="Şu anki zamanı al",
            inputSchema={"type": "object", "properties": {}}
        ))
        
        self.register_tool(Tool(
            name="calculate",
            description="Matematiksel hesaplama yap",
            inputSchema={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Matematik ifadesi (örn: 2+2*3)"}
                },
                "required": ["expression"]
            }
        ))
    
    def register_tool(self, tool: Tool):
        """Yeni bir araç kaydet"""
        self.tools[tool.name] = tool
        self.logger.info(f"Tool registered: {tool.name}")
    
    def register_resource(self, resource: Resource):
        """Yeni bir kaynak kaydet"""
        self.resources[resource.uri] = resource
        self.logger.info(f"Resource registered: {resource.uri}")
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """MCP request'lerini işle"""
        method = request.get("method")
        params = request.get("params", {})
        
        try:
            if method == "initialize":
                return await self.handle_initialize(params)
            elif method == "tools/list":
                return await self.handle_list_tools()
            elif method == "tools/call":
                return await self.handle_call_tool(params)
            elif method == "resources/list":
                return await self.handle_list_resources()
            elif method == "resources/read":
                return await self.handle_read_resource(params)
            else:
                return {
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}"
                    }
                }
        except Exception as e:
            self.logger.error(f"Error handling request: {e}")
            return {
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            }
    
    async def handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize handler"""
        return {
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {"listChanged": True},
                    "resources": {"subscribe": True, "listChanged": True}
                },
                "serverInfo": {
                    "name": self.name,
                    "version": self.version
                }
            }
        }
    
    async def handle_list_tools(self) -> Dict[str, Any]:
        """Araçları listele"""
        tools = []
        for tool in self.tools.values():
            tools.append({
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.inputSchema
            })
        
        return {"result": {"tools": tools}}
    
    async def handle_call_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Araç çağrısını işle"""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if tool_name not in self.tools:
            return {
                "error": {
                    "code": -32602,
                    "message": f"Unknown tool: {tool_name}"
                }
            }
        
        # Tool'u çalıştır
        result = await self.execute_tool(tool_name, arguments)
        return {"result": {"content": [{"type": "text", "text": result}]}}
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Araç mantığını çalıştır"""
        
        if tool_name == "create_note":
            return await self.create_note(arguments["title"], arguments["content"])
        
        elif tool_name == "list_notes":
            limit = arguments.get("limit", 10)
            return await self.list_notes(limit)
        
        elif tool_name == "search_notes":
            return await self.search_notes(arguments["query"])
        
        elif tool_name == "create_task":
            return await self.create_task(
                arguments["title"],
                arguments.get("description"),
                arguments.get("priority", "medium"),
                arguments.get("due_date")
            )
        
        elif tool_name == "list_tasks":
            return await self.list_tasks(
                arguments.get("status"),
                arguments.get("priority")
            )
        
        elif tool_name == "update_task_status":
            return await self.update_task_status(
                arguments["task_id"],
                arguments["status"]
            )
        
        elif tool_name == "get_current_time":
            return await self.get_current_time()
        
        elif tool_name == "calculate":
            return await self.calculate(arguments["expression"])
        
        else:
            return f"Tool {tool_name} not implemented"
    
    # Tool implementations
    async def create_note(self, title: str, content: str) -> str:
        """Not oluştur"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO notes (title, content) VALUES (?, ?)",
            (title, content)
        )
        
        note_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return f"Not başarıyla oluşturuldu! ID: {note_id}, Başlık: '{title}'"
    
    async def list_notes(self, limit: int = 10) -> str:
        """Notları listele"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT id, title, content, created_at FROM notes ORDER BY created_at DESC LIMIT ?",
            (limit,)
        )
        
        notes = cursor.fetchall()
        conn.close()
        
        if not notes:
            return "Henüz not bulunmuyor."
        
        result = f"Son {len(notes)} not:\n\n"
        for note_id, title, content, created_at in notes:
            result += f"ID: {note_id}\n"
            result += f"Başlık: {title}\n"
            result += f"İçerik: {content[:100]}{'...' if len(content) > 100 else ''}\n"
            result += f"Oluşturulma: {created_at}\n"
            result += "-" * 50 + "\n"
        
        return result
    
    async def search_notes(self, query: str) -> str:
        """Notlarda arama"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT id, title, content FROM notes WHERE title LIKE ? OR content LIKE ?",
            (f"%{query}%", f"%{query}%")
        )
        
        notes = cursor.fetchall()
        conn.close()
        
        if not notes:
            return f"'{query}' için sonuç bulunamadı."
        
        result = f"'{query}' için {len(notes)} sonuç bulundu:\n\n"
        for note_id, title, content in notes:
            result += f"ID: {note_id} - {title}\n"
            result += f"{content[:150]}{'...' if len(content) > 150 else ''}\n\n"
        
        return result
    
    async def create_task(self, title: str, description: str = None, 
                         priority: str = "medium", due_date: str = None) -> str:
        """Görev oluştur"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO tasks (title, description, priority, due_date) VALUES (?, ?, ?, ?)",
            (title, description, priority, due_date)
        )
        
        task_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return f"Görev başarıyla oluşturuldu! ID: {task_id}, Başlık: '{title}', Öncelik: {priority}"
    
    async def list_tasks(self, status: str = None, priority: str = None) -> str:
        """Görevleri listele"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT id, title, description, status, priority, created_at, due_date FROM tasks WHERE 1=1"
        params = []
        
        if status:
            query += " AND status = ?"
            params.append(status)
        
        if priority:
            query += " AND priority = ?"
            params.append(priority)
        
        query += " ORDER BY created_at DESC"
        
        cursor.execute(query, params)
        tasks = cursor.fetchall()
        conn.close()
        
        if not tasks:
            return "Görev bulunmuyor."
        
        result = f"{len(tasks)} görev bulundu:\n\n"
        for task_id, title, desc, status, priority, created, due in tasks:
            result += f"ID: {task_id} | {status.upper()} | {priority.upper()}\n"
            result += f"Başlık: {title}\n"
            if desc:
                result += f"Açıklama: {desc}\n"
            if due:
                result += f"Son Tarih: {due}\n"
            result += f"Oluşturulma: {created}\n"
            result += "-" * 50 + "\n"
        
        return result
    
    async def update_task_status(self, task_id: int, status: str) -> str:
        """Görev durumunu güncelle"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "UPDATE tasks SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (status, task_id)
        )
        
        if cursor.rowcount == 0:
            conn.close()
            return f"ID {task_id} ile görev bulunamadı."
        
        conn.commit()
        conn.close()
        
        return f"Görev {task_id} durumu '{status}' olarak güncellendi."
    
    async def get_current_time(self) -> str:
        """Şu anki zamanı döndür"""
        now = datetime.now()
        return f"Şu anki zaman: {now.strftime('%Y-%m-%d %H:%M:%S')}"
    
    async def calculate(self, expression: str) -> str:
        """Güvenli matematiksel hesaplama"""
        try:
            # Güvenlik için sadece temel matematik operasyonlarına izin ver
            allowed_chars = set("0123456789+-*/.() ")
            if not all(c in allowed_chars for c in expression):
                return "Hata: Sadece temel matematik operasyonlarına (+, -, *, /, parantez) izin verilir."
            
            # eval kullanımını güvenli hale getir
            result = eval(expression, {"__builtins__": {}}, {})
            return f"{expression} = {result}"
        
        except Exception as e:
            return f"Hesaplama hatası: {str(e)}"
    
    async def handle_list_resources(self) -> Dict[str, Any]:
        """Kaynakları listele"""
        resources = []
        for resource in self.resources.values():
            resources.append({
                "uri": resource.uri,
                "name": resource.name,
                "description": resource.description,
                "mimeType": resource.mimeType
            })
        
        return {"result": {"resources": resources}}
    
    async def handle_read_resource(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Kaynak okuma"""
        uri = params.get("uri")
        
        if uri not in self.resources:
            return {
                "error": {
                    "code": -32602,
                    "message": f"Resource not found: {uri}"
                }
            }
        
        # Kaynak içeriğini oku (örnek)
        content = f"Content of resource: {uri}"
        
        return {
            "result": {
                "contents": [{
                    "uri": uri,
                    "mimeType": self.resources[uri].mimeType,
                    "text": content
                }]
            }
        }


# Server runner
async def run_server(host: str = "localhost", port: int = 8080):
    """MCP sunucusunu başlat"""
    server = MCPServer()
    
    print(f"🚀 MCP Server başlatılıyor...")
    print(f"📍 Host: {host}:{port}")
    print(f"🛠️  Araçlar: {len(server.tools)}")
    print(f"📊 Kaynaklar: {len(server.resources)}")
    print("-" * 50)
    
    # Basit HTTP server simülasyonu
    # Gerçek implementasyonda WebSocket veya HTTP server kullanılır
    while True:
        print("⏳ Claude bağlantısı bekleniyor...")
        await asyncio.sleep(5)
        
        # Demo: Örnek request işleme
        demo_request = {
            "method": "tools/list",
            "params": {}
        }
        
        response = await server.handle_request(demo_request)
        print(f"📤 Demo response: {len(response.get('result', {}).get('tools', []))} tools available")


if __name__ == "__main__":
    # Server'ı başlat
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        print("\n👋 Server kapatılıyor...")
