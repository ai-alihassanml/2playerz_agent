#!/usr/bin/env python3
"""
Simple test client for the 2playz RAG Agent API
Tests both REST API and WebSocket functionality
"""

import asyncio
import aiohttp
import json
import websockets
from typing import Dict, Any

API_BASE_URL = "http://localhost:8000"
WS_BASE_URL = "ws://localhost:8000"

class RAGAgentClient:
    def __init__(self):
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def chat(self, message: str, thread_id: str = None) -> Dict[str, Any]:
        """Test regular chat endpoint"""
        url = f"{API_BASE_URL}/chat"
        data = {"message": message, "thread_id": thread_id}
        
        async with self.session.post(url, json=data) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"API Error {response.status}: {error_text}")
    
    async def chat_stream(self, message: str, thread_id: str = None):
        """Test streaming chat endpoint"""
        url = f"{API_BASE_URL}/chat/stream"
        data = {"message": message, "thread_id": thread_id}
        
        async with self.session.post(url, json=data) as response:
            if response.status == 200:
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith('data: '):
                        try:
                            data_content = json.loads(line[6:])
                            yield data_content
                        except json.JSONDecodeError:
                            continue
            else:
                error_text = await response.text()
                raise Exception(f"API Error {response.status}: {error_text}")
    
    async def websocket_chat(self, message: str, thread_id: str = None):
        """Test WebSocket chat"""
        uri = f"{WS_BASE_URL}/ws"
        
        async with websockets.connect(uri) as websocket:
            # Send chat message
            await websocket.send(json.dumps({
                "type": "chat",
                "content": message,
                "thread_id": thread_id
            }))
            
            # Receive responses
            async for message in websocket:
                data = json.loads(message)
                yield data

async def test_regular_chat():
    """Test regular chat endpoint"""
    print("=== Testing Regular Chat ===")
    
    async with RAGAgentClient() as client:
        try:
            response = await client.chat("Tell me about Death Stranding 2")
            print(f"Response: {response['response']}")
            print(f"Sources: {len(response['sources'])} documents")
            print(f"Language: {response['language_detected']}")
            print(f"Processing Time: {response['processing_time']:.2f}s")
            print(f"Router Decision: {response['router_decision']}")
            
            if response['sources']:
                print("\nSources:")
                for i, source in enumerate(response['sources'], 1):
                    print(f"{i}. {source['title']} (Slug: {source['slug']})")
            
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n" + "="*50 + "\n")

async def test_streaming_chat():
    """Test streaming chat endpoint"""
    print("=== Testing Streaming Chat ===")
    
    async with RAGAgentClient() as client:
        try:
            print("Question: What are the best RPG games?")
            print("Response: ", end="", flush=True)
            
            async for chunk in client.chat_stream("What are the best RPG games?"):
                if chunk['type'] == 'response':
                    print(chunk['content'], end="", flush=True)
                elif chunk['type'] == 'sources':
                    print(f"\n\nSources: {len(chunk['content'])} documents found")
                elif chunk['type'] == 'metadata':
                    print(f"\n\nMetadata: {chunk['content']}")
                elif chunk['type'] == 'blocked':
                    print(f"\nRequest blocked: {chunk['content']}")
                elif chunk['type'] == 'error':
                    print(f"\nError: {chunk['content']}")
                elif chunk['type'] == 'end':
                    print("\n[Stream completed]")
            
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n" + "="*50 + "\n")

async def test_websocket_chat():
    """Test WebSocket chat"""
    print("=== Testing WebSocket Chat ===")
    
    try:
        async with RAGAgentClient() as client:
            print("Question: Hello, what can you help me with?")
            
            async for message in client.websocket_chat("Hello, what can you help me with?"):
                if message['type'] == 'connection_established':
                    print(f"Connected: {message['content']}")
                elif message['type'] == 'processing':
                    print(f"Status: {message['content']}")
                elif message['type'] == 'complete_response':
                    print(f"Response: {message['content']}")
                elif message['type'] == 'final_result':
                    result = message['content']
                    print(f"Final Response: {result['response']}")
                    print(f"Sources: {len(result['sources'])} documents")
                    print(f"Processing Time: {result['processing_time']:.2f}s")
                elif message['type'] == 'error':
                    print(f"Error: {message['content']}")
                elif message['type'] == 'blocked':
                    print(f"Blocked: {message['content']}")
            
    except Exception as e:
        print(f"WebSocket Error: {e}")
    
    print("\n" + "="*50 + "\n")

async def test_api_status():
    """Test API status endpoints"""
    print("=== Testing API Status ===")
    
    async with RAGAgentClient() as client:
        try:
            # Test status endpoint
            async with client.session.get(f"{API_BASE_URL}/status") as response:
                if response.status == 200:
                    status = await response.json()
                    print(f"Status: {status['status']}")
                    print(f"Version: {status['version']}")
                    print(f"Active Connections: {status['active_connections']}")
                    print(f"Active Threads: {status['active_threads']}")
            
            # Test health endpoint
            async with client.session.get(f"{API_BASE_URL}/health") as response:
                if response.status == 200:
                    health = await response.json()
                    print(f"Health: {health['status']}")
                    print(f"Uptime: {health['uptime']}")
            
            # Test threads endpoint
            async with client.session.get(f"{API_BASE_URL}/threads") as response:
                if response.status == 200:
                    threads = await response.json()
                    print(f"Total Threads: {threads['total_threads']}")
            
        except Exception as e:
            print(f"Status Error: {e}")
    
    print("\n" + "="*50 + "\n")

async def main():
    """Run all tests"""
    print("2playz RAG Agent API Test Client")
    print("Make sure the API server is running on http://localhost:8000")
    print("="*60)
    
    try:
        await test_api_status()
        await test_regular_chat()
        await test_streaming_chat()
        await test_websocket_chat()
        
        print("All tests completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        print("Make sure the API server is running: python api.py")

if __name__ == "__main__":
    # Install dependencies: pip install aiohttp websockets
    try:
        import aiohttp
        import websockets
        asyncio.run(main())
    except ImportError as e:
        print(f"Please install required dependencies: pip install aiohttp websockets")
        print(f"Missing: {e}")
