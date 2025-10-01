#!/usr/bin/env python3
"""
Simple WebSocket test client for the RAG Agent API
"""
import asyncio
import json
import websockets
import sys

async def test_websocket():
    uri = "ws://localhost:8000/ws"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to WebSocket")
            
            # Wait for connection established message
            message = await websocket.recv()
            data = json.loads(message)
            print(f"📨 Received: {data['type']} - {data['content']}")
            
            # Send a test message
            test_message = {
                "type": "chat",
                "content": "Hello, what is Death Stranding?",
                "language": "auto"
            }
            
            print(f"📤 Sending: {test_message}")
            await websocket.send(json.dumps(test_message))
            
            # Listen for responses
            print("📥 Listening for responses...")
            response_count = 0
            
            while True:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                    data = json.loads(message)
                    
                    if data['type'] == 'chunk':
                        print(f"📝 Chunk: {data['content']}", end='', flush=True)
                    elif data['type'] == 'sources':
                        sources = data['content']
                        if sources:
                            print(f"\n📚 Sources ({len(sources)}):")
                            for i, source in enumerate(sources, 1):
                                print(f"  {i}. {source['title']} (slug: {source['slug']})")
                        else:
                            print("\n📚 No sources found")
                    elif data['type'] == 'complete':
                        print(f"\n✅ Complete: {data['content']}")
                        break
                    elif data['type'] == 'error':
                        print(f"\n❌ Error: {data['content']}")
                        break
                    else:
                        print(f"\n📨 {data['type']}: {data['content']}")
                        
                    response_count += 1
                    if response_count > 50:  # Safety limit
                        print("\n⚠️ Too many responses, breaking...")
                        break
                        
                except asyncio.TimeoutError:
                    print("\n⏰ Timeout waiting for response")
                    break
                    
    except websockets.exceptions.ConnectionRefused:
        print("❌ Connection refused. Make sure the server is running on localhost:8000")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("🧪 Testing WebSocket connection to RAG Agent API...")
    success = asyncio.run(test_websocket())
    if success:
        print("✅ Test completed successfully!")
    else:
        print("❌ Test failed!")
        sys.exit(1)
