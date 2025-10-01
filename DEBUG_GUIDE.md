# Debug Guide for 2playz RAG Agent WebSocket Streaming

## Issues Fixed

### ✅ Backend Issues Fixed
1. **WebSocket Streaming**: Added proper error handling in streaming functions
2. **Message Logging**: Added comprehensive logging to track message flow
3. **Thread Management**: Fixed thread creation and cleanup
4. **Source Display**: Ensured sources are properly sent via WebSocket

### ✅ Frontend Issues Fixed
1. **WebSocket Connection**: Added detailed logging for connection status
2. **Message Handling**: Enhanced message handling with debugging
3. **Error Handling**: Improved error handling and reconnection logic
4. **Source Display**: Fixed source display in ChatMessage component

## How to Debug

### 1. Check Backend Logs
When you start the backend, you should see:
```bash
cd RAG_agent
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Expected logs:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 2. Check Frontend Console
Open browser developer tools (F12) and check the Console tab. You should see:

**On Page Load:**
```
Connecting to WebSocket: ws://localhost:8000/ws
WebSocket connected to: ws://localhost:8000/ws
Received WebSocket message: {type: "connection_established", content: "Connected successfully...", metadata: {...}}
Connection established: Connected successfully. Client ID: [uuid]
```

**When Sending Message:**
```
Sending message: hi
Received WebSocket message: {type: "processing_started", content: "Processing your request...", metadata: {...}}
Processing started: Processing your request...
Received WebSocket message: {type: "chunk", content: "Hello", metadata: {...}}
Received chunk: Hello
Received WebSocket message: {type: "chunk", content: "! How", metadata: {...}}
Received chunk: ! How
...
Received WebSocket message: {type: "sources", content: [...], metadata: {...}}
Received sources: [...]
Received WebSocket message: {type: "complete", content: "Response completed", metadata: {...}}
Response complete
```

### 3. Check Backend Logs When Sending Message
You should see in the backend terminal:
```
INFO:     Client [uuid] connected. Total connections: 1
INFO:     Received chat message from [uuid]: hi
INFO:     Starting language detection...
INFO:     Language detected: en (confidence: 0.8)
INFO:     Running guardrail check...
INFO:     Guardrail check completed
INFO:     Starting agent processing...
INFO:     Sending message to [uuid]: processing_started
INFO:     ---NODE: ROUTING QUERY---
INFO:     ---Router Decision: llm---
INFO:     ---NODE: GENERATING ANSWER (NO RETRIEVAL)---
INFO:     Sending message to [uuid]: chunk
INFO:     Sending message to [uuid]: chunk
...
INFO:     Sending message to [uuid]: sources
INFO:     Sending message to [uuid]: complete
```

## Common Issues and Solutions

### Issue 1: WebSocket Connection Failed
**Symptoms:**
- Frontend shows "DISCONNECTED" status
- Console shows "WebSocket error"

**Solutions:**
1. Make sure backend is running on port 8000
2. Check if CORS is properly configured
3. Verify WebSocket URL is correct (ws://localhost:8000/ws)

### Issue 2: No Response After Sending Message
**Symptoms:**
- Message is sent but no response received
- Console shows "Sending message: [message]" but no further logs

**Solutions:**
1. Check backend logs for errors
2. Verify RAG agent is properly initialized
3. Check if FAISS index exists
4. Verify OpenAI API key is set

### Issue 3: Sources Not Displayed
**Symptoms:**
- Response is received but no sources shown
- Console shows sources received but UI doesn't display them

**Solutions:**
1. Check if sources array is not empty
2. Verify ChatMessage component is receiving sources prop
3. Check if showSources state is working

### Issue 4: Streaming Not Working
**Symptoms:**
- Response appears all at once instead of streaming
- No "chunk" messages in console

**Solutions:**
1. Check if streaming graph is being used
2. Verify WebSocket manager is passed correctly
3. Check if LLM streaming is working

## Testing Steps

### 1. Test WebSocket Connection
```bash
cd RAG_agent
python test_websocket.py
```

Expected output:
```
 Connected to WebSocket
 Received: connection_established - Connected successfully. Client ID: [uuid]
 Sending: {'type': 'chat', 'content': 'Hello, what is Death Stranding?', 'language': 'auto'}
 Listening for responses...
 Chunk: Hello! Death Stranding is...
 Sources (2):
  1. Death Stranding Review (slug: death-stranding-review)
  2. Gaming News (slug: gaming-news)
 Complete: Response completed
 Test completed successfully!
```

### 2. Test Frontend
1. Open browser to frontend URL
2. Check console for connection logs
3. Send a test message
4. Verify streaming response
5. Check sources display

## Environment Variables Check

Make sure these are set:
```bash
# Backend
OPENAI_API_KEY=your_openai_key(add your api key here )
HAGGINGFACEHUB_API_TOKEN=your_huggingface_token

# Frontend (optional)
VITE_API_URL=http://localhost:8000
```

## File Structure Check

Make sure these files exist:
```
RAG_agent/
├── api.py (updated with WebSocket streaming)
├── rag_agent.py (updated with streaming functions)
├── faiss_index/ (FAISS index directory)
│   ├── index.faiss
│   └── index.pkl
└── test_websocket.py (for testing)

RAG_Agent_Frontend/
├── src/
│   ├── pages/Index.tsx (updated with WebSocket)
│   └── components/ChatMessage.tsx (updated with sources)
```

## Still Having Issues?

1. **Check all logs** - Both frontend console and backend terminal
2. **Verify environment** - All required environment variables set
3. **Test WebSocket** - Use the test script to verify backend
4. **Check network** - Make sure no firewall blocking WebSocket
5. **Restart services** - Sometimes a restart fixes connection issues

The system should now work with:
- ✅ Real-time WebSocket streaming
- ✅ Proper thread management
- ✅ Source display with titles and slugs
- ✅ Error handling and reconnection
- ✅ ChatGPT-like streaming experience
