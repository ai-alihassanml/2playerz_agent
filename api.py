import json
import asyncio
import logging
from datetime import datetime
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag_agent import run_agent_once

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="2playz RAG Agent - Minimal", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message to send to the agent")
    thread_id: Optional[str] = Field(None, description="Thread ID for conversation continuity")
    language: Optional[str] = Field(None, description="Preferred language (auto-detect if not provided)")


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming endpoint that runs the agent once via `run_agent_once` and streams SSE events back:
      - start
      - response
      - sources
      - metadata
      - end
    """

    async def generate() -> AsyncGenerator[str, None]:
        yield f"data: {json.dumps({'type': 'start', 'content': 'Processing your request...'})}\n\n"

        try:
            result = await run_agent_once(
                query=request.message,
                thread_id=request.thread_id,
                conversation_history=None,
                config=None,
            )

            if result.get("blocked"):
                yield f"data: {json.dumps({'type': 'blocked', 'content': result['assistant_text']})}\n\n"
                yield f"data: {json.dumps({'type': 'end', 'content': 'blocked'})}\n\n"
                return

            # send response
            yield f"data: {json.dumps({'type': 'response', 'content': result.get('assistant_text', '')})}\n\n"

            # send sources
            sources = result.get('sources', [])
            if sources:
                yield f"data: {json.dumps({'type': 'sources', 'content': sources})}\n\n"

            metadata = {
                'detected_language': result.get('detected_language'),
                'blocked': result.get('blocked', False),
                'thread_id': request.thread_id,
            }
            yield f"data: {json.dumps({'type': 'metadata', 'content': metadata})}\n\n"

            yield f"data: {json.dumps({'type': 'end', 'content': 'Stream completed'})}\n\n"

        except Exception as e:
            logger.exception(f"Error running agent: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': 'Internal server error'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
import os
import uuid
import json
import asyncio
from datetime import datetime
from typing import Dict, Any, AsyncGenerator, Optional
import logging

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse
from starlette.websockets import WebSocketState
import asyncio
from asyncio import TimeoutError as AsyncioTimeoutError

# Import your agent utilities
from rag_agent import (
    graph, create_streaming_graph, llm_input_guardrails, llm_detect_language_and_intent,
    translate_text_async, GUARDRAIL_SENTINEL, AgentState
)
from langchain_core.messages import HumanMessage, AIMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="2playz RAG Agent API",
    description="Production-ready API for 2playz gaming website RAG agent with WebSocket support",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


# Application lifecycle events
@app.on_event("startup")
async def on_startup():
    logger.info("2playz RAG Agent starting up...")


@app.on_event("shutdown")
async def on_shutdown():
    logger.info("2playz RAG Agent shutting down...")
    # Attempt to close all active websockets gracefully
    for client_id, ws in list(manager.active_connections.items()):
        try:
            await ws.close(code=1001)
        except Exception:
            pass
    manager.active_connections.clear()


# Global exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTPException: {exc.detail}")
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

# Production-ready CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure specific origins for production
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, client_id: str):
        # Accept with a small timeout to avoid blocking forever
        try:
            await asyncio.wait_for(websocket.accept(), timeout=5)
        except AsyncioTimeoutError:
            logger.warning(f"Timed out accepting websocket for {client_id}")
            return
        async with self._lock:
            self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, client_id: str):
        # Synchronous removal (called from sync contexts)
        if client_id in self.active_connections:
            try:
                del self.active_connections[client_id]
            except Exception:
                pass
        logger.info(f"Client {client_id} disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: dict, client_id: str):
        websocket = self.active_connections.get(client_id)
        if not websocket:
            logger.warning(f"Client {client_id} not found in active connections")
            return

        # Don't attempt to send if socket is closed
        try:
            if websocket.client_state != WebSocketState.CONNECTED:
                logger.info(f"WebSocket for {client_id} not connected (state={websocket.client_state}), removing")
                self.disconnect(client_id)
                return
        except Exception:
            # Some ASGI servers may not expose client_state; fall back to send attempt
            pass

        try:
            logger.debug(f"Sending message to {client_id}: {message.get('type')}")
            # limit payload size to avoid large sends (safeguard)
            payload = json.dumps(message)
            if len(payload) > 200_000:  # ~200KB limit
                logger.warning(f"Payload too large for {client_id}, truncating message")
                message = {"type": "error", "content": "Payload too large to send"}
                payload = json.dumps(message)

            await asyncio.wait_for(websocket.send_text(payload), timeout=10)
        except (AsyncioTimeoutError, ConnectionResetError) as e:
            logger.error(f"Timeout/connection error sending message to {client_id}: {e}")
            self.disconnect(client_id)
        except Exception as e:
            logger.exception(f"Unexpected error sending message to {client_id}: {e}")
            self.disconnect(client_id)

    async def broadcast(self, message: dict):
        disconnected_clients = []
        for client_id, connection in self.active_connections.items():
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error broadcasting to {client_id}: {e}")
                disconnected_clients.append(client_id)
        
        for client_id in disconnected_clients:
            self.disconnect(client_id)

manager = ConnectionManager()

# Request/Response Models
class ChatRequest(BaseModel):
    message: str = Field(..., description="User message to send to the agent")
    thread_id: Optional[str] = Field(None, description="Thread ID for conversation continuity")
    language: Optional[str] = Field(None, description="Preferred language (auto-detect if not provided)")
    client_id: Optional[str] = Field(None, description="Client ID for WebSocket connection")

class ChatResponse(BaseModel):
    response: str = Field(..., description="Agent response")
    thread_id: str = Field(..., description="Thread ID for conversation continuity")
    sources: list = Field(default_factory=list, description="Retrieved document sources with slugs")
    language_detected: Optional[str] = Field(None, description="Detected language of input")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    router_decision: Optional[str] = Field(None, description="Whether RAG or LLM was used")
    blocked: bool = Field(False, description="Whether request was blocked by guardrails")

class WebSocketMessage(BaseModel):
    type: str = Field(..., description="Message type")
    content: Any = Field(..., description="Message content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")



# Global thread management
active_threads: Dict[str, Dict[str, Any]] = {}
client_threads: Dict[str, str] = {}  # Maps client_id to thread_id

def get_or_create_thread(thread_id: Optional[str] = None, client_id: Optional[str] = None) -> str:
    """Get existing thread or create new one"""
    # If client_id provided, check if they have an existing thread
    if client_id and client_id in client_threads:
        existing_thread = client_threads[client_id]
        if existing_thread in active_threads:
            active_threads[existing_thread]["last_activity"] = datetime.now()
            return existing_thread
        else:
            # Clean up stale reference
            del client_threads[client_id]
    
    # Use provided thread_id if valid
    if thread_id and thread_id in active_threads:
        active_threads[thread_id]["last_activity"] = datetime.now()
        if client_id:
            client_threads[client_id] = thread_id
        return thread_id
    
    # Create new thread
    new_thread_id = str(uuid.uuid4())
    active_threads[new_thread_id] = {
        "created_at": datetime.now(),
        "message_count": 0,
        "last_activity": datetime.now()
    }
    
    if client_id:
        client_threads[client_id] = new_thread_id
    
    return new_thread_id

def cleanup_client_thread(client_id: str):
    """Clean up thread when client disconnects"""
    if client_id in client_threads:
        thread_id = client_threads[client_id]
        if thread_id in active_threads:
            del active_threads[thread_id]
        del client_threads[client_id]


async def process_agent_query(
    query: str, 
    thread_id: str, 
    language: Optional[str] = None,
    websocket_manager: Optional[ConnectionManager] = None,
    client_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process query through the main RAG agent function
    This integrates with the main agent logic from rag_agent.py
    """
    config = {"configurable": {"thread_id": thread_id}}
    start_time = datetime.now()
    
    try:
    # 1) Language detection
        logger.info("Starting language detection...")
        try:
            detect_info = llm_detect_language_and_intent(query)
            detected_lang = detect_info.get("language", "en")
            is_question = detect_info.get("is_question", False)
            confidence = detect_info.get("confidence", 0.0)
            logger.info(f"Language detected: {detected_lang} (confidence: {confidence})")
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            detected_lang = language or "en"
            is_question = query.strip().endswith("?")

        # 2) Translate to English if needed
        if detected_lang and detected_lang != "en":
            logger.info(f"Translating from {detected_lang} to English...")
            try:
                english_query = await translate_text_async(query, src=detected_lang, dest="en")
                logger.info("Translation completed")
            except Exception as e:
                logger.error(f"Translation to English failed: {e}")
                english_query = query
        else:
            english_query = query

        # 3) Guardrail check
        logger.info("Running guardrail check...")
        try:
            sanitized = llm_input_guardrails(english_query)
            logger.info("Guardrail check completed")
        except Exception as e:
            logger.error(f"Moderation/guardrail check failed: {e}")
            sanitized = english_query

        # 4) Handle guardrail blocking
        if isinstance(sanitized, str) and sanitized.strip() == GUARDRAIL_SENTINEL:
            logger.warning("Request blocked by guardrails")
            if detected_lang and detected_lang != "en":
                try:
                    translated_guardrail = await translate_text_async(GUARDRAIL_SENTINEL, src="en", dest=detected_lang)
                except Exception as e:
                    logger.error(f"Translation of guardrail failed: {e}")
                    translated_guardrail = GUARDRAIL_SENTINEL
            else:
                translated_guardrail = GUARDRAIL_SENTINEL
                
            # Send WebSocket notification if connected
            if websocket_manager and client_id:
                await websocket_manager.send_personal_message({
                    "type": "blocked",
                    "content": translated_guardrail,
                    "metadata": {"language_detected": detected_lang}
                }, client_id)
                
            return {
                "response": translated_guardrail,
                "blocked": True,
                "language_detected": detected_lang,
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "sources": [],
                "router_decision": None
            }

        # 5) Build initial state and run graph (this is the main agent processing)
        logger.info("Starting agent processing...")
        initial_state = {
            "messages": [HumanMessage(content=sanitized)],
            "query": sanitized,
            "retrieved_docs": []
        }

        # Send WebSocket notification that processing started
        if websocket_manager and client_id:
            await websocket_manager.send_personal_message({
                "type": "processing_started",
                "content": "Processing your request...",
                "metadata": {"language_detected": detected_lang}
            }, client_id)

        # Create streaming graph if WebSocket is available
        if websocket_manager and client_id:
            streaming_graph = create_streaming_graph(websocket_manager, client_id)
            final_state = await streaming_graph.ainvoke(initial_state, config=config)
        else:
            final_state = await graph.ainvoke(initial_state, config=config)
        
        english_answer = final_state["messages"][-1].content

        # 6) Translate back to user language if needed
        if detected_lang and detected_lang != "en":
            logger.info(f"Translating response back to {detected_lang}...")
            try:
                final_response = await translate_text_async(english_answer, src="en", dest=detected_lang)
                logger.info("Translation back completed")
            except Exception as e:
                logger.error(f"Translation back to user language failed: {e}")
                final_response = english_answer
        else:
            final_response = english_answer

        # 7) Prepare sources with slugs
        sources = []
        if final_state.get("retrieved_docs"):
            logger.info(f"Processing {len(final_state['retrieved_docs'])} retrieved documents...")
            for doc in final_state["retrieved_docs"]:
                source_info = {
                    "title": doc.metadata.get("title", "N/A"),
                    "slug": doc.metadata.get("slug", "N/A"),
                    "excerpt": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "source": doc.metadata.get("source", "N/A"),
                    "chunk_id": doc.metadata.get("chunk_id", "N/A"),
                    "full_content": doc.page_content  # Include full content for reference
                }
                sources.append(source_info)

        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Update thread info
        active_threads[thread_id]["message_count"] += 1
        active_threads[thread_id]["last_activity"] = datetime.now()
        
        # Determine router decision
        router_decision = "rag" if sources else "llm"
        
        logger.info(f"Processing completed in {processing_time:.2f}s")
        
        # Send final response via WebSocket if connected
        if websocket_manager and client_id:
            await websocket_manager.send_personal_message({
                "type": "complete_response",
                "content": final_response,
                "metadata": {
                    "sources": sources,
                    "processing_time": processing_time,
                    "language_detected": detected_lang,
                    "router_decision": router_decision,
                    "thread_id": thread_id
                }
            }, client_id)
        
        return {
            "response": final_response,
            "sources": sources,
            "language_detected": detected_lang,
            "processing_time": processing_time,
            "blocked": False,
            "router_decision": router_decision,
            "thread_id": thread_id
        }

    except Exception as e:
        logger.error(f"Error in agent processing: {e}")
        # Send error via WebSocket if connected
        if websocket_manager and client_id:
            await websocket_manager.send_personal_message({
                "type": "error",
                "content": f"An error occurred: {str(e)}",
                "metadata": {"error_type": "processing_error"}
            }, client_id)
        
        # Return structured error instead of raising raw HTTPException so callers can decide
        raise RuntimeError(f"Agent execution failed: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "2playz RAG Agent API",
        "version": "2.0.0",
        "features": ["REST API", "WebSocket", "Real-time streaming", "Multi-language support"],
        "docs": "/docs",
        "websocket": "/ws"
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Complete chat endpoint with full agent processing (non-streaming)"""
    thread_id = get_or_create_thread(request.thread_id)

    try:
        result = await process_agent_query(
            query=request.message,
            thread_id=thread_id,
            language=request.language
        )

        if result.get("blocked"):
            raise HTTPException(status_code=403, detail=result["response"])

        return ChatResponse(
            response=result["response"],
            thread_id=result["thread_id"],
            sources=result.get("sources", []),
            language_detected=result.get("language_detected"),
            processing_time=result.get("processing_time"),
            router_decision=result.get("router_decision"),
            blocked=result.get("blocked", False)
        )

    except HTTPException:
        raise
    except RuntimeError as re:
        logger.error(f"Agent runtime error in /chat: {re}")
        raise HTTPException(status_code=500, detail=str(re))
    except Exception as e:
        logger.exception(f"Unhandled error in /chat: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/status")
async def get_status():
    """Get API and agent status"""
    return {
        "status": "running",
        "version": "2.0.0",
        "active_connections": len(manager.active_connections),
        "active_threads": len(active_threads),
        "timestamp": datetime.now().isoformat(),
        "features": ["REST API", "WebSocket", "Real-time streaming", "Multi-language support"]
    }

@app.get("/threads")
async def get_threads():
    """Get information about active threads"""
    return {
        "threads": [
            {
                "thread_id": thread_id,
                "created_at": info["created_at"].isoformat(),
                "last_activity": info["last_activity"].isoformat(),
                "message_count": info["message_count"]
            }
            for thread_id, info in active_threads.items()
        ],
        "total_threads": len(active_threads)
    }


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Server-Sent Events streaming endpoint"""
    try:
        thread_id = get_or_create_thread(request.thread_id)

        async def generate() -> AsyncGenerator[str, None]:
            try:
                yield f"data: {json.dumps({'type': 'start', 'content': 'Processing your request...'})}\n\n"
                
                result = await process_agent_query(
                    query=request.message,
                    thread_id=thread_id,
                    language=request.language
                )
                
                if result.get("blocked"):
                    yield f"data: {json.dumps({'type': 'blocked', 'content': result['response']})}\n\n"
                    return
                
                # Send response content
                yield f"data: {json.dumps({'type': 'response', 'content': result['response']})}\n\n"
                
                # Send sources
                if result.get('sources'):
                    yield f"data: {json.dumps({'type': 'sources', 'content': result['sources']})}\n\n"
                
                # Send metadata
                metadata = {
                    'processing_time': result.get('processing_time', 0),
                    'language_detected': result.get('language_detected', 'en'),
                    'router_decision': result.get('router_decision', 'unknown'),
                    'thread_id': result['thread_id']
                }
                yield f"data: {json.dumps({'type': 'metadata', 'content': metadata})}\n\n"
                
                yield f"data: {json.dumps({'type': 'end', 'content': 'Stream completed'})}\n\n"

            except Exception as e:
                logger.exception(f"Streaming error: {e}")
                yield f"data: {json.dumps({'type': 'error', 'content': 'Internal streaming error'})}\n\n"

        return StreamingResponse(
            generate(), 
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*"
            }
        )

    except Exception as e:
        logger.error(f"Streaming endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Streaming error: {str(e)}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for client connections."""
    client_id = str(uuid.uuid4())
    ping_task = None
    MAX_MESSAGE_SIZE = 200_000  # 200KB per message
    PING_INTERVAL = 20  # seconds
    PING_TIMEOUT = 10

    try:
        # ✅ Accept and register client
        await manager.connect(websocket, client_id)
        thread_id = get_or_create_thread(client_id)

        # ✅ Send initial handshake event
        await manager.send_personal_message({
            "type": "connection_established",
            "content": f"Connected successfully. Client ID: {client_id}",
            "metadata": {
                "client_id": client_id,
                "thread_id": thread_id
            }
        }, client_id)

        # ✅ Start background ping task to keep connection alive and detect dead peers
        async def ping_loop():
            try:
                while True:
                    await asyncio.sleep(PING_INTERVAL)
                    try:
                        await manager.send_personal_message({
                            "type": "ping",
                            "content": "ping",
                            "metadata": {"timestamp": datetime.now().isoformat()}
                        }, client_id)
                    except Exception:
                        # send_personal_message handles disconnects
                        break
            except asyncio.CancelledError:
                return

        ping_task = asyncio.create_task(ping_loop())

        # ✅ Main message loop with safeguards
        while True:
            try:
                # enforce a receive timeout so a misbehaving client doesn't hang
                data = await asyncio.wait_for(websocket.receive_text(), timeout=300)
            except WebSocketDisconnect:
                logger.info(f"Client {client_id} disconnected.")
                break
            except AsyncioTimeoutError:
                logger.info(f"Receive timeout for {client_id}, closing connection")
                break
            except Exception as e:
                logger.exception(f"Receive error from {client_id}: {e}")
                break

            # size guard
            if len(data) > MAX_MESSAGE_SIZE:
                await manager.send_personal_message({
                    "type": "error",
                    "content": "Message too large",
                    "metadata": {"error_type": "message_too_large"}
                }, client_id)
                continue

            try:
                message_data = json.loads(data)
            except json.JSONDecodeError:
                await manager.send_personal_message({
                    "type": "error",
                    "content": "Invalid JSON format",
                    "metadata": {"error_type": "json_error"}
                }, client_id)
                continue

            message_type = message_data.get("type", "chat")
            content = message_data.get("content", "")
            language = message_data.get("language")

            # === Chat messages ===
            if message_type == "chat" and content:
                try:
                    result = await process_agent_query(
                        query=content,
                        thread_id=thread_id,
                        language=language,
                        websocket_manager=manager,
                        client_id=client_id
                    )

                    # ✅ Send retriever sources (slug, title, description, etc.)
                    if result.get("sources"):
                        await manager.send_personal_message({
                            "type": "sources",
                            "content": result["sources"],
                            "metadata": {
                                "timestamp": datetime.now().isoformat(),
                                "processing_time": result.get("processing_time"),
                                "language_detected": result.get("language_detected"),
                                "router_decision": result.get("router_decision"),
                                "thread_id": thread_id
                            }
                        }, client_id)

                    # ✅ Send completion event
                    await manager.send_personal_message({
                        "type": "complete",
                        "content": "Response completed",
                        "metadata": {
                            "timestamp": datetime.now().isoformat(),
                            "processing_time": result.get("processing_time"),
                            "language_detected": result.get("language_detected"),
                            "router_decision": result.get("router_decision"),
                            "thread_id": thread_id
                        }
                    }, client_id)

                except Exception as e:
                    logger.exception(f"Processing error for {client_id}: {e}")
                    await manager.send_personal_message({
                        "type": "error",
                        "content": "Message processing error",
                        "metadata": {"error_type": "processing_error"}
                    }, client_id)

            # === Ping/Pong heartbeat ===
            elif message_type == "ping":
                await manager.send_personal_message({
                    "type": "pong",
                    "content": "pong",
                    "metadata": {"timestamp": datetime.now().isoformat()}
                }, client_id)

            # === Force new thread (if client asks) ===
            elif message_type == "new_thread":
                thread_id = get_or_create_thread(client_id)
                await manager.send_personal_message({
                    "type": "new_thread",
                    "content": f"New thread created: {thread_id}",
                    "metadata": {"thread_id": thread_id}
                }, client_id)

            else:
                await manager.send_personal_message({
                    "type": "error",
                    "content": f"Unknown message type: {message_type}",
                    "metadata": {"error_type": "invalid_message"}
                }, client_id)

    finally:
        # ✅ Always cleanup on disconnect or error
        cleanup_client_thread(client_id)
        manager.disconnect(client_id)
        if ping_task and not ping_task.done():
            ping_task.cancel()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime": "running",
        "version": "2.0.0"
    }

@app.get("/docs")
async def get_docs():
    """API documentation endpoint"""
    return {
        "title": "2playz RAG Agent API",
        "version": "2.0.0",
        "description": "Production-ready API for 2playz gaming website RAG agent",
        "endpoints": {
            "POST /chat": "Send message and get complete response",
            "POST /chat/stream": "Send message and get streaming response (SSE)",
            "WebSocket /ws": "Real-time WebSocket communication",
            "GET /status": "Get API status and metrics",
            "GET /threads": "Get active conversation threads",
            "DELETE /threads": "Clear all threads",
            "GET /health": "Health check"
        },
        "websocket_events": {
            "connection_established": "Sent when client connects",
            "processing": "Sent when message processing starts",
            "processing_started": "Sent when agent processing begins",
            "complete_response": "Sent when processing completes",
            "blocked": "Sent when request is blocked",
            "error": "Sent when an error occurs",
            "final_result": "Sent with complete response data"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "api:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True, 
        log_level="info",
        access_log=True
    )
