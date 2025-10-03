import os
import json
import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse

# Import the unified handler
from unified_rag_handler import unified_handler, UnifiedRAGResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="2playerz RAG Agent API (Unified)",
    description="Streamlined API for 2playerz gaming website RAG agent with SSE streaming support",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Production-ready CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure specific origins for production
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Request/Response Models
class ChatRequest(BaseModel):
    message: str = Field(..., description="User message to send to the agent", min_length=1)
    thread_id: Optional[str] = Field(None, description="Thread ID for conversation continuity")
    language: Optional[str] = Field(None, description="Preferred language (auto-detect if not provided)")
    conversation_history: Optional[List[str]] = Field(None, description="Previous conversation context for memory management")

class ChatResponse(BaseModel):
    response: str = Field(..., description="Agent response (translated if needed)")
    original_response: str = Field(..., description="Original English response")
    thread_id: str = Field(..., description="Thread ID for conversation continuity")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Retrieved document sources with metadata")
    blog_links: List[Dict[str, Any]] = Field(default_factory=list, description="Blog links when specifically requested")
    slugs: List[str] = Field(default_factory=list, description="List of slugs from sources")
    titles: List[str] = Field(default_factory=list, description="List of titles from sources")
    descriptions: List[str] = Field(default_factory=list, description="List of descriptions from sources")
    language_detected: str = Field(..., description="Detected language of input")
    processing_time: float = Field(..., description="Processing time in seconds")
    router_decision: str = Field(..., description="Whether RAG or LLM was used")
    blocked: bool = Field(False, description="Whether request was blocked by guardrails")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional processing metadata")

# Exception handlers
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

def unified_response_to_chat_response(response: UnifiedRAGResponse) -> ChatResponse:
    """Convert UnifiedRAGResponse to ChatResponse model"""
    return ChatResponse(
        response=response.translated_response,
        original_response=response.response,
        thread_id=response.thread_id,
        sources=response.sources,
        blog_links=response.blog_links,
        slugs=response.slugs,
        titles=response.titles,
        descriptions=response.descriptions,
        language_detected=response.language_detected,
        processing_time=response.processing_time,
        router_decision=response.router_decision,
        blocked=response.blocked,
        metadata=response.metadata
    )

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "2playerz RAG Agent API (Unified)",
        "version": "3.0.0",
        "features": [
            "Unified RAG processing", 
            "Server-Sent Events streaming", 
            "Multi-language support",
            "Complete response data (sources, slugs, titles, descriptions, blog links)",
            "Automatic language detection and translation"
        ],
        "endpoints": {
            "POST /chat": "Complete chat response (non-streaming)",
            "POST /chat/stream": "Streaming chat response (SSE)",
            "GET /status": "API status",
            "GET /threads": "Active threads info"
        },
        "docs": "/docs"
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Complete chat endpoint with full unified agent processing (non-streaming)"""
    try:
        logger.info(f"Processing non-streaming chat request: {request.message[:100]}...")
        
        response = await unified_handler.process_query(
            query=request.message,
            thread_id=request.thread_id,
            language=request.language,
            conversation_history=request.conversation_history
        )

        if response.blocked:
            raise HTTPException(status_code=403, detail=response.translated_response)

        return unified_response_to_chat_response(response)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error in /chat: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Server-Sent Events streaming endpoint with unified processing"""
    try:
        logger.info(f"Processing streaming chat request: {request.message[:100]}...")

        async def generate_sse_stream():
            """Generate Server-Sent Events stream"""
            try:
                async for event in unified_handler.stream_response(
                    query=request.message,
                    thread_id=request.thread_id,
                    language=request.language,
                    conversation_history=request.conversation_history
                ):
                    # Format as Server-Sent Event
                    event_data = json.dumps(event, ensure_ascii=False)
                    yield f"data: {event_data}\n\n"
                    
                    # Small delay to make streaming visible
                    await asyncio.sleep(0.1)
                
                # Send final end event
                yield f"data: {json.dumps({'type': 'end', 'content': 'Stream ended'})}\n\n"
                
            except Exception as e:
                logger.exception(f"Error in SSE stream: {e}")
                error_event = {
                    "type": "error",
                    "content": f"Stream error: {str(e)}",
                    "metadata": {"timestamp": datetime.now().isoformat()}
                }
                yield f"data: {json.dumps(error_event)}\n\n"

        return StreamingResponse(
            generate_sse_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )

    except Exception as e:
        logger.exception(f"Error in streaming endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Streaming error: {str(e)}")

@app.get("/status")
async def get_status():
    """Get API and agent status"""
    return {
        "status": "running",
        "version": "3.0.0",
        "active_threads": len(unified_handler.active_threads),
        "timestamp": datetime.now().isoformat(),
        "features": [
            "Unified RAG processing", 
            "Server-Sent Events streaming", 
            "Multi-language support",
            "Complete response data"
        ]
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
            for thread_id, info in unified_handler.active_threads.items()
        ],
        "total_threads": len(unified_handler.active_threads)
    }

@app.delete("/threads")
async def clear_threads():
    """Clear all active threads"""
    count = len(unified_handler.active_threads)
    unified_handler.active_threads.clear()
    return {
        "message": f"Cleared {count} threads",
        "timestamp": datetime.now().isoformat()
    }

@app.delete("/threads/{thread_id}")
async def delete_thread(thread_id: str):
    """Delete a specific thread"""
    if thread_id in unified_handler.active_threads:
        del unified_handler.active_threads[thread_id]
        return {"message": f"Thread {thread_id} deleted"}
    else:
        raise HTTPException(status_code=404, detail="Thread not found")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "3.0.0",
        "unified_handler": "active"
    }

@app.get("/test/simple")
async def test_simple():
    """Simple test endpoint to verify API is working"""
    try:
        response = await unified_handler.process_query(
            query="Hello, what is 2playerz?",
            thread_id=None,
            language="en"
        )
        
        return {
            "test": "success",
            "response_preview": response.translated_response[:100] + "..." if len(response.translated_response) > 100 else response.translated_response,
            "language_detected": response.language_detected,
            "processing_time": response.processing_time,
            "blocked": response.blocked
        }
    except Exception as e:
        logger.exception(f"Test endpoint error: {e}")
        return {"test": "failed", "error": str(e)}

@app.get("/test/memory")
async def test_memory():
    """Test memory management with conversation continuity"""
    try:
        # First message
        thread_id = str(uuid.uuid4())
        response1 = await unified_handler.process_query(
            query="My name is Alihassan",
            thread_id=thread_id,
            language="en"
        )
        
        # Second message in same thread
        response2 = await unified_handler.process_query(
            query="What is my name?",
            thread_id=thread_id,
            language="en"
        )
        
        return {
            "test": "memory_test",
            "thread_id": thread_id,
            "first_response": response1.translated_response[:100] + "...",
            "second_response": response2.translated_response[:100] + "...",
            "memory_working": "Alihassan" in response2.translated_response,
            "processing_time": response2.processing_time
        }
    except Exception as e:
        logger.exception(f"Memory test error: {e}")
        return {"test": "failed", "error": str(e)}

@app.get("/test/stream")
async def test_stream():
    """Test streaming endpoint"""
    async def test_stream_generator():
        try:
            async for event in unified_handler.stream_response(
                query="Tell me about gaming news",
                thread_id=None,
                language="en"
            ):
                yield f"data: {json.dumps(event)}\n\n"
                await asyncio.sleep(0.2)
        except Exception as e:
            error_event = {"type": "error", "content": f"Test stream error: {str(e)}"}
            yield f"data: {json.dumps(error_event)}\n\n"
    
    return StreamingResponse(
        test_stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )

# Application lifecycle
@app.on_event("startup")
async def on_startup():
    logger.info("2playerz RAG Agent API (Unified) starting up...")
    logger.info("Unified handler initialized and ready")

@app.on_event("shutdown")
async def on_shutdown():
    logger.info("2playerz RAG Agent API shutting down...")
    # Clear all threads on shutdown
    unified_handler.active_threads.clear()

if __name__ == "__main__":
    uvicorn.run(
        "api_new:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True, 
        log_level="info",
        access_log=True
    )