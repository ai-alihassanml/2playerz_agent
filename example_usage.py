import asyncio
import json
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from rag_agent import (
    graph, create_streaming_graph,
    llm_input_guardrails, llm_detect_language_and_intent,
    translate_text_async, GUARDRAIL_SENTINEL
)
from langchain_core.messages import HumanMessage


app = FastAPI(title="Simple Streaming RAG Agent API")

# ✅ Proper CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------
# Thread management
# ------------------------
active_threads = {}

def get_or_create_thread(thread_id: Optional[str] = None) -> str:
    """
    Get or create a thread. If thread_id is provided and exists, update its last_activity.
    If thread_id is provided but doesn't exist, create a new entry preserving the provided id.
    If thread_id is None, create a new thread id and initialize conversation_history.
    """
    if thread_id:
        # If the thread exists, just update last_activity
        if thread_id in active_threads:
            active_threads[thread_id]["last_activity"] = datetime.now()
            return thread_id
        # If the thread_id was provided but unknown, create it so callers can reuse an id
        active_threads[thread_id] = {
            "created_at": datetime.now(),
            "last_activity": datetime.now(),
            "conversation_history": [],
        }
        return thread_id

    new_thread_id = str(uuid.uuid4())
    active_threads[new_thread_id] = {
        "created_at": datetime.now(),
        "last_activity": datetime.now(),
        "conversation_history": [],
    }
    return new_thread_id


# ------------------------
# Agent runner
# ------------------------



async def run_agent(
    user_message: str,
    thread_id: str,
    language: Optional[str] = None,
    streaming_callback=None,  # function to call with updates if streaming
) -> Dict[str, Any]:
    """
    Main agent wrapper function to handle all steps:
    - Language detection
    - Translation
    - Guardrails
    - Agent execution
    - Translation back
    - Sources and blog handling
    """

    start_time = datetime.now()
    # Load conversation history from active_threads so memory persists across API calls
    thread_meta = active_threads.get(thread_id, {})
    conversation_history: List[str] = thread_meta.get("conversation_history", [])

    # 1️⃣ Detect language
    try:
        detect_info = llm_detect_language_and_intent(user_message)
        detected_lang = detect_info.get("language", "en")
        is_question = detect_info.get("is_question", False)
    except Exception:
        detected_lang = language or "en"
        is_question = user_message.strip().endswith("?")

    # 2️⃣ Translate to English for processing
    if detected_lang != "en":
        try:
            english_message = await translate_text_async(user_message, src=detected_lang, dest="en")
        except Exception:
            english_message = user_message
    else:
        english_message = user_message

    conversation_history.append(f"User: {user_message}")

    # 3️⃣ Guardrail check
    try:
        sanitized = llm_input_guardrails(english_message, history=conversation_history)
    except Exception:
        sanitized = english_message

    # 4️⃣ Blocked by guardrail
    if sanitized.strip() == GUARDRAIL_SENTINEL:
        if detected_lang != "en":
            try:
                sanitized = await translate_text_async(GUARDRAIL_SENTINEL, src="en", dest=detected_lang)
            except Exception:
                sanitized = GUARDRAIL_SENTINEL
        return {
            "response": sanitized,
            "blocked": True,
            "sources": [],
            "metadata": {
                "language_detected": detected_lang,
                "processing_time": (datetime.now() - start_time).total_seconds(),
            }
        }

    # 5️⃣ Build initial state and run agent
    initial_state = {"messages": [HumanMessage(content=sanitized)], "query": sanitized, "retrieved_docs": []}

    try:
        if streaming_callback:
            # Use streaming graph if callback provided
            streaming_graph = create_streaming_graph(streaming_callback)
            final_state = await streaming_graph.ainvoke(initial_state, config={"configurable": {"thread_id": thread_id}})
        else:
            final_state = await graph.ainvoke(initial_state, config={"configurable": {"thread_id": thread_id}})
    except Exception as e:
        return {
            "response": f"Agent execution error: {e}",
            "blocked": False,
            "sources": [],
            "metadata": {
                "language_detected": detected_lang,
                "processing_time": (datetime.now() - start_time).total_seconds(),
            }
        }

    english_answer = final_state["messages"][-1].content

    # 6️⃣ Translate back if needed
    if detected_lang != "en":
        try:
            final_response = await translate_text_async(english_answer, src="en", dest=detected_lang)
        except Exception:
            final_response = english_answer
    else:
        final_response = english_answer

    # 7️⃣ Prepare sources
    sources = []
    for doc in final_state.get("retrieved_docs", []):
        sources.append({
            "title": doc.metadata.get("title", "N/A"),
            "slug": doc.metadata.get("slug", "N/A"),
            "excerpt": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            "source": doc.metadata.get("source", "N/A"),
            "chunk_id": doc.metadata.get("chunk_id", "N/A"),
        })

    # 8️⃣ Return structured result
    metadata = {
        "language_detected": detected_lang,
        "processing_time": (datetime.now() - start_time).total_seconds(),
        "router_decision": "rag" if sources else "llm",
        "thread_id": thread_id,
    }

    # Persist the assistant response and update thread metadata so future calls reuse history
    try:
        # Append the assistant's raw English answer to the conversation history for future guardrail checks
        conversation_history.append(f"Assistant: {english_answer}")
        active_threads.setdefault(thread_id, {})["conversation_history"] = conversation_history
        active_threads[thread_id]["last_activity"] = datetime.now()
    except Exception as e:
        print(f"Warning: failed to persist thread history for {thread_id}: {e}")

    return {
        "response": final_response,
        "blocked": False,
        "sources": sources,
        "metadata": metadata,
    }


from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None
    language: Optional[str] = None
    # If provided, the server will retire this previous thread id and create a new one
    retire_previous_thread: Optional[str] = None


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    SSE streaming endpoint that uses run_agent()
    Accepts POST body with: { "message": "...", "thread_id": "...", "language": "..." }
    """

    if not req.message:
        raise HTTPException(status_code=400, detail="Message is required")

    # Your thread handling logic
    # If retire_previous_thread is provided, remove it from active_threads (so memory/checkpointer
    # won't be reused) and create a fresh thread for the client.
    if req.retire_previous_thread:
        prev = req.retire_previous_thread
        if prev in active_threads:
            # mark for removal; keep one-step removal to avoid race conditions
            del active_threads[prev]
    thread_id = get_or_create_thread(req.thread_id)

    async def event_generator():
        queue: asyncio.Queue = asyncio.Queue()

        async def streaming_callback(update):
            await queue.put(update)

        # Run your agent
        agent_task = asyncio.create_task(
            run_agent(
                req.message,
                thread_id=thread_id,
                language=req.language,
                streaming_callback=streaming_callback
            )
        )

        try:
            while True:
                if agent_task.done() and queue.empty():
                    final_result = agent_task.result()
                    payload = json.dumps({"type": "final", "content": final_result})
                    yield f"data: {payload}\n\n"
                    break

                try:
                    update = await asyncio.wait_for(queue.get(), timeout=0.2)
                    payload = json.dumps({"type": "update", "content": update})
                    yield f"data: {payload}\n\n"
                except asyncio.TimeoutError:
                    await asyncio.sleep(0.05)

        except Exception as e:
            payload = json.dumps({"type": "error", "content": str(e)})
            yield f"data: {payload}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ------------------------
# Health Check
# ------------------------
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("example_usage:app", host="0.0.0.0", port=8000, reload=True)






















# import asyncio
# from datetime import datetime
# from typing import Optional, Dict, Any, List
# from rag_agent import (
#     graph, create_streaming_graph,
#     llm_input_guardrails, llm_detect_language_and_intent,
#     translate_text_async, GUARDRAIL_SENTINEL
# )
# from langchain_core.messages import HumanMessage










# import asyncio
# import json
# import uuid
# from fastapi import FastAPI, Request, HTTPException
# from fastapi.responses import StreamingResponse
# from typing import Optional
# from datetime import datetime



# # Import your run_agent function here
# # from rag_agent_wrapper import run_agent  

# app = FastAPI(title="Simple Streaming RAG Agent API")


# from fastapi.middleware.cors import CORSMiddleware

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:8080"],  # frontend URL
#     allow_credentials=True,
#     allow_methods=["*"],   # allow GET, POST, etc.
#     allow_headers=["*"],   # allow headers like Content-Type
# )











# # Thread management
# active_threads = {}

# def get_or_create_thread(thread_id: Optional[str] = None) -> str:
#     """Reuse thread_id if provided, else create a new one."""
#     if thread_id:
#         # If already seen, update activity timestamp
#         if thread_id in active_threads:
#             active_threads[thread_id]["last_activity"] = datetime.now()
#         else:
#             # Store new thread
#             active_threads[thread_id] = {
#                 "created_at": datetime.now(),
#                 "last_activity": datetime.now(),
#             }
#         return thread_id

#     # Otherwise create fresh
#     new_thread_id = str(uuid.uuid4())
#     active_threads[new_thread_id] = {
#         "created_at": datetime.now(),
#         "last_activity": datetime.now(),
#     }
#     return new_thread_id



# @app.post("/chat/stream")
# async def chat_stream(request: Request):
#     """
#     SSE streaming endpoint that uses run_agent() function
#     """

#     data = await request.json()
#     user_message = data.get("message", "")
#     if not user_message:
#         raise HTTPException(status_code=400, detail="Message is required")

#     language = data.get("language")
#     # ✅ Now reuse thread_id if sent by frontend
#     thread_id = get_or_create_thread(data.get("thread_id"))

#     async def event_generator():
#         queue: asyncio.Queue = asyncio.Queue()

#         async def streaming_callback(update):
#             await queue.put(update)

#         # Run agent
#         agent_task = asyncio.create_task(run_agent(
#             user_message,
#             thread_id=thread_id,
#             language=language,
#             streaming_callback=streaming_callback
#         ))

#         try:
#             while True:
#                 if agent_task.done() and queue.empty():
#                     final_result = agent_task.result()

#                     # ✅ Handle guardrail blocked case
#                     if final_result.get("blocked", False):
#                         payload = json.dumps({"type": "final", "content": final_result})
#                         yield f"data: {payload}\n\n"
#                         break

#                     # Normal final response
#                     payload = json.dumps({"type": "final", "content": final_result})
#                     yield f"data: {payload}\n\n"
#                     break

#                 try:
#                     update = await asyncio.wait_for(queue.get(), timeout=0.1)
#                     payload = json.dumps({"type": "update", "content": update})
#                     yield f"data: {payload}\n\n"
#                 except asyncio.TimeoutError:
#                     await asyncio.sleep(0.05)

#         except Exception as e:
#             payload = json.dumps({"type": "error", "content": str(e)})
#             yield f"data: {payload}\n\n"

#     return StreamingResponse(
#         event_generator(),
#         media_type="text/event-stream",
#         headers={
#             "Cache-Control": "no-cache",
#             "Connection": "keep-alive",
#             "Access-Control-Allow-Origin": "*",
#         }
#     )




# @app.get("/health")
# async def health_check():
#     return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("stream_app:app", host="0.0.0.0", port=8000, reload=True)
