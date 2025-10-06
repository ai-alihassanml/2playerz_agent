import os
import uuid
import asyncio
import json

from typing_extensions import TypedDict, Annotated
import operator
from langchain_core.messages import HumanMessage  
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List

from langgraph.graph import StateGraph, END, START
from langchain_community.vectorstores import FAISS  
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpointEmbeddings

from langchain_core.documents import Document 
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.types import Command

from googletrans import Translator  
import asyncio as _asyncio
import inspect

from langgraph.checkpoint.memory import MemorySaver   


load_dotenv()

# --- 1. Agent State Definition ---
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    retrieved_docs: List[Document]
    query: str
    conversation_history: List[str]

# --- 2. Environment Setup & Component Loading ---
HUGGINGFACEHUB_API_TOKEN = os.getenv("HAGGINGFACEHUB_API_TOKEN")
if not HUGGINGFACEHUB_API_TOKEN:
    raise ValueError("HAGGINGFACEHUB_API_TOKEN not found in environment variables.")

# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# if not OPENROUTER_API_KEY:
#     raise ValueError("OPENROUTER_API_KEY not found in environment variables.")

OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

# initialize models / vector DB (adjust model names / keys as needed)
try:
    embeddings = HuggingFaceEndpointEmbeddings(
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    llm = ChatOpenAI(
        model="gpt-4",
        openai_api_key=OPENAI_API_KEY,
    )

    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    # Use similarity search (top-k) by default. The previous "similarity_score_threshold" mode
    # expects normalized relevance scores in [0,1], but FAISS may return raw distances or unnormalized
    # scores leading to unexpected filtering. Using plain similarity (k) and adding a fallback in
    # `retrieve_documents` ensures we still return top candidates when thresholding filters them out.
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
except Exception as e:
    print(f"Error loading models or FAISS index: {e}")
    raise SystemExit(1)

GUARDRAIL_SENTINEL = (
    "I am not able to assist you with this topic. "
    "If you have any questions related to gaming or the 2playerz website, please ask and I'll try to help."
)



BASE_URL = "https://2playerz.de/p/"



def llm_input_guardrails(input_text: str, history: list[str] | None = None) -> str:
    print("--- Running input guardrail moderation ---")
    # Ask the LLM to output a small JSON indicating decision and optional sanitized text.
    # This makes parsing deterministic. Example output:
    # {"decision": "ALLOW", "text": "sanitized text here"}
    # If conversation history is provided, include a short recent-history section for context
    history_section = ""
    if history:
        recent = history[-6:]
        history_section = "\nRecent conversation history (most recent last):\n" + "\n".join(recent) + "\n\n"

    prompt = ChatPromptTemplate.from_template("""
    You are a strict guardrail filter for a gaming website (2playerz.de) acting as the 2playerz AI assistant.
    Your ONLY job is to decide whether the user's input should be processed (ALLOW) or blocked (BLOCK).

    RULES (be strict):
      - ALLOW only gaming-related topics: game news, reviews, guides, tips, game titles, gaming hardware, streamers,
        gaming platforms, and non-offensive small talk or greetings.If the question is related to basic hello , hi ,name .who are you or user ask about tha the working of 2playerz website then allow it.
      - BLOCK anything not related to gaming, or anything offensive, illegal, or harmful.

    Output a single valid JSON object with two keys:
      - "decision": either "ALLOW" or "BLOCK"
      - "text": the sanitized text to use (for ALLOW) or an explanatory short phrase (for BLOCK)

    IMPORTANT: Output only the JSON object and nothing else.

    {history_section}Input: {input_text}
    """)

    moderation_chain = (prompt | llm | StrOutputParser())

    raw = ""
    try:
        raw = moderation_chain.invoke({"input_text": input_text, "history_section": history_section})
    except Exception as e:
        print(f"Guardrail LLM invocation error: {e}")

    print(f"Guardrail raw output: {raw!r}")

    # Try to parse JSON first
    try:
        parsed = json.loads(raw.strip())
        decision = str(parsed.get("decision", "BLOCK")).upper()
        text = parsed.get("text", "")
    except Exception:
        # Fallback: do a keyword-based parse
        lowered = (raw or "").lower()
        if "allow" in lowered and "block" not in lowered:
            decision = "ALLOW"
            text = input_text
        elif "block" in lowered:
            decision = "BLOCK"
            text = ""
        else:
            # If unsure, default to BLOCK for safety
            decision = "BLOCK"
            text = ""

    if decision == "BLOCK":
        print("Guardrail decision: BLOCK - returning sentinel")
        return GUARDRAIL_SENTINEL

    # If allowed, return sanitized text if provided, otherwise return original input
    sanitized = text.strip() if isinstance(text, str) and text.strip() else input_text
    print("Guardrail decision: ALLOW - passing through sanitized text")
    return sanitized



def llm_detect_language_and_intent(text: str, min_confidence: float = 0.60) -> dict:
    """
    Use the LLM to detect language and whether input is a question.
    Returns: {"language": "en", "is_question": True/False, "confidence": 0.0-1.0}
    If LLM confidence is below min_confidence we try googletrans.detect as fallback.
    """
    print("--- Running LLM-based language + intent detection ---")
    prompt = ChatPromptTemplate.from_template("""
    Detect the language of the following text.
    Answer only with a 2-letter ISO 639-1 code (like en, de, fr, es, it, hi).

    Text: {text}
    """)
    detector_chain = (prompt | llm | StrOutputParser())
    raw = detector_chain.invoke({"text": text})

    # The LLM may return a plain language code like 'de' or a JSON string.
    language = "und"
    is_question = text.strip().endswith("?")
    confidence = 0.0
    try:
        stripped = raw.strip()
        # If it's a plain two-letter code, accept it
        if len(stripped) == 2 and stripped.isalpha():
            language = stripped.lower()
            confidence = 0.8
        else:
            # try parse JSON
            parsed = json.loads(stripped)
            language = parsed.get("language", "und")
            is_question = bool(parsed.get("is_question", is_question))
            confidence = float(parsed.get("confidence", 0.0))
    except Exception as e:
        print(f"LLM detection parse error: {e}; raw output: {raw!r}")

    # If LLM is unsure, use googletrans detection as a backup
    if confidence < min_confidence or language == "und":
        try:
            # googletrans Translator may be sync or async in different installs; handle both
            def _detect_sync(t: str):
                tr = Translator()
                detect_method = tr.detect
                try:
                    # If detect_method is async, run it to completion
                    if inspect.iscoroutinefunction(detect_method):
                        return _asyncio.run(detect_method(t))
                    res = detect_method(t)
                    # If result is a coroutine (unexpected), run it
                    if inspect.iscoroutine(res):
                        return _asyncio.run(res)
                    return res
                except Exception:
                    # last-resort attempt: call and return whatever we get
                    return detect_method(t)

            # Run the detect function in a thread to avoid blocking
            detected = _asyncio.run(_asyncio.to_thread(_detect_sync, text))

            detected_lang = getattr(detected, 'lang', detected)
            if detected_lang:
                print(f"Fallback detect found language {detected_lang} (LLM conf {confidence})")
                language = detected_lang
                confidence = max(confidence, min(0.65, getattr(detected, 'confidence', 0.65)))
        except Exception as e:
            print(f"Fallback (googletrans) detection error: {e}")

    return {"language": language, "is_question": is_question, "confidence": confidence}


async def translate_text_async(text: str, src: str, dest: str) -> str:
    """
    Async wrapper around synchronous googletrans Translator using asyncio.to_thread.
    If src is 'und' or unknown, fall back to 'auto'.
    """
    try:
        def _translate_sync():
            tr = Translator()
            translate_method = tr.translate
            src_arg = src if src and src != "und" else "auto"
            # If the translate method is async, run it and return the finished result
            if inspect.iscoroutinefunction(translate_method):
                res = _asyncio.run(translate_method(text, src=src_arg, dest=dest))
            else:
                res = translate_method(text, src=src_arg, dest=dest)
                if inspect.iscoroutine(res):
                    res = _asyncio.run(res)
            # res may be an object with .text or a string
            return getattr(res, 'text', str(res))

        # Use asyncio.to_thread for proper async execution
        result = await _asyncio.to_thread(_translate_sync)
        return result
    except Exception as e:
        print(f"Async translation error ({src}->{dest}): {e}")
        return text


# Convenience wrappers using the new async translate_text_async
async def detect_and_translate_to_english(text: str, user_lang: str) -> str:
    if not user_lang or user_lang == "en":
        return text
    return await translate_text_async(text, src=user_lang, dest="en")


async def translate_back_to_user_lang(text: str, user_lang: str) -> str:
    if not user_lang or user_lang == "en":
        return text
    return await translate_text_async(text, src="en", dest=user_lang)

# --- Graph nodes ---
def route_query(state: AgentState):
    print("---NODE: ROUTING QUERY---")
    query = state['query']
    history = state.get('conversation_history', [])
    history_text = "\n".join(history[-6:]) if history else ""
    prompt = ChatPromptTemplate.from_template("""
    You are a routing agent for a gaming website (2playerz.de). Decide whether a user query should be answered 
    directly by the LLM ("llm") or by retrieving from the RAG knowledge base ("rag").

    IMPORTANT: When in doubt, prefer "rag" for any gaming-related content.
             - and if user query is block then prefer "llm" to handle it gracefully.
             - If querry is about 2playerz website or basic hello then prefer "llm" to handle it gracefully.

    Rules:
    - Use "llm" ONLY for:
      - Simple greetings ("hi", "hello", "good morning", "hey")
      - Questions about the AI itself ("what is your name?", "who are you?")
      - Basic small talk that passed the guardrail

    - Use "rag" for ALL gaming-related content:
      - ANY mention of video games, game titles, or gaming companies
      - Questions about gaming industry, reviews, guides, updates
      - Game names (like "Death Stranding", "Call of Duty", "FIFA", etc.)
      - Gaming companies (like "Naughty Dog", "Ubisoft", "EA", etc.)
      - Gaming personalities, streamers, esports
      - Gaming hardware, consoles, platforms
      - Gaming news, releases, updates
      - Declarative requests like "tell me about [game/company]", "what is [game]"
      - ANY gaming-related information request, even if not phrased as a question

    Examples that should use "rag":
    - "tell me about Death Stranding 2" → rag
    - "what is Call of Duty" → rag
    - "Naughty Dog games" → rag
    - "gaming news" → rag
    - "best RPG games" → rag
    - "Black Myth Wukong" → rag

    Examples that should use "llm":
    - "hello" → llm
    - "what is your name?" → llm
    - "hi there" → llm

    Respond with only one word: "llm" or "rag".

    Question: {question}
    RecentHistory: {recent_history}
    """)
    router_chain = (prompt | llm | StrOutputParser())
    decision = router_chain.invoke({"question": query, "recent_history": history_text}).strip().lower()
    print(f"---Router Decision: {decision}---")
    if "rag" in decision:
        return Command(goto="retrieve_documents", update={"router_decision": "rag"})
    else:
        return Command(goto="generate_answer_without_docs", update={"router_decision": "llm"})

async def generate_answer_without_docs(state: AgentState, websocket_manager=None, client_id=None):
    print("---NODE: GENERATING ANSWER (NO RETRIEVAL)---")
    messages = state['messages']
    history_str = "\n".join(f"{msg.type.capitalize()}: {msg.content}" for msg in messages)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are 2playerz AI assistant for 2playerz website , 2Playerz.de is your portal for news and game reviews from PlayStation, Xbox, and Nintendo. Which new games have been released and how do they perform in our reviews? When will new hardware be released from Sony, Microsoft, or Nintendo, and what are their capabilities? What rumors are currently circulating, and what's the truth behind them? We'll be the first to hear about them! IMPORTANT: You MUST respond ONLY in English. Do not switch to any other language during your response. Keep your entire response consistent in English throughout. Never use German, French, Spanish, or any other language. If you start a response in English, continue in English for the entire response. Answer in English and use history to answer."),
        ("human", "{history}"),
    ])
    
    # Stream the response
    formatted_messages = prompt.format_messages(history=history_str)
    stream = llm.astream(formatted_messages)
    answer = ""
    
    print("Assistant: ", end="", flush=True)
    async for chunk in stream:
        if hasattr(chunk, 'content') and chunk.content:
            print(chunk.content, end="", flush=True)
            answer += chunk.content
            
            # Send chunk via WebSocket if available
            if websocket_manager and client_id:
                try:
                    await websocket_manager.send_personal_message({
                        "type": "chunk",
                        "content": chunk.content,
                        "metadata": {"streaming": True}
                    }, client_id)
                except Exception as e:
                    print(f"Error sending chunk via WebSocket: {e}")
    
    print()  # New line after streaming
    return {"messages": [AIMessage(content=answer)]}

def retrieve_documents(state: AgentState):
    print("---NODE: RETRIEVING DOCUMENTS---")
    query = state['query']
    # First try the configured retriever (top-k similarity)
    try:
        retrieved_docs = retriever.invoke(query)
    except Exception as e:
        print(f"Retriever invocation error: {e}")
        retrieved_docs = []

    # If no docs were returned (for example due to unexpected relevance-score behavior),
    # fall back to FAISS similarity_search_with_relevance_scores to fetch top-k candidates
    # and ignore any score thresholding issues.
    try:
        if not retrieved_docs:
            print("No docs returned by retriever; falling back to FAISS top-k similarity search")
            # similarity_search_with_relevance_scores returns list of (Document, score)
            raw = db.similarity_search_with_relevance_scores(query, k=3)
            # extract just the Document objects
            retrieved_docs = [t[0] for t in raw]
    except Exception as e:
        print(f"Fallback FAISS search error: {e}")

    return {"retrieved_docs": retrieved_docs}

async def generate_answer(state: AgentState, websocket_manager=None, client_id=None):
    print("---NODE: GENERATING ANSWER---")
    query = state['query']
    retrieved_docs = state['retrieved_docs']
    template = """
    You are a 2playerz AI assistant for a gaming website (2playerz.de).
    Generate a comprehensive and accurate answer based on the provided context and question.
    Generate answer in humanized format. 
    Always try to be helpful, clear, and friendly. 
    
    IMPORTANT: You MUST respond ONLY in English. Do not switch to any other language during your response.
    Keep your entire response consistent in English throughout. Never use German, French, Spanish, or any other language.
    If you start a response in English, continue in English for the entire response.

    If the answer is not in the provided context:
    - Suggest related ideas or ask the user from what related you retrieve to clarify what they mean, so 
        you can guide them better.

    - If the question is blocked then.
     ans : I am not designed to assist with that topic. I am designed to help as 2playerz assistant. If you have any gaming-related questions, feel free to ask!

    Context:
    {context}

    Question: {question}

    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)
    context_str = "\n\n".join(
        f"Title: {doc.metadata.get('title','N/A')}\nSlug: {doc.metadata.get('slug','N/A')}\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )
    
    # If the user explicitly asked for blog links / posts, return a brief summary + links
    link_request_keywords = ["blog", "blogs", "link", "links", "post", "posts", "read more", "blog post", "give me links", "give me blog links", "show me links", "sources"]
    lowered = (query or "").lower()
    is_link_request = any(k in lowered for k in link_request_keywords)

    if is_link_request and retrieved_docs:
        # Build a compact, human-friendly list of blog links using BASE_URL + slug
        parts = ["Here are the related blog posts I found:\n"]
        for doc in retrieved_docs:
            title = doc.metadata.get("title", "(untitled)")
            slug = doc.metadata.get("slug")
            excerpt = (doc.page_content or "").strip().replace("\n", " ")[:200]
            if slug:
                url = BASE_URL.rstrip('/') + '/' + slug.lstrip('/')
                parts.append(f"- {title}: {excerpt}...\n  Link: {url}\n")
            else:
                # If no slug is present, include title + excerpt but no link
                parts.append(f"- {title}: {excerpt}...\n  Link: (no slug available)\n")

        answer = "\n".join(parts)
        # Send a single response (no streaming) and include retrieved_docs for provenance
        print(answer)
        return {"messages": [AIMessage(content=answer)], "retrieved_docs": retrieved_docs}

    # Stream the response (default behavior)
    formatted_messages = prompt.format_messages(context=context_str, question=query)
    stream = llm.astream(formatted_messages)
    answer = ""
    
    print("", end="", flush=True)
    async for chunk in stream:
        if hasattr(chunk, 'content') and chunk.content:
            print(chunk.content, end="", flush=True)
            answer += chunk.content
            
            # Send chunk via WebSocket if available
            if websocket_manager and client_id:
                try:
                    await websocket_manager.send_personal_message({
                        "type": "chunk",
                        "content": chunk.content,
                        "metadata": {"streaming": True}
                    }, client_id)
                except Exception as e:
                    print(f"Error sending chunk via WebSocket: {e}")

    print()  # New line after streaming
    return {"messages": [AIMessage(content=answer)], "retrieved_docs": retrieved_docs}



memory = MemorySaver()

workflow = StateGraph(AgentState, checkpointers=memory)

workflow.add_node("route_query", route_query)
workflow.add_node("retrieve_documents", retrieve_documents)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("generate_answer_without_docs", generate_answer_without_docs)

workflow.add_edge(START, "route_query")
workflow.add_edge("retrieve_documents", "generate_answer")
workflow.add_edge("generate_answer", END)
workflow.add_edge("generate_answer_without_docs", END)

graph = workflow.compile(checkpointer=memory)

# Create a streaming-enabled graph wrapper
def create_streaming_graph(websocket_manager=None, client_id=None):
    """Create a graph with streaming capabilities"""
    
    async def streaming_generate_answer_without_docs(state: AgentState):
        return await generate_answer_without_docs(state, websocket_manager, client_id)
    
    async def streaming_generate_answer(state: AgentState):
        return await generate_answer(state, websocket_manager, client_id)
    
    # Create new workflow with streaming functions
    streaming_workflow = StateGraph(AgentState, checkpointers=memory)
    streaming_workflow.add_node("route_query", route_query)
    streaming_workflow.add_node("retrieve_documents", retrieve_documents)
    streaming_workflow.add_node("generate_answer", streaming_generate_answer)
    streaming_workflow.add_node("generate_answer_without_docs", streaming_generate_answer_without_docs)
    
    streaming_workflow.add_edge(START, "route_query")
    streaming_workflow.add_edge("retrieve_documents", "generate_answer")
    streaming_workflow.add_edge("generate_answer", END)
    streaming_workflow.add_edge("generate_answer_without_docs", END)
    
    return streaming_workflow.compile(checkpointer=memory)

print("\n--- RAG Agent initialized. ---")
thread_uuid = uuid.uuid1()
print(f"Thread UUID: {thread_uuid}")

USER_LANG = None  # "en" or "de"





async def main():
    print("[info] Language detection switched to LLM-based auto-detection (multi-language).")
    # Keep a simple conversation history (strings) for context-aware guardrail decisions
    conversation_history: list[str] = []

    while True:
        thread_id = str(thread_uuid)
        config = {"configurable": {"thread_id": thread_id}}

        user_query = input("\n\nEnter your question (or 'exit' to quit): ").strip()
        if user_query.lower() == "exit":
            print("Exiting.")
            break
        if not user_query:
            continue

        # 1) LLM detects the user's language & whether it's a question
        try:
            detect_info = llm_detect_language_and_intent(user_query)
            detected_lang = detect_info.get("language", "en")
            is_question = detect_info.get("is_question", False)
            print(f"[detect] lang={detected_lang} is_question={is_question} conf={detect_info.get('confidence')}")
        except Exception as e:
            print(f"Language detection failed: {e}")
            detected_lang = "en"
            is_question = user_query.strip().endswith("?")

        # 2) Translate to English (for processing) if needed
        if detected_lang and detected_lang != "en":
            try:
                english_query = await translate_text_async(user_query, src=detected_lang, dest="en")
            except Exception as e:
                print(f"Translation to English failed: {e}")
                english_query = user_query
        else:
            english_query = user_query

        # Record the user input into history (raw, pre-translation) for context
        conversation_history.append(f"User: {user_query}")

        # 3) Sanitize/moderate via guardrail (runs on English)
        try:
            sanitized = llm_input_guardrails(english_query, history=conversation_history)
        except Exception as e:
            print(f"Moderation/guardrail check failed: {e}")
            sanitized = english_query

        # 4) Guardrail sentinel handling - BLOCK the request and stop processing
        if isinstance(sanitized, str) and sanitized.strip() == GUARDRAIL_SENTINEL:
            print(f"\n--- GUARDRAIL BLOCKED REQUEST ---")
            try:
                translated_guardrail = await translate_text_async(GUARDRAIL_SENTINEL, src="en", dest=detected_lang)
            except Exception as e:
                print(f"Translation of guardrail failed: {e}")
                translated_guardrail = GUARDRAIL_SENTINEL
            print(f"Assistant: {translated_guardrail}")
            print("--- REQUEST BLOCKED - NO FURTHER PROCESSING ---")
            # Append assistant block reply to history so future short replies like "yes" are judged in context
            conversation_history.append(f"Assistant: {GUARDRAIL_SENTINEL}")
            continue

        # 5) Build initial state (English) and run graph workflow (unchanged)
        initial_state = {
            "messages": [HumanMessage(content=sanitized)],
            "query": sanitized,
            "retrieved_docs": [],
            "conversation_history": conversation_history
        }

        try:
            final_state = await graph.ainvoke(initial_state, config=config)
            english_answer = final_state["messages"][-1].content

            # 6) Translate back to user language (if needed) - only if response wasn't already streamed
            if detected_lang and detected_lang != "en" and not english_answer:
                try:
                    final_response = await translate_text_async(english_answer, src="en", dest=detected_lang)
                    print("\n--- Agent's Final Response (Translated) ---")
                    print(f"Assistant: {final_response}")
                except Exception as e:
                    print(f"Translation back to user language failed: {e}")
                    print(f"Assistant: {english_answer}")
            elif detected_lang and detected_lang != "en" and english_answer:
                # Stream the translation back to user language
                try:
                    print("\n--- Translating response back to your language ---")
                    print("Assistant: ", end="", flush=True)
                    translated_response = await translate_text_async(english_answer, src="en", dest=detected_lang)
                    print(translated_response)
                except Exception as e:
                    print(f"Translation back to user language failed: {e}")
                    print(f"Assistant: {english_answer}")

                # Append assistant response to conversation history to provide context for next guardrail decisions
                conversation_history.append(f"Assistant: {english_answer}")

            # Print retrieved documents (sources)
            if final_state.get("retrieved_docs"):
                print("\n--- Retrieved Documents (sources) ---")
                for doc in final_state["retrieved_docs"]:
                    slug = doc.metadata.get("slug", "N/A")
                    title = doc.metadata.get("title", "N/A")
                    print(f"Title: {title}\nSlug: {slug}\nExcerpt: {doc.page_content[:150]}...\n")
            else:
                print("\n--- Retrieved Documents (sources) ---\nNone")

        except Exception as e:
            print(f"An error occurred during agent execution: {e}")


# Run the async loop
if __name__ == "__main__":
    asyncio.run(main())