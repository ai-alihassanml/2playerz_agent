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

    db = FAISS.load_local("faiss_index2", embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})
except Exception as e:
    print(f"Error loading models or FAISS index: {e}")
    raise SystemExit(1)

GUARDRAIL_SENTINEL = "I am not able to assist with that topic. I am designed to help as 2playz assistant.If you have any gaming-related questions, feel free to ask!"



def llm_input_guardrails(input_text: str) -> str:
    print("--- Running input guardrail moderation ---")

    prompt = ChatPromptTemplate.from_template("""
    You are a strict guardrail filter for a gaming website (2playz.de).  
    Your ONLY job is to decide if the input should be processed or blocked.  

    STRICT RULES - ONLY allow these:
    1. Gaming-related content (games, gaming companies, esports, gaming news, reviews, guides)
    2. Simple greetings ("hi", "hello", "good morning", "hey" , what is my name? etc)
    3. Questions about the AI assistant itself ("what is your name?", "who are you?")
    4. Basic small talk that is gaming-related or neutral
    5 .i want that understand the user question and if a simple chatbot can answer it then allow it

    BLOCK everything else including:
    - Not Allowed topics: politics, religion, adult content, violence, hate speech, illegal activities, personal data requests

    If the input is ALLOWED → output EXACTLY the same input.
    If the input should be BLOCKED → output EXACTLY: "{GUARDRAIL_SENTINEL}".

    Do NOT answer the question.  
    Do NOT rephrase or change the input.  
    Just return either the original input or the sentinel.  

    Input: {input_text}

    Output:
    """)

    moderation_chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    result = moderation_chain.invoke({
        "input_text": input_text,
        "GUARDRAIL_SENTINEL": GUARDRAIL_SENTINEL
    })

    print(f"Guardrail output: {result!r}")
    return result



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
    prompt = ChatPromptTemplate.from_template("""
    You are a routing agent for a gaming website (2playz.de). Decide whether a user query should be answered 
    directly by the LLM ("llm") or by retrieving from the RAG knowledge base ("rag").

    IMPORTANT: When in doubt, prefer "rag" for any gaming-related content.

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
    """)
    router_chain = (prompt | llm | StrOutputParser())
    decision = router_chain.invoke({"question": query}).strip().lower()
    print(f"---Router Decision: {decision}---")
    if "rag" in decision:
        return Command(goto="retrieve_documents", update={"router_decision": "rag"})
    else:
        return Command(goto="generate_answer_without_docs", update={"router_decision": "llm"})

async def generate_answer_without_docs(state: AgentState):
    print("---NODE: GENERATING ANSWER (NO RETRIEVAL)---")
    messages = state['messages']
    history_str = "\n".join(f"{msg.type.capitalize()}: {msg.content}" for msg in messages)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "you are a helpful 2playz assistant for 2playz website . this is game nes and blog related websit Answer in English and use history to answer."),
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
    
    print()  # New line after streaming
    return {"messages": [AIMessage(content=answer)]}

def retrieve_documents(state: AgentState):
    print("---NODE: RETRIEVING DOCUMENTS---")
    query = state['query']
    retrieved_docs = retriever.invoke(query)
    return {"retrieved_docs": retrieved_docs}

async def generate_answer(state: AgentState):
    print("---NODE: GENERATING ANSWER---")
    query = state['query']
    retrieved_docs = state['retrieved_docs']
    template = """
    You are a 2playz ai assistant for a gaming website (2playz.de).
    Genrate a comprehensive and accurate answer based on the provided context and question.
    if the answer is not in the context, say "I am not able to assist with that topic.Based the the provided  date i not have information about this.!"


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
    
    # Stream the response
    formatted_messages = prompt.format_messages(context=context_str, question=query)
    stream = llm.astream(formatted_messages)
    answer = ""
    
    print("Assistant: ", end="", flush=True)
    async for chunk in stream:
        if hasattr(chunk, 'content') and chunk.content:
            print(chunk.content, end="", flush=True)
            answer += chunk.content
    
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

print("\n--- RAG Agent initialized. ---")
thread_uuid = uuid.uuid1()
print(f"Thread UUID: {thread_uuid}")

# --- 5. Main Execution Loop with one-time language selection ---
USER_LANG = None  # "en" or "de"




# -------------------------------
# Main async loop
# -------------------------------
async def main():
    print("[info] Language detection switched to LLM-based auto-detection (multi-language).")

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

        # 3) Sanitize/moderate via guardrail (runs on English)
        try:
            sanitized = llm_input_guardrails(english_query)
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
            continue

        # 5) Build initial state (English) and run graph workflow (unchanged)
        initial_state = {
            "messages": [HumanMessage(content=sanitized)],
            "query": sanitized,
            "retrieved_docs": []
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