import os
import uuid
import asyncio
import json
from datetime import datetime
from typing import Dict, Any, Optional, AsyncGenerator, List
import logging

from langchain_core.messages import HumanMessage, AIMessage

# Import your existing RAG agent components
from rag_agent import (
    graph, llm_input_guardrails, llm_detect_language_and_intent,
    translate_text_async, GUARDRAIL_SENTINEL, AgentState, BASE_URL
)

logger = logging.getLogger(__name__)

class UnifiedRAGResponse:
    """Structure for unified RAG agent response"""
    def __init__(self):
        self.response: str = ""
        self.translated_response: str = ""
        self.language_detected: str = "en"
        self.sources: List[Dict] = []
        self.blog_links: List[Dict] = []
        self.slugs: List[str] = []
        self.titles: List[str] = []
        self.descriptions: List[str] = []
        self.processing_time: float = 0.0
        self.blocked: bool = False
        self.router_decision: str = "unknown"
        self.thread_id: str = ""
        self.metadata: Dict[str, Any] = {}

class UnifiedRAGHandler:
    """Unified handler for all RAG agent operations"""
    
    def __init__(self):
        self.active_threads = {}
        
    def get_or_create_thread(self, thread_id: Optional[str] = None) -> str:
        """Get existing thread or create new one"""
        if thread_id and thread_id in self.active_threads:
            self.active_threads[thread_id]["last_activity"] = datetime.now()
            return thread_id
        
        new_thread_id = str(uuid.uuid4())
        self.active_threads[new_thread_id] = {
            "created_at": datetime.now(),
            "message_count": 0,
            "last_activity": datetime.now()
        }
        return new_thread_id

    async def process_query(
        self, 
        query: str, 
        thread_id: Optional[str] = None,
        language: Optional[str] = None,
        conversation_history: Optional[List[str]] = None
    ) -> UnifiedRAGResponse:
        """
        Unified function to process all types of RAG queries.
        
        Returns complete response with:
        - Original and translated responses
        - Language detection
        - Sources with metadata (slugs, titles, descriptions)
        - Blog links when requested
        - All processing metadata
        """
        response = UnifiedRAGResponse()
        start_time = datetime.now()
        
        # Ensure we have a thread ID
        if not thread_id:
            thread_id = self.get_or_create_thread()
        response.thread_id = thread_id
        
        # CRITICAL: Use LangGraph's checkpointing system properly
        # The checkpointing system automatically maintains conversation history
        # We need to pass the conversation_history to the guardrail function
        # and let LangGraph handle the messages field
        final_conversation_history = conversation_history or []
        
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            # 1) Language detection
            logger.info("Starting language detection...")
            try:
                detect_info = llm_detect_language_and_intent(query)
                response.language_detected = detect_info.get("language", "en")
                is_question = detect_info.get("is_question", False)
                confidence = detect_info.get("confidence", 0.0)
                
                response.metadata.update({
                    "is_question": is_question,
                    "language_confidence": confidence
                })
                
                logger.info(f"Language detected: {response.language_detected} (confidence: {confidence})")
            except Exception as e:
                logger.error(f"Language detection failed: {e}")
                response.language_detected = language or "en"
                is_question = query.strip().endswith("?")

            # 2) Translate to English if needed
            english_query = query
            if response.language_detected and response.language_detected != "en":
                logger.info(f"Translating from {response.language_detected} to English...")
                try:
                    english_query = await translate_text_async(query, src=response.language_detected, dest="en")
                    logger.info("Translation to English completed")
                except Exception as e:
                    logger.error(f"Translation to English failed: {e}")
                    english_query = query

            # 3) Guardrail check
            logger.info("Running guardrail check...")
            try:
                # Pass conversation history to guardrail for context-aware moderation
                sanitized = llm_input_guardrails(english_query, history=final_conversation_history)
                logger.info("Guardrail check completed")
            except Exception as e:
                logger.error(f"Moderation/guardrail check failed: {e}")
                sanitized = english_query

            # 4) Handle guardrail blocking
            if isinstance(sanitized, str) and sanitized.strip() == GUARDRAIL_SENTINEL:
                logger.warning("Request blocked by guardrails")
                response.blocked = True
                response.response = GUARDRAIL_SENTINEL
                
                # Translate guardrail message if needed
                if response.language_detected and response.language_detected != "en":
                    try:
                        response.translated_response = await translate_text_async(
                            GUARDRAIL_SENTINEL, src="en", dest=response.language_detected
                        )
                    except Exception as e:
                        logger.error(f"Translation of guardrail failed: {e}")
                        response.translated_response = GUARDRAIL_SENTINEL
                else:
                    response.translated_response = GUARDRAIL_SENTINEL
                
                response.processing_time = (datetime.now() - start_time).total_seconds()
                return response

            # 5) Build initial state and run graph
            logger.info("Starting agent processing...")
            
            # CRITICAL: LangGraph's checkpointing system automatically maintains conversation history
            # The messages field in the state will contain the full conversation when using the same thread_id
            # We only need to provide the current message - LangGraph will append it to existing messages
            initial_state = {
                "messages": [HumanMessage(content=sanitized)],  # LangGraph appends this to existing conversation
                "query": sanitized,
                "retrieved_docs": [],
                "conversation_history": final_conversation_history  # For guardrail context
            }

            # Run the agent graph - LangGraph's checkpointing will maintain conversation memory
            final_state = await graph.ainvoke(initial_state, config=config)
            english_answer = final_state["messages"][-1].content
            response.response = english_answer

            # 6) Translate back to user language if needed
            if response.language_detected and response.language_detected != "en":
                logger.info(f"Translating response back to {response.language_detected}...")
                try:
                    response.translated_response = await translate_text_async(
                        english_answer, src="en", dest=response.language_detected
                    )
                    logger.info("Translation back completed")
                except Exception as e:
                    logger.error(f"Translation back to user language failed: {e}")
                    response.translated_response = english_answer
            else:
                response.translated_response = english_answer

            # 7) Process retrieved documents and extract all metadata
            if final_state.get("retrieved_docs"):
                logger.info(f"Processing {len(final_state['retrieved_docs'])} retrieved documents...")
                
                for doc in final_state["retrieved_docs"]:
                    # Extract metadata
                    title = doc.metadata.get("title", "N/A")
                    slug = doc.metadata.get("slug", "N/A")
                    excerpt = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    
                    # Build comprehensive source info
                    source_info = {
                        "title": title,
                        "slug": slug,
                        "excerpt": excerpt,
                        "description": doc.metadata.get("description", excerpt),
                        "source": doc.metadata.get("source", "N/A"),
                        "chunk_id": doc.metadata.get("chunk_id", "N/A"),
                        "full_content": doc.page_content,
                        "url": f"{BASE_URL.rstrip('/')}/{slug.lstrip('/')}" if slug != "N/A" else None
                    }
                    
                    response.sources.append(source_info)
                    
                    # Extract individual components for easy access
                    if title != "N/A":
                        response.titles.append(title)
                    if slug != "N/A":
                        response.slugs.append(slug)
                    response.descriptions.append(source_info["description"])
                
                # Check if user requested blog links specifically
                link_request_keywords = [
                    "blog", "blogs", "link", "links", "post", "posts", 
                    "read more", "blog post", "give me links", "give me blog links", 
                    "show me links", "sources", "articles", "show me articles"
                ]
                
                query_lower = query.lower()
                is_link_request = any(keyword in query_lower for keyword in link_request_keywords)
                
                if is_link_request:
                    # Build blog links response
                    blog_links_text = ["Here are the related blog posts I found:\n"]
                    
                    for source in response.sources:
                        if source["url"]:
                            blog_links_text.append(
                                f"• **{source['title']}**\n"
                                f"  {source['description']}\n"
                                f"  🔗 Link: {source['url']}\n"
                            )
                            
                            response.blog_links.append({
                                "title": source["title"],
                                "url": source["url"],
                                "description": source["description"],
                                "slug": source["slug"]
                            })
                    
                    # Override the response with blog links format if specifically requested
                    if response.blog_links:
                        blog_response = "\n".join(blog_links_text)
                        response.response = blog_response
                        
                        # Translate blog links response if needed
                        if response.language_detected and response.language_detected != "en":
                            try:
                                response.translated_response = await translate_text_async(
                                    blog_response, src="en", dest=response.language_detected
                                )
                            except Exception as e:
                                logger.error(f"Translation of blog links failed: {e}")
                                response.translated_response = blog_response

            # 8) Determine router decision
            response.router_decision = "rag" if response.sources else "llm"

            # 9) Update thread info
            if thread_id in self.active_threads:
                self.active_threads[thread_id]["message_count"] += 1
                self.active_threads[thread_id]["last_activity"] = datetime.now()
                # Note: Conversation history is automatically managed by LangGraph's checkpointing system

            # 10) Set processing time and final metadata
            response.processing_time = (datetime.now() - start_time).total_seconds()
            response.metadata.update({
                "original_query": query,
                "english_query": english_query,
                "sanitized_query": sanitized,
                "total_sources": len(response.sources),
                "total_blog_links": len(response.blog_links),
                "is_link_request": len(response.blog_links) > 0
            })
            
            logger.info(f"Processing completed in {response.processing_time:.2f}s")
            return response

        except Exception as e:
            logger.error(f"Error in unified agent processing: {e}")
            response.blocked = True
            response.response = f"An error occurred while processing your request: {str(e)}"
            response.translated_response = response.response
            response.processing_time = (datetime.now() - start_time).total_seconds()
            response.metadata["error"] = str(e)
            return response

    async def stream_response(
        self,
        query: str,
        thread_id: Optional[str] = None,
        language: Optional[str] = None,
        conversation_history: Optional[List[str]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream the response processing in real-time using Server-Sent Events format.
        
        Yields progress updates and final complete response data.
        """
        try:
            # Send initial processing start event
            yield {
                "type": "start",
                "content": "Starting to process your request...",
                "metadata": {"timestamp": datetime.now().isoformat()}
            }

            # Process the query with real-time streaming
            async for event in self._stream_agent_processing(query, thread_id, language, conversation_history):
                yield event

        except Exception as e:
            logger.error(f"Error in stream_response: {e}")
            yield {
                "type": "error",
                "content": f"An error occurred: {str(e)}",
                "metadata": {
                    "error_type": "stream_error",
                    "timestamp": datetime.now().isoformat()
                }
            }

    async def _stream_agent_processing(
        self,
        query: str,
        thread_id: Optional[str] = None,
        language: Optional[str] = None,
        conversation_history: Optional[List[str]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream the actual agent processing with real-time updates.
        """
        try:
            # Ensure we have a thread ID
            if not thread_id:
                thread_id = self.get_or_create_thread()
            
            config = {"configurable": {"thread_id": thread_id}}
            
            # 1) Language detection
            yield {
                "type": "processing",
                "content": "Detecting language...",
                "metadata": {"step": "language_detection"}
            }
            
            try:
                from rag_agent import llm_detect_language_and_intent
                detect_info = llm_detect_language_and_intent(query)
                detected_lang = detect_info.get("language", "en")
                confidence = detect_info.get("confidence", 0.0)
                
                yield {
                    "type": "language_detected",
                    "content": f"Language detected: {detected_lang}",
                    "metadata": {
                        "language": detected_lang,
                        "confidence": confidence
                    }
                }
            except Exception as e:
                logger.error(f"Language detection failed: {e}")
                detected_lang = language or "en"
                confidence = 0.0

            # 2) Translation if needed
            if detected_lang and detected_lang != "en":
                yield {
                    "type": "processing",
                    "content": f"Translating from {detected_lang} to English...",
                    "metadata": {"step": "translation"}
                }
                
                try:
                    from rag_agent import translate_text_async
                    english_query = await translate_text_async(query, src=detected_lang, dest="en")
                    yield {
                        "type": "translation_complete",
                        "content": "Translation completed",
                        "metadata": {"original_language": detected_lang}
                    }
                except Exception as e:
                    logger.error(f"Translation failed: {e}")
                    english_query = query
            else:
                english_query = query

            # 3) Guardrail check
            yield {
                "type": "processing",
                "content": "Running content moderation...",
                "metadata": {"step": "guardrails"}
            }
            
            try:
                from rag_agent import llm_input_guardrails, GUARDRAIL_SENTINEL, translate_text_async
                sanitized = llm_input_guardrails(english_query, history=conversation_history or [])
                
                if isinstance(sanitized, str) and sanitized.strip() == GUARDRAIL_SENTINEL:
                    # Translate guardrail message if needed
                    blocked_message = GUARDRAIL_SENTINEL
                    if detected_lang and detected_lang != "en":
                        try:
                            blocked_message = await translate_text_async(GUARDRAIL_SENTINEL, src="en", dest=detected_lang)
                        except Exception as e:
                            logger.error(f"Translation of guardrail failed: {e}")
                            blocked_message = GUARDRAIL_SENTINEL
                    
                    yield {
                        "type": "blocked",
                        "content": blocked_message,
                        "metadata": {
                            "language_detected": detected_lang,
                            "blocked": True,
                            "thread_id": thread_id
                        }
                    }
                    return
                    
            except Exception as e:
                logger.error(f"Guardrail check failed: {e}")
                sanitized = english_query

            # 4) Route query
            yield {
                "type": "processing",
                "content": "Determining processing method...",
                "metadata": {"step": "routing"}
            }
            
            # Import the graph and run it with streaming
            from rag_agent import graph
            
            # Build initial state
            initial_state = {
                "messages": [HumanMessage(content=sanitized)],
                "query": sanitized,
                "retrieved_docs": [],
                "conversation_history": conversation_history or []
            }
            
            # 5) Run the graph and get the response
            yield {
                "type": "routing",
                "content": "Generating response...",
                "metadata": {"step": "generation"}
            }
            
            # Run the graph - this will complete the processing
            final_state = await graph.ainvoke(initial_state, config=config)
            english_answer = final_state["messages"][-1].content
            
            # Stream the complete response immediately (no artificial word-by-word delay)
            yield {
                "type": "response",
                "content": english_answer,
                "metadata": {"streaming": True, "complete": True}
            }
            
            # 6) Translate back if needed
            if detected_lang and detected_lang != "en":
                yield {
                    "type": "processing",
                    "content": f"Translating response to {detected_lang}...",
                    "metadata": {"step": "translation_back"}
                }
                
                try:
                    from rag_agent import translate_text_async
                    translated_response = await translate_text_async(english_answer, src="en", dest=detected_lang)
                    
                    yield {
                        "type": "translation_complete",
                        "content": "Translation completed",
                        "metadata": {"target_language": detected_lang}
                    }
                except Exception as e:
                    logger.error(f"Translation back failed: {e}")
                    translated_response = english_answer
            else:
                translated_response = english_answer

            # 7) Process sources
            sources = []
            if final_state.get("retrieved_docs"):
                for doc in final_state["retrieved_docs"]:
                    source_info = {
                        "title": doc.metadata.get("title", "N/A"),
                        "slug": doc.metadata.get("slug", "N/A"),
                        "excerpt": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                        "source": doc.metadata.get("source", "N/A"),
                        "chunk_id": doc.metadata.get("chunk_id", "N/A"),
                        "full_content": doc.page_content
                    }
                    sources.append(source_info)

            # 8) Send final complete response with sources
            yield {
                "type": "complete",
                "content": {
                    "response": translated_response,
                    "original_response": english_answer,
                    "sources": sources,
                    "language_detected": detected_lang,
                    "thread_id": thread_id
                },
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "status": "completed",
                    "sources_count": len(sources)
                }
            }

        except Exception as e:
            logger.error(f"Error in _stream_agent_processing: {e}")
            yield {
                "type": "error",
                "content": f"Processing error: {str(e)}",
                "metadata": {
                    "error_type": "processing_error",
                    "timestamp": datetime.now().isoformat()
                }
            }

# Global handler instance
unified_handler = UnifiedRAGHandler()