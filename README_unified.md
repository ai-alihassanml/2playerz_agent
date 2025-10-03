# 2playerz RAG Agent - Unified API

## Overview

This is the unified version of the 2playerz RAG Agent API that streamlines all functionality into a single handler and provides Server-Sent Events (SSE) streaming instead of WebSockets.

## Key Features

- 🎯 **Unified Processing**: Single function handles all types of queries
- 🌊 **SSE Streaming**: Real-time responses without WebSocket complexity
- 🌍 **Multi-language Support**: Automatic detection and translation
- 📚 **Complete Metadata**: Sources, slugs, titles, descriptions, blog links
- 🛡️ **Built-in Guardrails**: Content filtering and safety
- 🔀 **Intelligent Routing**: Automatic RAG vs LLM decision making

## Architecture

### Files Structure

```
RAG_agent/
├── rag_agent.py              # Original RAG agent logic
├── unified_rag_handler.py    # NEW: Unified handler (main logic)
├── api_new.py               # NEW: Streamlined API with SSE
├── api.py                   # OLD: WebSocket-based API (deprecated)
├── example_usage.py         # NEW: Usage examples
└── README_unified.md        # This file
```

### Core Components

1. **UnifiedRAGHandler**: Main processing class that handles all query types
2. **UnifiedRAGResponse**: Structured response containing all data
3. **Server-Sent Events API**: Streaming without WebSocket complexity

## Usage

### 1. Start the API Server

```bash
python api_new.py
```

The API will be available at: `http://localhost:8000`

### 2. API Endpoints

#### Regular Chat (Non-streaming)
```
POST /chat
```

**Request:**
```json
{
    "message": "Tell me about the latest Call of Duty game",
    "language": "en",
    "thread_id": "optional-thread-id",
    "conversation_history": ["previous", "messages"]
}
```

**Response:**
```json
{
    "response": "Translated response for user",
    "original_response": "Original English response",
    "thread_id": "generated-or-provided-thread-id",
    "sources": [
        {
            "title": "Article Title",
            "slug": "article-slug",
            "excerpt": "Brief excerpt...",
            "description": "Full description",
            "url": "https://2playerz.de/p/article-slug",
            "source": "source-identifier",
            "full_content": "Complete article content"
        }
    ],
    "blog_links": [
        {
            "title": "Blog Post Title",
            "url": "https://2playerz.de/p/blog-slug",
            "description": "Blog description",
            "slug": "blog-slug"
        }
    ],
    "slugs": ["slug1", "slug2"],
    "titles": ["Title 1", "Title 2"],
    "descriptions": ["Description 1", "Description 2"],
    "language_detected": "en",
    "processing_time": 2.45,
    "router_decision": "rag",
    "blocked": false,
    "metadata": {
        "is_question": true,
        "language_confidence": 0.95,
        "total_sources": 3,
        "is_link_request": false
    }
}
```

#### Streaming Chat (SSE)
```
POST /chat/stream
```

**Request:** Same as regular chat

**Response:** Server-Sent Events stream:
```
data: {"type": "start", "content": "Starting to process...", "metadata": {...}}

data: {"type": "language_detected", "content": "Language detected: en", "metadata": {...}}

data: {"type": "routing", "content": "Using RAG processing", "metadata": {...}}

data: {"type": "chunk", "content": "Response text chunk...", "metadata": {...}}

data: {"type": "sources", "content": [...], "metadata": {...}}

data: {"type": "blog_links", "content": [...], "metadata": {...}}

data: {"type": "complete", "content": {...}, "metadata": {...}}

data: {"type": "end", "content": "Stream ended"}
```

### 3. Response Types

#### Regular Gaming Questions
- Provides comprehensive answers with sources
- Includes metadata from retrieved documents
- Automatic language translation

#### Blog Link Requests
When users ask for "blog links", "give me links", "sources", etc.:
- Special formatting for blog links
- Direct URLs to articles
- Enhanced link metadata

#### Multi-language Queries
- Automatic language detection
- Translation to English for processing
- Response translated back to user's language
- Both original and translated responses provided

### 4. Usage Examples

#### Python Example
```python
import asyncio
import aiohttp

async def chat_example():
    async with aiohttp.ClientSession() as session:
        request = {
            "message": "Tell me about Naughty Dog games",
            "language": "en"
        }
        
        async with session.post("http://localhost:8000/chat", json=request) as response:
            result = await response.json()
            print(f"Response: {result['response']}")
            print(f"Sources: {len(result['sources'])}")

asyncio.run(chat_example())
```

#### JavaScript Example (SSE)
```javascript
const response = await fetch('http://localhost:8000/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        message: 'Give me blog links about Death Stranding 2',
        language: 'en'
    })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    
    const chunk = decoder.decode(value);
    const lines = chunk.split('\n');
    
    for (const line of lines) {
        if (line.startsWith('data: ')) {
            const event = JSON.parse(line.slice(6));
            console.log(`${event.type}: ${event.content}`);
        }
    }
}
```

#### cURL Example
```bash
# Regular chat
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "What is 2playerz?", "language": "en"}'

# Streaming chat
curl -X POST "http://localhost:8000/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me about gaming news", "language": "en"}' \
  --no-buffer
```

## Special Features

### 1. Blog Link Detection
The system automatically detects when users request blog links using keywords:
- "blog", "blogs", "link", "links"
- "give me links", "show me links"
- "sources", "articles"
- "read more", "blog post"

### 2. Complete Metadata Extraction
Every response includes:
- **Slugs**: URL slugs for all sources
- **Titles**: Article/post titles
- **Descriptions**: Full descriptions or excerpts
- **URLs**: Direct links to content
- **Source metadata**: Additional context

### 3. Language Support
- Automatic detection of 15+ languages
- Translation to English for processing
- Response translated back to user's language
- Confidence scoring for language detection

### 4. Intelligent Processing
- **Router Decision**: Automatically chooses RAG vs LLM processing
- **Guardrails**: Content filtering and safety checks
- **Thread Management**: Conversation continuity
- **Error Handling**: Graceful degradation

## Testing

### Run Example Usage
```bash
# Install requirements
pip install aiohttp

# Run examples
python example_usage.py
```

### Test Endpoints
```bash
# Health check
curl http://localhost:8000/health

# API status
curl http://localhost:8000/status

# Simple test
curl http://localhost:8000/test/simple

# Stream test
curl http://localhost:8000/test/stream --no-buffer
```

## Migration from WebSocket API

### Old WebSocket Approach
- Complex connection management
- Client-specific state tracking
- Connection lifecycle handling
- Limited scalability

### New SSE Approach
- Simpler HTTP-based streaming
- Stateless request/response
- Better browser compatibility
- Easier to proxy and scale

### Migration Steps
1. Replace WebSocket connections with SSE endpoints
2. Update client code to handle SSE events
3. Use unified handler for consistent responses
4. Leverage complete response metadata

## Error Handling

### HTTP Status Codes
- `200`: Success
- `403`: Request blocked by guardrails
- `422`: Validation error
- `500`: Server error

### Error Response Format
```json
{
    "detail": "Error description",
    "type": "error_type",
    "metadata": {
        "timestamp": "2024-01-01T12:00:00",
        "error_context": "..."
    }
}
```

### Streaming Errors
```
data: {"type": "error", "content": "Error message", "metadata": {...}}
```

## Configuration

### Environment Variables
Ensure these are set in your `.env` file:
```
OPENAI_API_KEY=your_openai_key
HAGGINGFACEHUB_API_TOKEN=your_huggingface_token
```

### API Configuration
- Port: 8000 (configurable in `api_new.py`)
- CORS: Enabled for all origins (configure for production)
- Logging: INFO level

## Production Considerations

1. **CORS Configuration**: Set specific allowed origins
2. **Rate Limiting**: Implement request rate limiting
3. **Authentication**: Add API key authentication
4. **Monitoring**: Set up logging and metrics
5. **Scaling**: Consider load balancing for multiple instances

## Troubleshooting

### Common Issues
1. **API not starting**: Check environment variables
2. **No responses**: Verify FAISS index exists
3. **Translation errors**: Check internet connection
4. **Streaming issues**: Ensure client handles SSE correctly

### Debug Mode
Enable debug logging in `api_new.py`:
```python
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

When making changes:
1. Update the unified handler for core logic changes
2. Maintain backward compatibility in API responses
3. Add examples for new features
4. Update this README for any new functionality