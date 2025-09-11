# FeedHope RAG Chatbot

If you'd like to use this chatbot, please email me for an API key 
Gmail = digupathania25@gmail.com

A Retrieval-Augmented Generation (RAG) chatbot built with FastAPI that helps users find information about food distribution opportunities and volunteer activities. The system combines vector search using Pinecone with LLM responses from Groq to provide contextual, accurate answers based on a knowledge base.

## 🌟 Features

- **RAG Architecture**: Combines retrieval from Pinecone vector database with Groq LLM generation
- **Smart Rate Limiting**: Multi-tier rate limiting (IP-based, session-based, token-based)
- **Conversation Memory**: Session-based conversation history for context-aware responses
- **Comprehensive Logging**: Structured logging with request tracking and performance metrics
- **Input Validation**: Sanitization and validation of user inputs for security
- **Health Monitoring**: Built-in health checks and debug endpoints
- **Production Ready**: Deployed on Render with Redis for caching and rate limiting

## 🚀 Live Demo

**Base URL**: https://rag-chatbot-uvwl.onrender.com/

## 📋 API Endpoints

### Health Check
```bash
GET /health
```

### Chat Endpoint
```bash
POST /chat
```

### Rate Limit Debug
```bash
GET /debug/rate-limit
```

## 🧪 Testing the API

### Basic Chat Request

```bash
curl -X POST "https://rag-chatbot-uvwl.onrender.com/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "How can I volunteer for food distribution?",
    "session_id": "test-session-123"
  }'
```

### Health Check
```bash
curl "https://rag-chatbot-uvwl.onrender.com/health"
```

### Rate Limit Status
```bash
curl "https://rag-chatbot-uvwl.onrender.com/debug/rate-limit?session_id=test-session-123"
```

## 🔧 Local Development

### Prerequisites
- Python 3.8+
- Redis server
- API keys for:
  - Groq (for LLM)
  - Google AI (for embeddings)
  - Pinecone (for vector database)

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd feedhope-rag-chatbot
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
Create a `.env` file:
```env
# LLM Configuration
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=mixtral-8x7b-32768

# Embedding Configuration  
GOOGLE_API_KEY=your_google_api_key
GOOGLE_EMBED_MODEL=models/text-embedding-004

# Vector Database
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
PINECONE_INDEX_NAME=your_index_name

# Redis (for rate limiting)
REDIS_URL=redis://localhost:6379/0

# API Security (optional)
API_KEY=your_optional_api_key

# Rate Limiting Configuration
TOP_K=4
SESSION_RPM=10
SESSION_RPD=500
SESSION_TPM=5000
IP_RPM=5
IP_RPD=200
IP_TPM=3000
```

5. **Start Redis server**
```bash
redis-server
```

6. **Run the application**
```bash
python app.py
```

The API will be available at `http://localhost:8000`

## 🏗️ Architecture

### Components

1. **FastAPI Application** (`app.py`)
   - Main web service handling HTTP requests
   - Request tracking middleware
   - Input validation and sanitization

2. **Google Embeddings Adapter** 
   - Custom adapter for Google's embedding API
   - Batch processing with fallback to individual requests
   - Retry logic and error handling

3. **Rate Limiter** (`rate_limiter.py`)
   - Multi-tier rate limiting (RPM, RPD, TPM)
   - Redis-based storage with automatic expiration
   - Separate limits for sessions vs IP addresses

4. **Logging System** (`basic_logging.py`)
   - Structured logging with multiple loggers
   - File and console output
   - Request correlation IDs

### Data Flow

1. **Request Processing**:
   - Middleware adds request tracking
   - Rate limiting validation
   - Input sanitization and validation

2. **Document Retrieval**:
   - Query embedding via Google API
   - Vector similarity search in Pinecone
   - Top-K document retrieval

3. **Response Generation**:
   - Context combination with conversation history
   - LLM generation via Groq
   - Response formatting with sources

4. **Session Management**:
   - Conversation history storage
   - Context window management
   - Memory cleanup

## 📊 Rate Limiting

The system implements sophisticated rate limiting:

### Limits by Identity Type

**Session-based** (when `session_id` provided):
- Requests per minute: 10
- Requests per day: 500
- Tokens per minute: 5,000

**IP-based** (when no session_id):
- Requests per minute: 5
- Requests per day: 200
- Tokens per minute: 3,000

### Global Limits
- Google Embedding API limits
- Token consumption tracking
- Automatic backoff and retry

## 🔒 Security Features

- **Input Sanitization**: HTML/script content filtering
- **Rate Limiting**: Multiple tiers of protection
- **API Key Authentication**: Optional API key protection
- **Request Validation**: Pydantic models with custom validators
- **Error Handling**: Secure error responses without sensitive data leakage

## 📝 Request/Response Examples

### Successful Chat Request
```bash
curl -X POST "https://rag-chatbot-uvwl.onrender.com/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are the requirements to volunteer?",
    "session_id": "user-456"
  }'
```

### With API Key (if enabled)
```bash
curl -X POST "https://rag-chatbot-uvwl.onrender.com/chat" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "message": "How do I sign up for food distribution events?",
    "session_id": "authenticated-user-789"
  }'
```

### Error Response (Rate Limited)
```json
{
  "detail": "Rate limit exceeded: 10 requests per minute (identity=user-123, type=session)"
}
```

## 🔍 Monitoring and Debugging

### Logs
The application generates structured logs in `app.log` with:
- Request/response tracking
- Performance metrics
- Error details with stack traces
- Rate limiting decisions

### Debug Endpoint
Check rate limit status:
```bash
curl "https://rag-chatbot-uvwl.onrender.com/debug/rate-limit?session_id=your-session"
```

### Health Check
Monitor system status:
```bash
curl "https://rag-chatbot-uvwl.onrender.com/health"
```

## 🚀 Deployment

The application is deployed on Render with:
- **Web Service**: FastAPI application
- **Redis Service**: For rate limiting and caching
- **Environment Variables**: Securely stored API keys
- **Health Checks**: Automatic monitoring
- **Scaling**: Automatic scaling based on traffic

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Troubleshooting

### Common Issues

**Rate Limiting Errors**: Check your usage with the debug endpoint and wait for limits to reset.

**Empty Responses**: Verify that your Pinecone index contains documents and embeddings.

**API Key Issues**: Ensure all required API keys are set in environment variables.

**Redis Connection**: Verify Redis is running and accessible at the configured URL.

### Support

For issues and questions, please check the logs first, then create an issue with:
- Request ID (from response headers)
- Error message
- Steps to reproduce
- Expected vs actual behavior
