# RAG Chatbot - Execution Workflow & Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RAG Chatbot Architecture                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                │
│  │   User Input │────▶│ Query Processing│──▶│  ChromaDB    │                │
│  │   (CLI)      │     │               │     │  Vector DB   │                │
│  └──────────────┘     └──────────────┘     └──────┬───────┘                │
│         │                                          │                        │
│         │                                          ▼                        │
│         │                              ┌─────────────────────┐             │
│         │                              │  Document Retrieval │             │
│         │                              │  (Top 3 chunks)     │             │
│         │                              └──────────┬──────────┘             │
│         │                                         │                        │
│         ▼                                         ▼                        │
│  ┌──────────────────────────────────────────────────────────┐             │
│  │              Context + System Prompt Builder              │             │
│  │  ┌────────────────────────────────────────────────────┐  │             │
│  │  │ System Prompt:                                    │  │             │
│  │  │ - Knowledge base context (retrieved docs)         │  │             │
│  │  │ - STRICT instructions: ONLY use context           │  │             │
│  │  │ - Refuse if answer not in context                 │  │             │
│  │  └────────────────────────────────────────────────────┘  │             │
│  └────────────────────────────┬─────────────────────────────┘             │
│                               │                                            │
│                               ▼                                            │
│  ┌──────────────────────────────────────────────────────────┐             │
│  │              DashScope/Qwen LLM API                       │             │
│  │  - Model: qwen-plus (or qwen-turbo for speed)            │             │
│  │  - Temperature: 0.0 (deterministic)                      │             │
│  │  - Max tokens: 500                                       │             │
│  │  - Streaming: Enabled                                    │             │
│  └────────────────────────────┬─────────────────────────────┘             │
│                               │                                            │
│                               ▼                                            │
│  ┌──────────────────────────────────────────────────────────┐             │
│  │              Response (Streaming)                         │             │
│  │  - Word-by-word display                                  │             │
│  │  - Stored in conversation history                        │             │
│  └────────────────────────────┬─────────────────────────────┘             │
│                               │                                            │
│                               ▼                                            │
│  ┌──────────────────────────────────────────────────────────┐             │
│  │              User Display (Terminal)                      │             │
│  └──────────────────────────────────────────────────────────┘             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Execution Workflow

### 1. Initialization Phase
```
┌─────────────┐
│   Start     │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│ Load .env variables │
│ - DASHSCOPE_API_KEY │
│ - DASHSCOPE_BASE_URL│
│ - LLM_MODEL         │
│ - EMBEDDING_MODEL   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Initialize Clients  │
│ - OpenAI client     │
│ - ChromaDB client   │
│ - Embedding function│
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Load/Create         │
│ ChromaDB Collection │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Display KB Status   │
│ (Document count)    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Wait for Input    │
└─────────────────────┘
```

### 2. Query Processing Phase
```
┌─────────────────────┐
│   User Enters Query │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Check for Commands  │
│ (/help, /upload,    │
│  /clear, /count)    │
└──────────┬──────────┘
           │
    ┌──────┴──────┐
    │             │
    ▼             ▼
┌────────┐   ┌─────────────────┐
│Command │   │ Regular Query   │
│Execute │   │ (Proceed to RAG)│
└────────┘   └────────┬────────┘
                      │
                      ▼
           ┌─────────────────────┐
           │ Display "Thinking..."│
           └──────────┬──────────┘
                      │
                      ▼
           ┌─────────────────────┐
           │ Embed Query using   │
           │ DashScope Embeddings│
           └──────────┬──────────┘
                      │
                      ▼
           ┌─────────────────────┐
           │ Search ChromaDB     │
           │ (Cosine Similarity) │
           │ n_results=3         │
           └──────────┬──────────┘
                      │
                      ▼
           ┌─────────────────────┐
           │ Retrieve Top Chunks │
           └──────────┬──────────┘
                      │
                      ▼
           ┌─────────────────────┐
           │ Build Context String│
           │ [Doc 1], [Doc 2]... │
           └─────────────────────┘
```

### 3. LLM Request Phase
```
┌─────────────────────────────────────────────────────────┐
│                 Build Messages Array                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Message 1: System Prompt (ALWAYS included)             │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Role: system                                    │   │
│  │                                                 │   │
│  │ You are a restricted enterprise knowledge       │   │
│  │ assistant with NO access to external info.      │   │
│  │                                                 │   │
│  │ CONTEXT FROM KNOWLEDGE BASE:                    │   │
│  │ [retrieved documents here]                      │   │
│  │                                                 │   │
│  │ ABSOLUTE RULES:                                 │   │
│  │ 1. ONLY use info from CONTEXT                   │   │
│  │ 2. If answer not in context → "I am not         │   │
│  │    authorized to comment on this topic."        │   │
│  │ 3. NEVER use external knowledge                 │   │
│  │ 4. NEVER answer general knowledge questions     │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  Message 2: User Query                                   │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Role: user                                      │   │
│  │ Content: [user's question]                      │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│              Send to DashScope API                       │
├─────────────────────────────────────────────────────────┤
│  Parameters:                                            │
│    - model: qwen-plus (or qwen-turbo)                   │
│    - messages: [system_prompt, user_message]            │
│    - stream: true                                       │
│    - temperature: 0.0 (strict/deterministic)            │
│    - max_tokens: 500                                    │
└─────────────────────────────────────────────────────────┘
```

### 4. Response Streaming Phase
```
┌─────────────────────┐
│ Clear "Thinking..." │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Stream Response     │
│ Chunk by Chunk      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Display Each Token  │
│ (Real-time output)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Store Full Response │
│ in Conversation     │
│ History (max 20)    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Wait for Next     │
│   User Input        │
└─────────────────────┘
```

## Key Design Decisions

### 1. Knowledge Base Enforcement
| Aspect | Implementation |
|--------|---------------|
| **System Prompt** | Refreshed on EVERY query (not just first) |
| **Context Inclusion** | Retrieved docs embedded in system prompt |
| **Strictness** | `temperature=0.0` for deterministic responses |
| **Fallback** | Hardcoded refusal message for out-of-scope queries |

### 2. Performance Optimizations
| Optimization | Impact |
|-------------|--------|
| **Streaming** | Perceived speed - tokens appear immediately |
| **n_results=3** | Reduced context size, faster processing |
| **max_tokens=500** | Prevents long, slow responses |
| **qwen-turbo option** | Faster model alternative |

### 3. Data Flow
```
User Query → Embedding → Vector Search → Context Assembly → 
LLM Request → Streaming Response → Display
```

## Command Reference

| Command | Action |
|---------|--------|
| `/upload` | Interactive file upload |
| `/add <path>` | Add file by path |
| `/add-text <text>` | Add raw text |
| `/count` | Show document count |
| `/clear` | Clear all documents |
| `/clear-history` | Clear conversation |
| `/status` | Show system status |
| `/help` | Show all commands |
| `quit` / `exit` / `q` | Exit chatbot |

## File Structure

```
rag_chatbot/
├── simple_rag_chromadb.py    # Main chatbot application
├── upload_to_kb.py           # Standalone upload script
├── reset_kb.py               # Knowledge base reset utility
├── chroma_db/                # Vector database storage
│   └── [collection data]
├── .env                      # Environment variables
└── requirements.txt          # Python dependencies
```

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `DASHSCOPE_API_KEY` | API authentication | Required |
| `DASHSCOPE_BASE_URL` | API endpoint | `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` |
| `LLM_MODEL` | Chat model | `qwen-plus` |
| `EMBEDDING_MODEL` | Embedding model | `text-embedding-v3` |

## Security Considerations

1. **API Key**: Stored in `.env`, never hardcoded
2. **Knowledge Isolation**: Strict prompt enforcement prevents data leakage
3. **No External Knowledge**: Model restricted to provided context only
