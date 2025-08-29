# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup
```bash
# Install dependencies
uv sync

# Create environment file (required)
echo "ANTHROPIC_API_KEY=your_key_here" > .env
```

### Dependency Management
**Important**: This project uses `uv` for all Python package management. Always use `uv` commands instead of `pip`:
```bash
# Add new dependencies
uv add package-name

# Add dev dependencies  
uv add --dev package-name

# Remove dependencies
uv remove package-name

# Run Python commands
uv run python script.py
uv run pytest
uv run uvicorn app:app --reload
```

### Running the Application
```bash
# Quick start (recommended)
./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Code Quality Tools
```bash
# Format code automatically
./scripts/format.sh

# Check code quality (formatting, imports, linting)
./scripts/quality.sh

# Individual tools
uv run black backend/           # Format code
uv run isort backend/           # Sort imports  
uv run flake8 backend/          # Lint code

# Check without making changes
uv run black backend/ --check
uv run isort backend/ --check
```

### Development Server
- Web interface: http://localhost:8000
- API docs: http://localhost:8000/docs
- API endpoints: http://localhost:8000/api/

## Architecture Overview

This is a **Retrieval-Augmented Generation (RAG) system** for querying educational course materials. The architecture follows a layered approach with clear separation of concerns.

### Core Components Flow
```
Frontend (HTML/JS) → FastAPI → RAG System → AI Generator → Claude API
                                    ↓
                            Tool Manager → Search Tool → Vector Store (ChromaDB)
```

### Key Architectural Patterns

**RAG System (`rag_system.py`)** - Central orchestrator that coordinates:
- Document processing and chunking
- Vector storage management  
- AI generation with tool calling
- Session management for conversation history

**Tool-Based Search Architecture** - Uses Anthropic's tool calling feature:
- `search_tools.py` defines search capabilities as tools
- Claude API decides when to search based on user queries
- `CourseSearchTool` performs semantic search with course/lesson filtering
- Results are formatted with source attribution

**Document Processing Pipeline** (`document_processor.py`):
1. Parse structured course documents (Course Title/Link/Instructor format)
2. Extract lessons using regex patterns (`Lesson N: Title`)
3. Chunk text using sentence-based splitting with overlap
4. Add contextual metadata to each chunk for better retrieval

**Vector Storage Strategy** (`vector_store.py`):
- Uses ChromaDB with SentenceTransformers embeddings
- Separate collections for course metadata and content chunks
- Supports filtering by course name and lesson number
- Maintains document hierarchy for source attribution

### Configuration System

All settings centralized in `config.py` using environment variables:
- `ANTHROPIC_API_KEY` - Required for Claude API access
- `ANTHROPIC_MODEL` - Currently uses "claude-sonnet-4-20250514" 
- `EMBEDDING_MODEL` - Default "all-MiniLM-L6-v2"
- Chunk size (800), overlap (100), max results (5), max history (2)
- ChromaDB storage path ("./chroma_db")

### Data Models (`models.py`)

**Course Hierarchy**:
- `Course` → contains multiple `Lesson` objects
- `CourseChunk` → text chunks with course/lesson metadata
- Maintains relationship between content and its source

### Session Management

**Conversation Context** (`session_manager.py`):
- Tracks conversation history per session
- Limits history to prevent token overflow
- Provides context for follow-up questions

### Frontend Integration

**API Communication** (`frontend/script.js`):
- Single-page application with real-time chat interface
- Manages session state and loading states
- Displays responses with collapsible source attribution
- Handles errors gracefully with user feedback

## Document Format Expected

Course documents should follow this structure:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [instructor]

Lesson 0: [lesson title]
Lesson Link: [optional lesson url]
[lesson content...]

Lesson 1: [next lesson title]
[content continues...]
```

## Key Development Notes

- **ChromaDB Persistence**: Vector store persists to `./chroma_db` directory
- **Automatic Document Loading**: System loads documents from `./docs/` on startup
- **Tool Integration**: Search functionality implemented as Claude tools rather than direct vector queries
- **Error Handling**: Graceful fallbacks throughout the pipeline with user-friendly error messages
- **CORS Configuration**: Allows all origins for development (configured in `app.py`)
- **Static File Serving**: Frontend served directly by FastAPI with no-cache headers for development

## Important File Relationships

- `app.py` serves both API endpoints and static frontend files
- `rag_system.py` imports and coordinates all other backend modules
- Configuration flows from `config.py` through dependency injection
- `search_tools.py` bridges between AI generation and vector storage
- Frontend communicates only with FastAPI endpoints, never directly with Python modules