import pytest
import os
import sys
import tempfile
import shutil
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# Add the backend directory to sys.path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import Course, Lesson, CourseChunk, SourceWithLink
from vector_store import SearchResults
from config import Config
from fastapi.testclient import TestClient


@dataclass
class TestConfig:
    """Test configuration that doesn't require real API keys"""
    ANTHROPIC_API_KEY: str = "test_key"
    ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    MAX_RESULTS: int = 5
    MAX_HISTORY: int = 2
    CHROMA_PATH: str = "./test_chroma_db"


@pytest.fixture
def test_config():
    """Provides test configuration"""
    return TestConfig()


@pytest.fixture
def sample_lessons():
    """Sample lessons for testing"""
    return [
        Lesson(
            lesson_number=0,
            title="Introduction",
            lesson_link="https://example.com/lesson0"
        ),
        Lesson(
            lesson_number=1, 
            title="Basic Concepts",
            lesson_link="https://example.com/lesson1"
        ),
        Lesson(
            lesson_number=2,
            title="Advanced Topics",
            lesson_link="https://example.com/lesson2"
        )
    ]


@pytest.fixture  
def sample_course(sample_lessons):
    """Sample course for testing"""
    return Course(
        title="Test Course on AI",
        course_link="https://example.com/course",
        instructor="Test Instructor",
        lessons=sample_lessons
    )


@pytest.fixture
def sample_course_chunks():
    """Sample course chunks for testing"""
    return [
        CourseChunk(
            content="This is an introduction to AI concepts. We'll cover machine learning basics.",
            course_title="Test Course on AI",
            lesson_number=0,
            chunk_index=0
        ),
        CourseChunk(
            content="Neural networks are fundamental building blocks of deep learning systems.",
            course_title="Test Course on AI", 
            lesson_number=1,
            chunk_index=1
        ),
        CourseChunk(
            content="Advanced topics include transformer architectures and attention mechanisms.",
            course_title="Test Course on AI",
            lesson_number=2, 
            chunk_index=2
        )
    ]


@pytest.fixture
def sample_search_results():
    """Sample search results for testing"""
    return SearchResults(
        documents=[
            "This is an introduction to AI concepts. We'll cover machine learning basics.",
            "Neural networks are fundamental building blocks of deep learning systems."
        ],
        metadata=[
            {
                "course_title": "Test Course on AI",
                "lesson_number": 0,
                "chunk_index": 0
            },
            {
                "course_title": "Test Course on AI", 
                "lesson_number": 1,
                "chunk_index": 1
            }
        ],
        distances=[0.1, 0.2]
    )


@pytest.fixture
def empty_search_results():
    """Empty search results for testing"""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[]
    )


@pytest.fixture
def error_search_results():
    """Error search results for testing"""
    return SearchResults.empty("No course found matching 'nonexistent'")


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing"""
    mock = Mock()
    mock.search.return_value = SearchResults(
        documents=["Sample content"],
        metadata=[{"course_title": "Test Course", "lesson_number": 1}],
        distances=[0.1]
    )
    mock.get_lesson_link.return_value = "https://example.com/lesson1"
    return mock


@pytest.fixture
def mock_anthropic_response_with_tools():
    """Mock Anthropic response that uses tools"""
    mock_response = Mock()
    mock_response.stop_reason = "tool_use"
    
    # Create mock content block for tool use
    mock_tool_content = Mock()
    mock_tool_content.type = "tool_use"
    mock_tool_content.name = "search_course_content"
    mock_tool_content.input = {"query": "neural networks", "course_name": "AI"}
    mock_tool_content.id = "tool_123"
    
    mock_response.content = [mock_tool_content]
    return mock_response


@pytest.fixture
def mock_anthropic_response_direct():
    """Mock Anthropic response without tool use"""
    mock_response = Mock()
    mock_response.stop_reason = "end_turn"
    
    # Create mock content block for text response
    mock_text_content = Mock()
    mock_text_content.text = "This is a general knowledge answer about AI concepts."
    
    mock_response.content = [mock_text_content]
    return mock_response


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for testing"""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.content = [Mock(text="Test response")]
    mock_response.stop_reason = "end_turn"
    mock_client.messages.create.return_value = mock_response
    return mock_client


@pytest.fixture
def sample_sources():
    """Sample sources for testing UI integration"""
    return [
        {
            "text": "Test Course on AI - Lesson 0",
            "url": "https://example.com/lesson0"
        },
        {
            "text": "Test Course on AI - Lesson 1", 
            "url": "https://example.com/lesson1"
        }
    ]


# Test data for different query types
@pytest.fixture
def test_queries():
    """Various test queries for different scenarios"""
    return {
        "content_specific": "How do neural networks work in deep learning?",
        "course_specific": "What topics are covered in the AI course?", 
        "lesson_specific": "What is covered in lesson 1 of the AI course?",
        "general_knowledge": "What is the capital of France?",
        "course_outline": "What lessons are in the MCP course?"
    }


# API Testing Fixtures

@pytest.fixture
def temp_chroma_path():
    """Create temporary ChromaDB path for testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_app(temp_chroma_path):
    """Create test FastAPI app without static file mounting issues"""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import List, Optional
    from unittest.mock import Mock, patch
    
    # Create test app without static files
    app = FastAPI(title="Test Course Materials RAG System")
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Mock RAG system for testing
    mock_rag_system = Mock()
    mock_rag_system.query.return_value = ("Test response", [{"text": "Test source", "url": "https://example.com"}])
    mock_rag_system.get_course_analytics.return_value = {"total_courses": 1, "course_titles": ["Test Course"]}
    mock_rag_system.session_manager.create_session.return_value = "test_session_123"
    mock_rag_system.session_manager.clear_session.return_value = None
    
    # Pydantic models
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None
    
    class QueryResponse(BaseModel):
        answer: str
        sources: List[Dict[str, Any]]
        session_id: str
    
    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]
    
    # API endpoints
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            rag_system = app.state.mock_rag_system
            session_id = request.session_id or rag_system.session_manager.create_session()
            answer, sources = rag_system.query(request.query, session_id)
            
            # Convert sources to proper format
            source_objects = []
            for source in sources:
                if isinstance(source, dict):
                    source_objects.append(source)
                else:
                    source_objects.append({"text": str(source), "url": None})
            
            return QueryResponse(
                answer=answer,
                sources=source_objects,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            rag_system = app.state.mock_rag_system
            analytics = rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/sessions/{session_id}/clear")
    async def clear_session(session_id: str):
        try:
            rag_system = app.state.mock_rag_system
            rag_system.session_manager.clear_session(session_id)
            return {"message": "Session cleared successfully", "session_id": session_id}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/")
    async def root():
        return {"message": "Course Materials RAG System"}
    
    # Store mock for access in tests
    app.state.mock_rag_system = mock_rag_system
    return app


@pytest.fixture
def client(test_app):
    """Test client for API testing"""
    return TestClient(test_app)


@pytest.fixture
def api_test_data():
    """Test data for API testing"""
    return {
        "valid_query": {
            "query": "What is machine learning?",
            "session_id": "test_session_123"
        },
        "query_without_session": {
            "query": "Explain neural networks"
        },
        "invalid_query": {
            "invalid_field": "This should fail validation"
        },
        "empty_query": {
            "query": ""
        }
    }


@pytest.fixture
def mock_rag_system_with_error():
    """Mock RAG system that raises errors for testing error handling"""
    mock = Mock()
    mock.query.side_effect = Exception("Test error")
    mock.get_course_analytics.side_effect = Exception("Analytics error")
    mock.session_manager.clear_session.side_effect = Exception("Session error")
    return mock


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client with configurable responses"""
    mock_client = Mock()
    
    # Default response
    mock_response = Mock()
    mock_response.content = [Mock(text="Test AI response")]
    mock_response.stop_reason = "end_turn"
    mock_client.messages.create.return_value = mock_response
    
    return mock_client


@pytest.fixture
def sample_course_data():
    """Sample course data for comprehensive testing"""
    return {
        "course": Course(
            title="Advanced Python Programming",
            course_link="https://example.com/python-course",
            instructor="Dr. Python Expert",
            lessons=[
                Lesson(lesson_number=0, title="Introduction to Python", lesson_link="https://example.com/lesson0"),
                Lesson(lesson_number=1, title="Object-Oriented Programming", lesson_link="https://example.com/lesson1"),
                Lesson(lesson_number=2, title="Advanced Topics", lesson_link="https://example.com/lesson2")
            ]
        ),
        "chunks": [
            CourseChunk(
                content="Python is a high-level programming language known for its simplicity and readability.",
                course_title="Advanced Python Programming",
                lesson_number=0,
                chunk_index=0
            ),
            CourseChunk(
                content="Object-oriented programming (OOP) is a programming paradigm based on the concept of objects.",
                course_title="Advanced Python Programming",
                lesson_number=1,
                chunk_index=1
            ),
            CourseChunk(
                content="Advanced Python topics include decorators, context managers, and metaclasses.",
                course_title="Advanced Python Programming",
                lesson_number=2,
                chunk_index=2
            )
        ]
    }