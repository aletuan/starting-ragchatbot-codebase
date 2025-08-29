import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock

import pytest

# Add the backend directory to sys.path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import Config
from models import Course, CourseChunk, Lesson, SourceWithLink
from vector_store import SearchResults


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
            lesson_link="https://example.com/lesson0",
        ),
        Lesson(
            lesson_number=1,
            title="Basic Concepts",
            lesson_link="https://example.com/lesson1",
        ),
        Lesson(
            lesson_number=2,
            title="Advanced Topics",
            lesson_link="https://example.com/lesson2",
        ),
    ]


@pytest.fixture
def sample_course(sample_lessons):
    """Sample course for testing"""
    return Course(
        title="Test Course on AI",
        course_link="https://example.com/course",
        instructor="Test Instructor",
        lessons=sample_lessons,
    )


@pytest.fixture
def sample_course_chunks():
    """Sample course chunks for testing"""
    return [
        CourseChunk(
            content="This is an introduction to AI concepts. We'll cover machine learning basics.",
            course_title="Test Course on AI",
            lesson_number=0,
            chunk_index=0,
        ),
        CourseChunk(
            content="Neural networks are fundamental building blocks of deep learning systems.",
            course_title="Test Course on AI",
            lesson_number=1,
            chunk_index=1,
        ),
        CourseChunk(
            content="Advanced topics include transformer architectures and attention mechanisms.",
            course_title="Test Course on AI",
            lesson_number=2,
            chunk_index=2,
        ),
    ]


@pytest.fixture
def sample_search_results():
    """Sample search results for testing"""
    return SearchResults(
        documents=[
            "This is an introduction to AI concepts. We'll cover machine learning basics.",
            "Neural networks are fundamental building blocks of deep learning systems.",
        ],
        metadata=[
            {"course_title": "Test Course on AI", "lesson_number": 0, "chunk_index": 0},
            {"course_title": "Test Course on AI", "lesson_number": 1, "chunk_index": 1},
        ],
        distances=[0.1, 0.2],
    )


@pytest.fixture
def empty_search_results():
    """Empty search results for testing"""
    return SearchResults(documents=[], metadata=[], distances=[])


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
        distances=[0.1],
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
        {"text": "Test Course on AI - Lesson 0", "url": "https://example.com/lesson0"},
        {"text": "Test Course on AI - Lesson 1", "url": "https://example.com/lesson1"},
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
        "course_outline": "What lessons are in the MCP course?",
    }
