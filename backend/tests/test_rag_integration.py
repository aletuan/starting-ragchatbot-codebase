import pytest
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from rag_system import RAGSystem
from config import Config
from models import Course, Lesson, CourseChunk


class TestRAGSystemIntegration:
    """End-to-end integration tests for RAG System"""

    @pytest.fixture
    def temp_chroma_path(self):
        """Create temporary ChromaDB path for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def test_config(self, temp_chroma_path):
        """Test configuration with temporary ChromaDB path"""
        config = Config()
        config.CHROMA_PATH = temp_chroma_path
        config.ANTHROPIC_API_KEY = "test_key"
        return config

    @pytest.fixture
    def rag_system_with_data(self, test_config):
        """RAG system with sample data loaded"""
        rag = RAGSystem(test_config)
        
        # Add test course data
        test_course = Course(
            title="Test AI Fundamentals",
            course_link="https://example.com/ai-course",
            instructor="Dr. Test",
            lessons=[
                Lesson(lesson_number=0, title="Introduction", lesson_link="https://example.com/lesson0"),
                Lesson(lesson_number=1, title="Neural Networks", lesson_link="https://example.com/lesson1"),
                Lesson(lesson_number=2, title="Deep Learning", lesson_link="https://example.com/lesson2")
            ]
        )
        
        test_chunks = [
            CourseChunk(
                content="Artificial intelligence is the simulation of human intelligence by machines. It includes machine learning, natural language processing, and computer vision.",
                course_title="Test AI Fundamentals",
                lesson_number=0,
                chunk_index=0
            ),
            CourseChunk(
                content="Neural networks are computational models inspired by biological neural networks. They consist of interconnected nodes that process information.",
                course_title="Test AI Fundamentals", 
                lesson_number=1,
                chunk_index=1
            ),
            CourseChunk(
                content="Deep learning uses neural networks with multiple layers to learn complex patterns. It has revolutionized computer vision and natural language processing.",
                course_title="Test AI Fundamentals",
                lesson_number=2, 
                chunk_index=2
            )
        ]
        
        rag.vector_store.add_course_metadata(test_course)
        rag.vector_store.add_course_content(test_chunks)
        
        return rag

    @patch('ai_generator.anthropic')
    def test_content_specific_query_triggers_search(self, mock_anthropic, rag_system_with_data):
        """Test that content-specific queries trigger CourseSearchTool"""
        # Setup mock client to simulate tool use response
        mock_client = Mock()
        
        # Mock initial response that uses search tool
        mock_initial_response = Mock()
        mock_initial_response.stop_reason = "tool_use"
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.input = {"query": "neural networks"}
        mock_tool_content.id = "tool_123"
        mock_initial_response.content = [mock_tool_content]
        
        # Mock final response after tool execution
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Neural networks are computational models inspired by biological neural networks. They process information through interconnected nodes and are fundamental to deep learning systems.")]
        
        mock_client.messages.create.side_effect = [mock_initial_response, mock_final_response]
        mock_anthropic.Anthropic.return_value = mock_client
        
        response, sources = rag_system_with_data.query("How do neural networks work?")
        
        # Verify response was generated
        assert "Neural networks are computational models" in response
        
        # Verify search tool was used (2 API calls: initial + after tool)
        assert mock_client.messages.create.call_count == 2
        
        # Verify sources were returned
        assert len(sources) > 0
        assert sources[0]["text"] == "Test AI Fundamentals - Lesson 1"

    @patch('ai_generator.anthropic')
    def test_general_knowledge_query_no_tools(self, mock_anthropic, rag_system_with_data):
        """Test that general knowledge queries don't trigger tools"""
        # Setup mock client for direct response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="Paris is the capital of France.")]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client
        
        response, sources = rag_system_with_data.query("What is the capital of France?")
        
        # Verify direct response
        assert response == "Paris is the capital of France."
        
        # Verify only one API call (no tool execution)
        assert mock_client.messages.create.call_count == 1
        
        # Verify no sources returned
        assert len(sources) == 0

    @patch('ai_generator.anthropic')
    def test_course_outline_query_triggers_outline_tool(self, mock_anthropic, rag_system_with_data):
        """Test that course outline queries trigger CourseOutlineTool"""
        # Setup mock client to simulate outline tool use
        mock_client = Mock()
        
        # Mock initial response that uses outline tool
        mock_initial_response = Mock()
        mock_initial_response.stop_reason = "tool_use"
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "get_course_outline"
        mock_tool_content.input = {"course_title": "Test AI Fundamentals"}
        mock_tool_content.id = "tool_456"
        mock_initial_response.content = [mock_tool_content]
        
        # Mock final response after tool execution
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="The Test AI Fundamentals course covers: Introduction (Lesson 0), Neural Networks (Lesson 1), and Deep Learning (Lesson 2).")]
        
        mock_client.messages.create.side_effect = [mock_initial_response, mock_final_response]
        mock_anthropic.Anthropic.return_value = mock_client
        
        response, sources = rag_system_with_data.query("What lessons are in the AI Fundamentals course?")
        
        # Verify outline response
        assert "Test AI Fundamentals course covers" in response
        assert "Introduction (Lesson 0)" in response
        
        # Verify outline tool was used
        assert mock_client.messages.create.call_count == 2

    @patch('ai_generator.anthropic')
    def test_course_filtering_query(self, mock_anthropic, rag_system_with_data):
        """Test queries with course name filtering"""
        mock_client = Mock()
        
        # Mock tool use for filtered search
        mock_initial_response = Mock()
        mock_initial_response.stop_reason = "tool_use"
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.input = {"query": "deep learning", "course_name": "AI Fundamentals"}
        mock_tool_content.id = "tool_789"
        mock_initial_response.content = [mock_tool_content]
        
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Deep learning uses neural networks with multiple layers to learn complex patterns.")]
        
        mock_client.messages.create.side_effect = [mock_initial_response, mock_final_response]
        mock_anthropic.Anthropic.return_value = mock_client
        
        response, sources = rag_system_with_data.query("Tell me about deep learning in the AI Fundamentals course")
        
        assert "Deep learning uses neural networks" in response

    @patch('ai_generator.anthropic')
    def test_lesson_specific_query(self, mock_anthropic, rag_system_with_data):
        """Test queries targeting specific lessons"""
        mock_client = Mock()
        
        # Mock tool use for lesson-specific search
        mock_initial_response = Mock()
        mock_initial_response.stop_reason = "tool_use"
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.input = {"query": "introduction", "lesson_number": 0}
        mock_tool_content.id = "tool_000"
        mock_initial_response.content = [mock_tool_content]
        
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="The introduction covers the basics of artificial intelligence and its applications.")]
        
        mock_client.messages.create.side_effect = [mock_initial_response, mock_final_response]
        mock_anthropic.Anthropic.return_value = mock_client
        
        response, sources = rag_system_with_data.query("What is covered in lesson 0?")
        
        assert "introduction covers the basics" in response

    @patch('ai_generator.anthropic')
    def test_conversation_history_integration(self, mock_anthropic, rag_system_with_data):
        """Test that conversation history is properly maintained and used"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="Based on our previous discussion about neural networks, here's more detail.")]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client
        
        session_id = "test_session_123"
        
        # First query
        rag_system_with_data.query("What are neural networks?", session_id=session_id)
        
        # Second query should include history
        response, sources = rag_system_with_data.query("Tell me more about that", session_id=session_id)
        
        # Verify history was included in system prompt
        call_args = mock_client.messages.create.call_args
        system_content = call_args[1]["system"]
        assert "Previous conversation:" in system_content

    @patch('ai_generator.anthropic')  
    def test_session_isolation(self, mock_anthropic, rag_system_with_data):
        """Test that different sessions maintain separate conversation histories"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="Response")]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client
        
        # Query in session 1
        rag_system_with_data.query("First session query", session_id="session_1")
        
        # Query in session 2 should not have session 1 history
        rag_system_with_data.query("Second session query", session_id="session_2")
        
        # Get history for each session
        history_1 = rag_system_with_data.session_manager.get_conversation_history("session_1")
        history_2 = rag_system_with_data.session_manager.get_conversation_history("session_2")
        
        assert "First session query" in history_1
        assert "First session query" not in history_2
        assert "Second session query" in history_2

    @patch('ai_generator.anthropic')
    def test_empty_search_results_handling(self, mock_anthropic, rag_system_with_data):
        """Test handling of queries that return no search results"""
        mock_client = Mock()
        
        # Mock tool use that returns empty results
        mock_initial_response = Mock()
        mock_initial_response.stop_reason = "tool_use"
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.input = {"query": "quantum computing"}
        mock_tool_content.id = "tool_empty"
        mock_initial_response.content = [mock_tool_content]
        
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="I don't have specific information about quantum computing in the available courses.")]
        
        mock_client.messages.create.side_effect = [mock_initial_response, mock_final_response]
        mock_anthropic.Anthropic.return_value = mock_client
        
        response, sources = rag_system_with_data.query("Tell me about quantum computing")
        
        assert "don't have specific information" in response
        assert len(sources) == 0

    @patch('ai_generator.anthropic')
    def test_course_not_found_error_handling(self, mock_anthropic, rag_system_with_data):
        """Test handling when specified course is not found"""
        mock_client = Mock()
        
        # Mock tool use for non-existent course
        mock_initial_response = Mock()
        mock_initial_response.stop_reason = "tool_use"
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.input = {"query": "basics", "course_name": "Nonexistent Course"}
        mock_tool_content.id = "tool_error"
        mock_initial_response.content = [mock_tool_content]
        
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="I couldn't find a course matching 'Nonexistent Course'.")]
        
        mock_client.messages.create.side_effect = [mock_initial_response, mock_final_response]
        mock_anthropic.Anthropic.return_value = mock_client
        
        response, sources = rag_system_with_data.query("Tell me the basics from the Nonexistent Course")
        
        assert "couldn't find a course matching" in response

    def test_course_analytics(self, rag_system_with_data):
        """Test course analytics functionality"""
        analytics = rag_system_with_data.get_course_analytics()
        
        assert "total_courses" in analytics
        assert "course_titles" in analytics
        assert analytics["total_courses"] >= 1
        assert "Test AI Fundamentals" in analytics["course_titles"]

    @patch('ai_generator.anthropic')
    def test_source_attribution_with_links(self, mock_anthropic, rag_system_with_data):
        """Test that sources include lesson links when available"""
        mock_client = Mock()
        
        # Mock successful search with results
        mock_initial_response = Mock()
        mock_initial_response.stop_reason = "tool_use"
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.input = {"query": "neural networks"}
        mock_tool_content.id = "tool_links"
        mock_initial_response.content = [mock_tool_content]
        
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Neural networks information from the course.")]
        
        mock_client.messages.create.side_effect = [mock_initial_response, mock_final_response]
        mock_anthropic.Anthropic.return_value = mock_client
        
        response, sources = rag_system_with_data.query("Explain neural networks")
        
        # Verify sources have proper structure
        assert len(sources) > 0
        source = sources[0]
        assert "text" in source
        assert "url" in source
        assert "Test AI Fundamentals" in source["text"]
        # Should have lesson link from test data
        assert source["url"] is not None

    def test_query_without_session_id(self, rag_system_with_data):
        """Test queries work without session ID (no conversation history)"""
        with patch('ai_generator.anthropic') as mock_anthropic:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.stop_reason = "end_turn"
            mock_response.content = [Mock(text="Response without session")]
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.Anthropic.return_value = mock_client
            
            response, sources = rag_system_with_data.query("Test query")
            
            assert response == "Response without session"
            
            # Verify no conversation history was passed
            call_args = mock_client.messages.create.call_args
            system_content = call_args[1]["system"]
            assert "Previous conversation:" not in system_content

    @patch('ai_generator.anthropic')
    def test_multiple_consecutive_queries_same_session(self, mock_anthropic, rag_system_with_data):
        """Test multiple queries in the same session build conversation history"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="Response")]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client
        
        session_id = "multi_query_session"
        
        # Multiple queries
        queries = [
            "What is machine learning?",
            "How does it relate to AI?",
            "Can you give examples?"
        ]
        
        for query in queries:
            rag_system_with_data.query(query, session_id=session_id)
        
        # Check that conversation history contains all exchanges
        history = rag_system_with_data.session_manager.get_conversation_history(session_id)
        
        for query in queries:
            assert query in history

    def test_add_course_document_integration(self, test_config):
        """Test adding course documents to the RAG system"""
        rag = RAGSystem(test_config)
        
        # Create a temporary test document
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("""Course Title: Integration Test Course
Course Link: https://example.com/integration
Course Instructor: Test Instructor

Lesson 1: Basic Concepts
This lesson covers the fundamental concepts.

Lesson 2: Advanced Topics  
This lesson explores advanced topics in detail.""")
            temp_file = f.name
        
        try:
            # Add the document
            course, chunk_count = rag.add_course_document(temp_file)
            
            assert course is not None
            assert course.title == "Integration Test Course"
            assert chunk_count > 0
            
            # Verify course was added to vector store
            analytics = rag.get_course_analytics()
            assert "Integration Test Course" in analytics["course_titles"]
            
        finally:
            os.unlink(temp_file)

    def test_tool_manager_integration(self, rag_system_with_data):
        """Test that tool manager is properly integrated"""
        # Verify tools are registered
        tool_definitions = rag_system_with_data.tool_manager.get_tool_definitions()
        tool_names = [tool["name"] for tool in tool_definitions]
        
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names
        assert len(tool_definitions) == 2

    @patch('ai_generator.anthropic')
    def test_api_error_propagation(self, mock_anthropic, rag_system_with_data):
        """Test that API errors are properly handled"""
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("API Error")
        mock_anthropic.Anthropic.return_value = mock_client
        
        # Should propagate the exception
        with pytest.raises(Exception, match="API Error"):
            rag_system_with_data.query("Test query")

    def test_source_reset_between_queries(self, rag_system_with_data):
        """Test that sources are reset between different queries"""
        with patch('ai_generator.anthropic') as mock_anthropic:
            mock_client = Mock()
            
            # First query with tool use
            mock_initial_response = Mock()
            mock_initial_response.stop_reason = "tool_use"
            mock_tool_content = Mock()
            mock_tool_content.type = "tool_use"
            mock_tool_content.name = "search_course_content"
            mock_tool_content.input = {"query": "neural networks"}
            mock_tool_content.id = "tool_1"
            mock_initial_response.content = [mock_tool_content]
            
            mock_final_response = Mock()
            mock_final_response.content = [Mock(text="First response")]
            
            # Second query without tool use
            mock_direct_response = Mock()
            mock_direct_response.stop_reason = "end_turn"
            mock_direct_response.content = [Mock(text="Second response")]
            
            mock_client.messages.create.side_effect = [
                mock_initial_response, mock_final_response,  # First query
                mock_direct_response  # Second query
            ]
            mock_anthropic.Anthropic.return_value = mock_client
            
            # First query should have sources
            response1, sources1 = rag_system_with_data.query("Tell me about neural networks")
            assert len(sources1) > 0
            
            # Second query should have empty sources
            response2, sources2 = rag_system_with_data.query("What is the capital of France?")
            assert len(sources2) == 0