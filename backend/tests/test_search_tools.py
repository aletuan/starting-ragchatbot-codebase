from unittest.mock import MagicMock, Mock, patch

import pytest
from search_tools import CourseOutlineTool, CourseSearchTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchTool:
    """Test cases for CourseSearchTool"""

    def test_get_tool_definition(self, mock_vector_store):
        """Test that tool definition is correctly structured"""
        tool = CourseSearchTool(mock_vector_store)
        definition = tool.get_tool_definition()

        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["type"] == "object"

        # Check required and optional parameters
        properties = definition["input_schema"]["properties"]
        assert "query" in properties
        assert "course_name" in properties
        assert "lesson_number" in properties
        assert definition["input_schema"]["required"] == ["query"]

    def test_execute_query_only(self, mock_vector_store, sample_search_results):
        """Test execute with query parameter only"""
        mock_vector_store.search.return_value = sample_search_results
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="neural networks")

        mock_vector_store.search.assert_called_once_with(
            query="neural networks", course_name=None, lesson_number=None
        )
        assert "Test Course on AI" in result
        assert len(tool.last_sources) == 2

    def test_execute_with_course_name(self, mock_vector_store, sample_search_results):
        """Test execute with query and course_name parameters"""
        mock_vector_store.search.return_value = sample_search_results
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="neural networks", course_name="AI Course")

        mock_vector_store.search.assert_called_once_with(
            query="neural networks", course_name="AI Course", lesson_number=None
        )
        assert "Test Course on AI" in result

    def test_execute_with_lesson_number(self, mock_vector_store, sample_search_results):
        """Test execute with query and lesson_number parameters"""
        mock_vector_store.search.return_value = sample_search_results
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="neural networks", lesson_number=1)

        mock_vector_store.search.assert_called_once_with(
            query="neural networks", course_name=None, lesson_number=1
        )
        assert "Lesson 1" in result

    def test_execute_with_all_parameters(
        self, mock_vector_store, sample_search_results
    ):
        """Test execute with all parameters"""
        mock_vector_store.search.return_value = sample_search_results
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(
            query="neural networks", course_name="AI Course", lesson_number=1
        )

        mock_vector_store.search.assert_called_once_with(
            query="neural networks", course_name="AI Course", lesson_number=1
        )
        assert "Test Course on AI" in result
        assert "Lesson 1" in result

    def test_execute_error_handling(self, mock_vector_store, error_search_results):
        """Test execute handles errors from vector store"""
        mock_vector_store.search.return_value = error_search_results
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="nonexistent course")

        assert "No course found matching 'nonexistent'" in result
        assert len(tool.last_sources) == 0

    def test_execute_empty_results(self, mock_vector_store, empty_search_results):
        """Test execute handles empty search results"""
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="obscure topic")

        assert "No relevant content found" in result
        assert len(tool.last_sources) == 0

    def test_execute_empty_results_with_filters(
        self, mock_vector_store, empty_search_results
    ):
        """Test execute handles empty results with filter information"""
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="topic", course_name="AI", lesson_number=3)

        assert "No relevant content found in course 'AI' in lesson 3" in result

    def test_format_results_basic(self, mock_vector_store):
        """Test _format_results with basic search results"""
        tool = CourseSearchTool(mock_vector_store)

        results = SearchResults(
            documents=["Content about neural networks"],
            metadata=[{"course_title": "AI Course", "lesson_number": 1}],
            distances=[0.1],
        )

        formatted = tool._format_results(results)

        assert "[AI Course - Lesson 1]" in formatted
        assert "Content about neural networks" in formatted

    def test_format_results_no_lesson_number(self, mock_vector_store):
        """Test _format_results with metadata missing lesson_number"""
        tool = CourseSearchTool(mock_vector_store)

        results = SearchResults(
            documents=["General course content"],
            metadata=[{"course_title": "AI Course"}],
            distances=[0.1],
        )

        formatted = tool._format_results(results)

        assert "[AI Course]" in formatted
        assert "General course content" in formatted

    def test_format_results_unknown_course(self, mock_vector_store):
        """Test _format_results with unknown course title"""
        tool = CourseSearchTool(mock_vector_store)

        results = SearchResults(
            documents=["Some content"], metadata=[{"lesson_number": 2}], distances=[0.1]
        )

        formatted = tool._format_results(results)

        assert "[unknown - Lesson 2]" in formatted

    def test_source_tracking_with_lesson_links(self, mock_vector_store):
        """Test that sources are correctly tracked with lesson links"""
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"
        tool = CourseSearchTool(mock_vector_store)

        results = SearchResults(
            documents=["Content with link"],
            metadata=[{"course_title": "AI Course", "lesson_number": 1}],
            distances=[0.1],
        )

        tool._format_results(results)

        assert len(tool.last_sources) == 1
        source = tool.last_sources[0]
        assert source["text"] == "AI Course - Lesson 1"
        assert source["url"] == "https://example.com/lesson1"

        mock_vector_store.get_lesson_link.assert_called_once_with("AI Course", 1)

    def test_source_tracking_without_lesson_links(self, mock_vector_store):
        """Test source tracking when lesson links are not available"""
        mock_vector_store.get_lesson_link.return_value = None
        tool = CourseSearchTool(mock_vector_store)

        results = SearchResults(
            documents=["Content without link"],
            metadata=[{"course_title": "AI Course", "lesson_number": 1}],
            distances=[0.1],
        )

        tool._format_results(results)

        assert len(tool.last_sources) == 1
        source = tool.last_sources[0]
        assert source["text"] == "AI Course - Lesson 1"
        assert source["url"] is None

    def test_source_tracking_no_lesson_number(self, mock_vector_store):
        """Test source tracking when no lesson number is available"""
        tool = CourseSearchTool(mock_vector_store)

        results = SearchResults(
            documents=["General content"],
            metadata=[{"course_title": "AI Course"}],
            distances=[0.1],
        )

        tool._format_results(results)

        assert len(tool.last_sources) == 1
        source = tool.last_sources[0]
        assert source["text"] == "AI Course"
        assert source["url"] is None

    def test_multiple_results_formatting(self, mock_vector_store):
        """Test formatting of multiple search results"""
        mock_vector_store.get_lesson_link.side_effect = [
            "https://example.com/lesson1",
            "https://example.com/lesson2",
        ]
        tool = CourseSearchTool(mock_vector_store)

        results = SearchResults(
            documents=["First content", "Second content"],
            metadata=[
                {"course_title": "AI Course", "lesson_number": 1},
                {"course_title": "AI Course", "lesson_number": 2},
            ],
            distances=[0.1, 0.2],
        )

        formatted = tool._format_results(results)

        assert "[AI Course - Lesson 1]" in formatted
        assert "[AI Course - Lesson 2]" in formatted
        assert "First content" in formatted
        assert "Second content" in formatted
        assert len(tool.last_sources) == 2


class TestCourseOutlineTool:
    """Test cases for CourseOutlineTool"""

    def test_get_tool_definition(self, mock_vector_store):
        """Test that outline tool definition is correctly structured"""
        tool = CourseOutlineTool(mock_vector_store)
        definition = tool.get_tool_definition()

        assert definition["name"] == "get_course_outline"
        assert "description" in definition
        assert definition["input_schema"]["required"] == ["course_title"]

    def test_execute_success(self, mock_vector_store):
        """Test successful course outline retrieval"""
        mock_outline = {
            "course_title": "AI Course",
            "course_link": "https://example.com/course",
            "instructor": "Dr. Smith",
            "lessons": [
                {"lesson_number": 1, "lesson_title": "Introduction"},
                {"lesson_number": 2, "lesson_title": "Advanced Topics"},
            ],
        }
        mock_vector_store.get_course_outline.return_value = mock_outline
        tool = CourseOutlineTool(mock_vector_store)

        result = tool.execute("AI Course")

        assert "Course: AI Course" in result
        assert "Link: https://example.com/course" in result
        assert "Instructor: Dr. Smith" in result
        assert "Lesson 1: Introduction" in result
        assert "Lesson 2: Advanced Topics" in result
        assert "2 total" in result

    def test_execute_course_not_found(self, mock_vector_store):
        """Test course not found scenario"""
        mock_vector_store.get_course_outline.return_value = None
        tool = CourseOutlineTool(mock_vector_store)

        result = tool.execute("Nonexistent Course")

        assert "Course 'Nonexistent Course' not found" in result

    def test_execute_with_error(self, mock_vector_store):
        """Test outline tool with error from vector store"""
        mock_vector_store.get_course_outline.return_value = {
            "error": "Database connection failed"
        }
        tool = CourseOutlineTool(mock_vector_store)

        result = tool.execute("AI Course")

        assert "Database connection failed" in result

    def test_format_outline_minimal(self, mock_vector_store):
        """Test formatting with minimal course information"""
        tool = CourseOutlineTool(mock_vector_store)
        outline = {"course_title": "Basic Course", "lessons": []}

        formatted = tool._format_outline(outline)

        assert "Course: Basic Course" in formatted
        assert "No lessons found" in formatted

    def test_format_outline_complete(self, mock_vector_store):
        """Test formatting with complete course information"""
        tool = CourseOutlineTool(mock_vector_store)
        outline = {
            "course_title": "Complete Course",
            "course_link": "https://example.com/course",
            "instructor": "Dr. Jones",
            "lessons": [
                {"lesson_number": 0, "lesson_title": "Introduction"},
                {"lesson_number": 1, "lesson_title": "Fundamentals"},
            ],
        }

        formatted = tool._format_outline(outline)

        assert "Course: Complete Course" in formatted
        assert "Link: https://example.com/course" in formatted
        assert "Instructor: Dr. Jones" in formatted
        assert "Lessons (2 total)" in formatted
        assert "Lesson 0: Introduction" in formatted
        assert "Lesson 1: Fundamentals" in formatted


class TestToolManager:
    """Test cases for ToolManager"""

    def test_register_tool(self, mock_vector_store):
        """Test tool registration"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)

        manager.register_tool(tool)

        assert "search_course_content" in manager.tools

    def test_register_tool_without_name(self, mock_vector_store):
        """Test registration fails for tool without name"""
        manager = ToolManager()

        # Create a mock tool with invalid definition
        mock_tool = Mock()
        mock_tool.get_tool_definition.return_value = {"description": "Test tool"}

        with pytest.raises(ValueError, match="Tool must have a 'name'"):
            manager.register_tool(mock_tool)

    def test_get_tool_definitions(self, mock_vector_store):
        """Test getting all tool definitions"""
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        outline_tool = CourseOutlineTool(mock_vector_store)

        manager.register_tool(search_tool)
        manager.register_tool(outline_tool)

        definitions = manager.get_tool_definitions()

        assert len(definitions) == 2
        tool_names = [def_["name"] for def_ in definitions]
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names

    def test_execute_tool(self, mock_vector_store, sample_search_results):
        """Test tool execution through manager"""
        mock_vector_store.search.return_value = sample_search_results
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        result = manager.execute_tool("search_course_content", query="neural networks")

        assert "Test Course on AI" in result

    def test_execute_nonexistent_tool(self):
        """Test executing tool that doesn't exist"""
        manager = ToolManager()

        result = manager.execute_tool("nonexistent_tool", query="test")

        assert "Tool 'nonexistent_tool' not found" in result

    def test_get_last_sources(self, mock_vector_store, sample_search_results):
        """Test getting sources from the last search"""
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson"

        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        # Execute a search to populate sources
        manager.execute_tool("search_course_content", query="test")
        sources = manager.get_last_sources()

        assert len(sources) == 2
        assert sources[0]["text"] == "Test Course on AI - Lesson 0"

    def test_get_last_sources_empty(self, mock_vector_store):
        """Test getting sources when none exist"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        sources = manager.get_last_sources()

        assert sources == []

    def test_reset_sources(self, mock_vector_store, sample_search_results):
        """Test resetting sources from all tools"""
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson"

        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        # Execute search to populate sources
        manager.execute_tool("search_course_content", query="test")
        assert len(manager.get_last_sources()) > 0

        # Reset sources
        manager.reset_sources()
        assert manager.get_last_sources() == []
