import pytest
from unittest.mock import Mock, patch, MagicMock
from ai_generator import AIGenerator


class TestAIGenerator:
    """Test cases for AIGenerator integration with Anthropic API and tools"""

    def test_init(self):
        """Test AIGenerator initialization"""
        generator = AIGenerator("test_api_key", "test_model")
        
        assert generator.model == "test_model"
        assert generator.base_params["model"] == "test_model"
        assert generator.base_params["temperature"] == 0
        assert generator.base_params["max_tokens"] == 800

    @patch('ai_generator.anthropic')
    def test_generate_response_without_tools(self, mock_anthropic):
        """Test generating response without tool usage"""
        # Setup mock client and response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="This is a direct response")]
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client
        
        generator = AIGenerator("test_key", "test_model")
        
        result = generator.generate_response("What is AI?")
        
        assert result == "This is a direct response"
        mock_client.messages.create.assert_called_once()
        
        # Verify call parameters
        call_args = mock_client.messages.create.call_args
        assert call_args[1]["messages"][0]["content"] == "What is AI?"
        assert "tools" not in call_args[1]

    @patch('ai_generator.anthropic')
    def test_generate_response_with_conversation_history(self, mock_anthropic):
        """Test generating response with conversation history"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Response with history")]
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client
        
        generator = AIGenerator("test_key", "test_model")
        history = "Previous: What is ML?\nAssistant: Machine learning is..."
        
        result = generator.generate_response("What about deep learning?", conversation_history=history)
        
        assert result == "Response with history"
        
        # Verify system prompt includes history
        call_args = mock_client.messages.create.call_args
        system_content = call_args[1]["system"]
        assert "Previous conversation:" in system_content
        assert history in system_content

    @patch('ai_generator.anthropic')
    def test_generate_response_with_tools_no_tool_use(self, mock_anthropic):
        """Test response with tools available but no tool use"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="General knowledge response")]
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client
        
        generator = AIGenerator("test_key", "test_model")
        tools = [{"name": "search_course_content", "description": "Search tool"}]
        
        result = generator.generate_response("What is Python?", tools=tools)
        
        assert result == "General knowledge response"
        
        # Verify tools were passed to API
        call_args = mock_client.messages.create.call_args
        assert call_args[1]["tools"] == tools
        assert call_args[1]["tool_choice"] == {"type": "auto"}

    @patch('ai_generator.anthropic')
    def test_generate_response_with_tool_use(self, mock_anthropic):
        """Test response that uses tools"""
        # Setup mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool execution result"
        
        # Setup mock initial response with tool use
        mock_client = Mock()
        mock_initial_response = Mock()
        mock_initial_response.stop_reason = "tool_use"
        
        # Create mock tool use content block
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.input = {"query": "neural networks"}
        mock_tool_content.id = "tool_123"
        mock_initial_response.content = [mock_tool_content]
        
        # Setup mock final response after tool execution
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Final response after tool use")]
        
        # Configure client to return different responses for different calls
        mock_client.messages.create.side_effect = [mock_initial_response, mock_final_response]
        mock_anthropic.Anthropic.return_value = mock_client
        
        generator = AIGenerator("test_key", "test_model")
        tools = [{"name": "search_course_content", "description": "Search tool"}]
        
        result = generator.generate_response(
            "How do neural networks work?", 
            tools=tools, 
            tool_manager=mock_tool_manager
        )
        
        assert result == "Final response after tool use"
        
        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", 
            query="neural networks"
        )
        
        # Verify two API calls were made (initial + after tool execution)
        assert mock_client.messages.create.call_count == 2

    def test_handle_tool_execution_single_tool(self):
        """Test _handle_tool_execution with single tool call"""
        generator = AIGenerator("test_key", "test_model")
        
        # Setup mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search results about AI"
        
        # Create mock initial response
        mock_initial_response = Mock()
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.input = {"query": "AI"}
        mock_tool_content.id = "tool_456"
        mock_initial_response.content = [mock_tool_content]
        
        # Mock the final API call
        with patch.object(generator.client.messages, 'create') as mock_create:
            mock_final_response = Mock()
            mock_final_response.content = [Mock(text="Processed response")]
            mock_create.return_value = mock_final_response
            
            base_params = {
                "messages": [{"role": "user", "content": "Tell me about AI"}],
                "system": "System prompt"
            }
            
            result = generator._handle_tool_execution(
                mock_initial_response, 
                base_params, 
                mock_tool_manager
            )
            
            assert result == "Processed response"
            mock_tool_manager.execute_tool.assert_called_once_with("search_course_content", query="AI")
            
            # Verify the final API call structure
            final_call_args = mock_create.call_args
            messages = final_call_args[1]["messages"]
            
            # Should have original user message, assistant tool use, and tool results
            assert len(messages) == 3
            assert messages[0]["role"] == "user"
            assert messages[1]["role"] == "assistant"
            assert messages[2]["role"] == "user"
            assert messages[2]["content"][0]["type"] == "tool_result"

    def test_handle_tool_execution_multiple_tools(self):
        """Test _handle_tool_execution with multiple tool calls"""
        generator = AIGenerator("test_key", "test_model")
        
        # Setup mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "First tool result",
            "Second tool result"
        ]
        
        # Create mock initial response with two tool calls
        mock_initial_response = Mock()
        mock_tool_content1 = Mock()
        mock_tool_content1.type = "tool_use"
        mock_tool_content1.name = "search_course_content"
        mock_tool_content1.input = {"query": "AI"}
        mock_tool_content1.id = "tool_1"
        
        mock_tool_content2 = Mock()
        mock_tool_content2.type = "tool_use" 
        mock_tool_content2.name = "get_course_outline"
        mock_tool_content2.input = {"course_title": "AI Course"}
        mock_tool_content2.id = "tool_2"
        
        mock_initial_response.content = [mock_tool_content1, mock_tool_content2]
        
        # Mock the final API call
        with patch.object(generator.client.messages, 'create') as mock_create:
            mock_final_response = Mock()
            mock_final_response.content = [Mock(text="Combined response")]
            mock_create.return_value = mock_final_response
            
            base_params = {
                "messages": [{"role": "user", "content": "Tell me about the AI course"}],
                "system": "System prompt"
            }
            
            result = generator._handle_tool_execution(
                mock_initial_response,
                base_params,
                mock_tool_manager
            )
            
            assert result == "Combined response"
            assert mock_tool_manager.execute_tool.call_count == 2
            
            # Verify both tools were executed
            calls = mock_tool_manager.execute_tool.call_args_list
            assert calls[0][0] == ("search_course_content",)
            assert calls[0][1] == {"query": "AI"}
            assert calls[1][0] == ("get_course_outline",)
            assert calls[1][1] == {"course_title": "AI Course"}
            
            # Verify tool results structure
            final_call_args = mock_create.call_args
            tool_results = final_call_args[1]["messages"][2]["content"]
            assert len(tool_results) == 2
            assert tool_results[0]["tool_use_id"] == "tool_1"
            assert tool_results[1]["tool_use_id"] == "tool_2"

    def test_handle_tool_execution_non_tool_content(self):
        """Test _handle_tool_execution skips non-tool content blocks"""
        generator = AIGenerator("test_key", "test_model")
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool result"
        
        # Create mock response with mixed content types
        mock_initial_response = Mock()
        mock_text_content = Mock()
        mock_text_content.type = "text"
        
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.input = {"query": "test"}
        mock_tool_content.id = "tool_1"
        
        mock_initial_response.content = [mock_text_content, mock_tool_content]
        
        with patch.object(generator.client.messages, 'create') as mock_create:
            mock_final_response = Mock()
            mock_final_response.content = [Mock(text="Final response")]
            mock_create.return_value = mock_final_response
            
            base_params = {
                "messages": [{"role": "user", "content": "Test query"}],
                "system": "System prompt"
            }
            
            result = generator._handle_tool_execution(
                mock_initial_response,
                base_params,
                mock_tool_manager
            )
            
            # Should only execute one tool despite two content blocks
            mock_tool_manager.execute_tool.assert_called_once()
            
            # Should have one tool result despite two content blocks
            final_call_args = mock_create.call_args
            tool_results = final_call_args[1]["messages"][2]["content"]
            assert len(tool_results) == 1

    @patch('ai_generator.anthropic')
    def test_system_prompt_structure(self, mock_anthropic):
        """Test that system prompt is properly structured"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Response")]
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client
        
        generator = AIGenerator("test_key", "test_model")
        
        generator.generate_response("Test query")
        
        call_args = mock_client.messages.create.call_args
        system_content = call_args[1]["system"]
        
        # Verify key parts of system prompt
        assert "educational content" in system_content
        assert "Content Search Tool" in system_content
        assert "Course Outline Tool" in system_content
        assert "Tool Usage Guidelines" in system_content

    def test_base_params_configuration(self):
        """Test that base parameters are correctly configured"""
        generator = AIGenerator("test_key", "custom_model")
        
        assert generator.base_params["model"] == "custom_model"
        assert generator.base_params["temperature"] == 0
        assert generator.base_params["max_tokens"] == 800

    @patch('ai_generator.anthropic')
    def test_api_error_handling(self, mock_anthropic):
        """Test handling of API errors"""
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("API Error")
        mock_anthropic.Anthropic.return_value = mock_client
        
        generator = AIGenerator("test_key", "test_model")
        
        # Should propagate the exception
        with pytest.raises(Exception, match="API Error"):
            generator.generate_response("Test query")

    def test_tool_execution_without_tool_manager(self):
        """Test that tool execution is skipped when no tool manager provided"""
        generator = AIGenerator("test_key", "test_model")
        
        # Create mock response that would normally trigger tool execution
        mock_response = Mock()
        mock_response.stop_reason = "tool_use"
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_response.content = [mock_tool_content]
        
        with patch.object(generator.client.messages, 'create', return_value=mock_response):
            # This should handle the case where tool_manager is None
            result = generator.generate_response("Test query", tools=[{"name": "test"}])
            
            # Should return something (likely empty or error), not crash
            assert result is not None

    def test_empty_tool_results_handling(self):
        """Test handling when tool execution returns empty results"""
        generator = AIGenerator("test_key", "test_model")
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = ""
        
        mock_initial_response = Mock()
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.input = {"query": "empty"}
        mock_tool_content.id = "tool_1"
        mock_initial_response.content = [mock_tool_content]
        
        with patch.object(generator.client.messages, 'create') as mock_create:
            mock_final_response = Mock()
            mock_final_response.content = [Mock(text="No results found")]
            mock_create.return_value = mock_final_response
            
            base_params = {
                "messages": [{"role": "user", "content": "Test"}],
                "system": "System prompt"
            }
            
            result = generator._handle_tool_execution(
                mock_initial_response,
                base_params,
                mock_tool_manager
            )
            
            assert result == "No results found"
            
            # Verify empty tool result was still included
            final_call_args = mock_create.call_args
            tool_results = final_call_args[1]["messages"][2]["content"]
            assert len(tool_results) == 1
            assert tool_results[0]["content"] == ""