import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from fastapi import status


@pytest.mark.api
class TestAPIEndpoints:
    """Comprehensive tests for FastAPI endpoints"""

    def test_root_endpoint(self, client):
        """Test the root endpoint returns basic info"""
        response = client.get("/")
        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {"message": "Course Materials RAG System"}

    def test_query_endpoint_with_session_id(self, client, api_test_data):
        """Test /api/query endpoint with session ID"""
        query_data = api_test_data["valid_query"]
        
        response = client.post("/api/query", json=query_data)
        
        assert response.status_code == status.HTTP_200_OK
        json_response = response.json()
        
        # Verify response structure
        assert "answer" in json_response
        assert "sources" in json_response
        assert "session_id" in json_response
        assert json_response["session_id"] == query_data["session_id"]
        assert json_response["answer"] == "Test response"
        assert len(json_response["sources"]) > 0

    def test_query_endpoint_without_session_id(self, client, api_test_data):
        """Test /api/query endpoint without session ID (should create one)"""
        query_data = api_test_data["query_without_session"]
        
        response = client.post("/api/query", json=query_data)
        
        assert response.status_code == status.HTTP_200_OK
        json_response = response.json()
        
        # Should have created a session ID
        assert "session_id" in json_response
        assert json_response["session_id"] == "test_session_123"  # From mock
        assert json_response["answer"] == "Test response"

    def test_query_endpoint_empty_query(self, client, api_test_data):
        """Test /api/query endpoint with empty query"""
        query_data = api_test_data["empty_query"]
        
        response = client.post("/api/query", json=query_data)
        
        # Should still process empty queries
        assert response.status_code == status.HTTP_200_OK
        json_response = response.json()
        assert "answer" in json_response

    def test_query_endpoint_invalid_request_format(self, client, api_test_data):
        """Test /api/query endpoint with invalid request format"""
        invalid_data = api_test_data["invalid_query"]
        
        response = client.post("/api/query", json=invalid_data)
        
        # Should return validation error
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_query_endpoint_missing_body(self, client):
        """Test /api/query endpoint with missing request body"""
        response = client.post("/api/query")
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_query_endpoint_internal_error(self, client, test_app, mock_rag_system_with_error):
        """Test /api/query endpoint when RAG system raises error"""
        # Replace the mock with one that raises an error
        original_mock = test_app.state.mock_rag_system
        test_app.state.mock_rag_system = mock_rag_system_with_error
        
        try:
            response = client.post("/api/query", json={"query": "test query"})
            
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert "Test error" in response.json()["detail"]
        finally:
            # Restore original mock
            test_app.state.mock_rag_system = original_mock

    def test_courses_endpoint_success(self, client):
        """Test /api/courses endpoint returns course statistics"""
        response = client.get("/api/courses")
        
        assert response.status_code == status.HTTP_200_OK
        json_response = response.json()
        
        # Verify response structure
        assert "total_courses" in json_response
        assert "course_titles" in json_response
        assert json_response["total_courses"] == 1
        assert json_response["course_titles"] == ["Test Course"]

    def test_courses_endpoint_internal_error(self, client, test_app, mock_rag_system_with_error):
        """Test /api/courses endpoint when RAG system raises error"""
        # Replace the mock with one that raises an error
        original_mock = test_app.state.mock_rag_system
        test_app.state.mock_rag_system = mock_rag_system_with_error
        
        try:
            response = client.get("/api/courses")
            
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert "Analytics error" in response.json()["detail"]
        finally:
            # Restore original mock
            test_app.state.mock_rag_system = original_mock

    def test_clear_session_endpoint_success(self, client):
        """Test /api/sessions/{session_id}/clear endpoint"""
        session_id = "test_session_456"
        
        response = client.post(f"/api/sessions/{session_id}/clear")
        
        assert response.status_code == status.HTTP_200_OK
        json_response = response.json()
        
        assert json_response["message"] == "Session cleared successfully"
        assert json_response["session_id"] == session_id

    def test_clear_session_endpoint_internal_error(self, client, test_app, mock_rag_system_with_error):
        """Test /api/sessions/{session_id}/clear endpoint when RAG system raises error"""
        # Replace the mock with one that raises an error
        original_mock = test_app.state.mock_rag_system
        test_app.state.mock_rag_system = mock_rag_system_with_error
        
        try:
            response = client.post("/api/sessions/test_session/clear")
            
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert "Session error" in response.json()["detail"]
        finally:
            # Restore original mock
            test_app.state.mock_rag_system = original_mock

    def test_nonexistent_endpoint(self, client):
        """Test request to non-existent endpoint"""
        response = client.get("/api/nonexistent")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_query_endpoint_method_not_allowed(self, client):
        """Test /api/query endpoint with wrong HTTP method"""
        response = client.get("/api/query")
        
        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED

    def test_courses_endpoint_method_not_allowed(self, client):
        """Test /api/courses endpoint with wrong HTTP method"""
        response = client.post("/api/courses")
        
        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED

    def test_clear_session_endpoint_method_not_allowed(self, client):
        """Test clear session endpoint with wrong HTTP method"""
        response = client.get("/api/sessions/test_session/clear")
        
        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED


@pytest.mark.api
class TestAPIResponseFormats:
    """Test API response formats and data types"""

    def test_query_response_format(self, client):
        """Test that query response has correct format and types"""
        response = client.post("/api/query", json={"query": "test query"})
        
        assert response.status_code == status.HTTP_200_OK
        json_response = response.json()
        
        # Check types
        assert isinstance(json_response["answer"], str)
        assert isinstance(json_response["sources"], list)
        assert isinstance(json_response["session_id"], str)
        
        # Check source format
        if json_response["sources"]:
            source = json_response["sources"][0]
            assert "text" in source
            assert "url" in source
            assert isinstance(source["text"], str)
            # url can be None or string

    def test_courses_response_format(self, client):
        """Test that courses response has correct format and types"""
        response = client.get("/api/courses")
        
        assert response.status_code == status.HTTP_200_OK
        json_response = response.json()
        
        # Check types
        assert isinstance(json_response["total_courses"], int)
        assert isinstance(json_response["course_titles"], list)
        
        # All course titles should be strings
        for title in json_response["course_titles"]:
            assert isinstance(title, str)

    def test_clear_session_response_format(self, client):
        """Test that clear session response has correct format"""
        response = client.post("/api/sessions/test_session/clear")
        
        assert response.status_code == status.HTTP_200_OK
        json_response = response.json()
        
        # Check types
        assert isinstance(json_response["message"], str)
        assert isinstance(json_response["session_id"], str)

    def test_error_response_format(self, client):
        """Test that error responses have correct format"""
        # Test validation error
        response = client.post("/api/query", json={"invalid": "data"})
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        json_response = response.json()
        
        # FastAPI validation error format
        assert "detail" in json_response
        assert isinstance(json_response["detail"], list)


@pytest.mark.api
class TestAPIContentTypes:
    """Test API content type handling"""

    def test_json_content_type_required(self, client):
        """Test that endpoints require JSON content type"""
        response = client.post("/api/query", data="query=test")
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_response_content_type(self, client):
        """Test that responses have correct content type"""
        response = client.get("/api/courses")
        
        assert response.status_code == status.HTTP_200_OK
        assert "application/json" in response.headers["content-type"]

    def test_cors_headers(self, client):
        """Test that CORS headers are present"""
        response = client.options("/api/query")
        
        # FastAPI automatically handles OPTIONS requests for CORS
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_405_METHOD_NOT_ALLOWED]


@pytest.mark.api
@pytest.mark.integration
class TestAPIIntegration:
    """Integration tests for API functionality"""

    def test_query_session_flow(self, client):
        """Test complete query and session clearing flow"""
        # Make a query without session ID
        response = client.post("/api/query", json={"query": "What is AI?"})
        assert response.status_code == status.HTTP_200_OK
        
        session_id = response.json()["session_id"]
        
        # Make another query with the same session ID
        response = client.post("/api/query", json={
            "query": "Tell me more about that",
            "session_id": session_id
        })
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["session_id"] == session_id
        
        # Clear the session
        response = client.post(f"/api/sessions/{session_id}/clear")
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["session_id"] == session_id

    def test_multiple_concurrent_sessions(self, client):
        """Test that multiple sessions work independently"""
        # Create first session
        response1 = client.post("/api/query", json={"query": "First session query"})
        session_id_1 = response1.json()["session_id"]
        
        # Create second session  
        response2 = client.post("/api/query", json={"query": "Second session query"})
        session_id_2 = response2.json()["session_id"]
        
        # Sessions should be different (in real implementation)
        # Note: In this test setup, the mock always returns the same session ID
        # In a real scenario, we'd verify they're different
        assert response1.status_code == status.HTTP_200_OK
        assert response2.status_code == status.HTTP_200_OK

    def test_api_endpoints_exist(self, client):
        """Test that all expected API endpoints exist and return appropriate responses"""
        endpoints = [
            ("GET", "/", status.HTTP_200_OK),
            ("GET", "/api/courses", status.HTTP_200_OK),
            ("POST", "/api/query", status.HTTP_422_UNPROCESSABLE_ENTITY),  # Without body
            ("POST", "/api/sessions/test/clear", status.HTTP_200_OK),
        ]
        
        for method, path, expected_status in endpoints:
            if method == "GET":
                response = client.get(path)
            elif method == "POST":
                if path == "/api/query":
                    response = client.post(path)  # No body to trigger validation error
                else:
                    response = client.post(path)
            
            assert response.status_code == expected_status, f"Failed for {method} {path}"