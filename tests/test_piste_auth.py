import pytest
from unittest.mock import patch, MagicMock
from piste_auth import AuthClient  # Assuming this is the class handling PISTE OAuth

class TestPisteAuth:

    @pytest.fixture
    def auth_client(self):
        # Setup a new AuthClient instance for each test
        return AuthClient(client_id='test_id', client_secret='test_secret')

    @patch('piste_auth.requests')  # Patch the requests module used in AuthClient
    def test_token_caching(self, mock_requests, auth_client):
        # Simulate getting a token for the first time
        mock_response = MagicMock()
        mock_response.json.return_value = {'access_token': 'test_token', 'expires_in': 3600}
        mock_requests.post.return_value = mock_response

        # First token request
        token = auth_client.get_token()
        assert token == 'test_token'

        # Second token request should return the cached token
        cached_token = auth_client.get_token()
        assert cached_token == 'test_token'
        mock_requests.post.assert_called_once()  # Should only call once

    @patch('piste_auth.requests')  # Patch the requests module again for the next test
    def test_token_auto_refresh(self, mock_requests, auth_client):
        # Simulate the token nearing expiration
        auth_client.token = 'expired_token'
        auth_client.token_expiry = 0  # Set expiry to the past, forcing a refresh

        # Simulated response for a refresh
        mock_response = MagicMock()
        mock_response.json.return_value = {'access_token': 'new_token', 'expires_in': 3600}
        mock_requests.post.return_value = mock_response

        new_token = auth_client.get_token()
        assert new_token == 'new_token'
        assert auth_client.token == 'new_token'  # Ensure the token is updated

    @patch('piste_auth.requests')  # Patch the requests module again
    def test_error_handling(self, mock_requests, auth_client):
        # Simulate a failed token request
        mock_response = MagicMock()
        mock_response.status_code = 400  # Bad request
        mock_requests.post.return_value = mock_response

        with pytest.raises(Exception) as exc_info:
            auth_client.get_token()
        assert 'Unable to retrieve token' in str(exc_info.value)  # Check for correct error message
