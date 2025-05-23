import pytest
from unittest.mock import patch, MagicMock, mock_open
import os
import json # Added json import
from typing import BinaryIO

# Assuming the app structure, adjust paths if necessary
from app.domain.message_service import (
    transcribe_audio_file,
    transcribe_audio,
    download_file_from_facebook,
    send_whatsapp_message,
    respond_and_send_message,
    authenticate_user_by_phone_number, # Added this import
)
from app.schema import Audio, User # Added User import


# Tests for transcribe_audio_file
@patch('app.domain.message_service.llm.audio.transcriptions.create')
def test_transcribe_audio_file_success(mock_create_transcription):
    """Tests successful transcription of an audio file."""
    mock_transcription_object = MagicMock()
    mock_transcription_object.text = "This is a test transcription."
    mock_create_transcription.return_value = mock_transcription_object
    
    mock_audio_file = MagicMock(spec=BinaryIO)
    
    result = transcribe_audio_file(mock_audio_file)
    
    mock_create_transcription.assert_called_once_with(
        file=mock_audio_file,
        model="whisper-1",
        response_format="text"
    )
    assert result == "This is a test transcription."

def test_transcribe_audio_file_no_file_provided():
    """Tests behavior when no audio file is provided."""
    result = transcribe_audio_file(None)
    assert result == "No audio file provided"

@patch('app.domain.message_service.llm.audio.transcriptions.create')
def test_transcribe_audio_file_api_error(mock_create_transcription):
    """Tests behavior when OpenAI API call fails."""
    mock_create_transcription.side_effect = Exception("API Error")
    
    mock_audio_file = MagicMock(spec=BinaryIO)
    
    with pytest.raises(ValueError, match="Error transcribing audio: API Error"):
        transcribe_audio_file(mock_audio_file)


# Tests for transcribe_audio
@patch('app.domain.message_service.os.remove')
@patch('app.domain.message_service.transcribe_audio_file')
@patch('app.domain.message_service.download_file_from_facebook')
@patch('builtins.open', new_callable=mock_open)
def test_transcribe_audio_success(
    mock_builtin_open,
    mock_download_file,
    mock_transcribe_file,
    mock_os_remove
):
    """Tests successful audio transcription flow."""
    dummy_file_path = "dummy_audio.wav"
    expected_transcription = "Test transcription"
    
    mock_download_file.return_value = dummy_file_path
    mock_transcribe_file.return_value = expected_transcription
    
    audio_object = Audio(
        id="audio_id_123",
        mime_type="audio/wav",
        sha256="dummy",
        voice=False,
    )
    
    result = transcribe_audio(audio_object)
    
    mock_download_file.assert_called_once_with(
        audio_object.id, "audio", audio_object.mime_type
    )
    mock_builtin_open.assert_called_once_with(dummy_file_path, 'rb')
    # The file object passed to transcribe_audio_file is the one returned by mock_open
    mock_transcribe_file.assert_called_once_with(mock_builtin_open.return_value)
    mock_os_remove.assert_called_once_with(dummy_file_path)
    assert result == expected_transcription

@patch('app.domain.message_service.os.remove')
@patch('app.domain.message_service.transcribe_audio_file')
@patch('app.domain.message_service.download_file_from_facebook')
def test_transcribe_audio_download_fails(
    mock_download_file,
    mock_transcribe_file,
    mock_os_remove
):
    """Tests behavior when download_file_from_facebook fails."""
    mock_download_file.side_effect = ValueError("Download failed")
    
    audio_object = Audio(id="audio_id_456", mime_type="audio/ogg")
    
    with pytest.raises(ValueError, match="Download failed"):
        transcribe_audio(audio_object)
        
    mock_transcribe_file.assert_not_called()
    mock_os_remove.assert_not_called()

# TODO: Add tests for download_file_from_facebook
@patch('app.domain.message_service.os.environ.get') # To control WHATSAPP_API_KEY
@patch('app.domain.message_service.requests.get')
@patch('builtins.open', new_callable=mock_open)
def test_download_file_from_facebook_success(
    mock_builtin_open,
    mock_requests_get,
    mock_os_environ_get,
):
    """Tests successful file download from Facebook."""
    mock_os_environ_get.return_value = "dummy_whatsapp_api_key" # Mock WHATSAPP_API_KEY
    
    # Configure requests.get for two calls
    mock_response_get_url = MagicMock()
    mock_response_get_url.status_code = 200
    mock_response_get_url.json.return_value = {'url': 'http://dummy_download_url/file.mp3'}
    
    mock_response_download_file = MagicMock()
    mock_response_download_file.status_code = 200
    mock_response_download_file.content = b'dummy_audio_content'
    
    mock_requests_get.side_effect = [mock_response_get_url, mock_response_download_file]
    
    file_id = "fb_file_id_123"
    file_type = "audio"
    mime_type = "audio/mpeg"
    expected_filename = f"{file_id}.mpeg" # from mime_type audio/mpeg
    
    result_path = download_file_from_facebook(file_id, file_type, mime_type)
    
    # Assert requests.get calls
    assert mock_requests_get.call_count == 2
    first_call_args, first_call_kwargs = mock_requests_get.call_args_list[0]
    second_call_args, second_call_kwargs = mock_requests_get.call_args_list[1]
    
    assert first_call_args[0] == f"https://graph.facebook.com/v19.0/{file_id}"
    assert "headers" in first_call_kwargs
    assert first_call_kwargs["headers"]["Authorization"] == "Bearer dummy_whatsapp_api_key"
    
    assert second_call_args[0] == 'http://dummy_download_url/file.mp3'
    assert "headers" in second_call_kwargs
    assert second_call_kwargs["headers"]["Authorization"] == "Bearer dummy_whatsapp_api_key"
    
    # Assert builtins.open call
    mock_builtin_open.assert_called_once_with(expected_filename, 'wb')
    
    # Assert file write
    mock_file_handle = mock_builtin_open.return_value
    mock_file_handle.write.assert_called_once_with(b'dummy_audio_content')
    
    assert result_path == expected_filename

@patch('app.domain.message_service.os.environ.get')
@patch('app.domain.message_service.requests.get')
def test_download_file_from_facebook_get_url_fails(
    mock_requests_get,
    mock_os_environ_get
):
    """Tests failure when retrieving download URL from Facebook."""
    mock_os_environ_get.return_value = "dummy_whatsapp_api_key"
    
    mock_response_get_url_fail = MagicMock()
    mock_response_get_url_fail.status_code = 400
    mock_requests_get.return_value = mock_response_get_url_fail # Only one call will be made
    
    with pytest.raises(ValueError, match="Failed to retrieve download URL. Status: 400, Response: "):
        download_file_from_facebook("fb_id_fail_url", "audio", "audio/aac")
        
    mock_requests_get.assert_called_once() # Ensure it was called for the URL retrieval

@patch('app.domain.message_service.os.environ.get')
@patch('app.domain.message_service.requests.get')
def test_download_file_from_facebook_download_file_fails(
    mock_requests_get,
    mock_os_environ_get
):
    """Tests failure when downloading the actual file after getting URL."""
    mock_os_environ_get.return_value = "dummy_whatsapp_api_key"

    mock_response_get_url_success = MagicMock()
    mock_response_get_url_success.status_code = 200
    mock_response_get_url_success.json.return_value = {'url': 'http://dummy_download_url/file.ogg'}
    
    mock_response_download_fail = MagicMock()
    mock_response_download_fail.status_code = 403 # Forbidden, for example
    
    mock_requests_get.side_effect = [mock_response_get_url_success, mock_response_download_fail]
    
    with pytest.raises(ValueError, match="Failed to download file. Status: 403, Response: "):
        download_file_from_facebook("fb_id_fail_download", "audio", "audio/ogg")
        
    assert mock_requests_get.call_count == 2

@patch('app.domain.message_service.os.environ.get')
@patch('app.domain.message_service.requests.get')
@patch('builtins.open', new_callable=mock_open)
def test_download_file_from_facebook_image_success(
    mock_builtin_open,
    mock_requests_get,
    mock_os_environ_get
):
    """Tests successful image file download from Facebook with correct extension."""
    mock_os_environ_get.return_value = "dummy_whatsapp_api_key"
    
    mock_response_get_url = MagicMock()
    mock_response_get_url.status_code = 200
    mock_response_get_url.json.return_value = {'url': 'http://dummy_download_url/image.jpg'}
    
    mock_response_download_file = MagicMock()
    mock_response_download_file.status_code = 200
    mock_response_download_file.content = b'dummy_image_content'
    
    mock_requests_get.side_effect = [mock_response_get_url, mock_response_download_file]
    
    file_id = "fb_image_id_789"
    file_type = "image" # This param is not used in the function currently
    mime_type = "image/jpeg" 
    expected_filename = f"{file_id}.jpeg" # from mime_type image/jpeg
    
    result_path = download_file_from_facebook(file_id, file_type, mime_type)
    
    assert mock_requests_get.call_count == 2
    # Basic check for URL and headers
    first_call_args, first_call_kwargs = mock_requests_get.call_args_list[0]
    assert first_call_args[0] == f"https://graph.facebook.com/v19.0/{file_id}"
    assert "headers" in first_call_kwargs
    assert first_call_kwargs["headers"]["Authorization"] == "Bearer dummy_whatsapp_api_key"
    
    second_call_args, second_call_kwargs = mock_requests_get.call_args_list[1]
    assert second_call_args[0] == 'http://dummy_download_url/image.jpg'
    assert "headers" in second_call_kwargs
    assert second_call_kwargs["headers"]["Authorization"] == "Bearer dummy_whatsapp_api_key"

    mock_builtin_open.assert_called_once_with(expected_filename, 'wb')
    mock_file_handle = mock_builtin_open.return_value
    mock_file_handle.write.assert_called_once_with(b'dummy_image_content')
    assert result_path == expected_filename

# TODO: Add tests for send_whatsapp_message
# TODO: Add tests for respond_and_send_message

# Tests for authenticate_user_by_phone_number
def test_authenticate_user_by_phone_number_allowed():
    """Tests authentication with an allowed phone number."""
    # Assuming "+1234567890" is in the allowed_users list with specific details
    # Based on the provided allowed_users list in message_service.py:
    # {
    #     "phone": "+1234567890",
    #     "first_name": "Test",
    #     "last_name": "User",
    #     "role": "admin"
    # }
    phone_number = "+1234567890"
    user = authenticate_user_by_phone_number(phone_number)
    
    assert user is not None
    assert isinstance(user, User)
    assert user.id == phone_number # id is the phone number
    assert user.phone == phone_number
    assert user.first_name == "Test"
    assert user.last_name == "User"
    assert user.role == "admin"

def test_authenticate_user_by_phone_number_not_allowed():
    """Tests authentication with a non-allowed phone number."""
    phone_number = "+1112223333" # This number should not be in allowed_users
    user = authenticate_user_by_phone_number(phone_number)
    assert user is None

def test_authenticate_user_by_phone_number_empty():
    """Tests authentication with an empty phone number string."""
    user = authenticate_user_by_phone_number("")
    assert user is None

# Tests for send_whatsapp_message
@patch('app.domain.message_service.os.environ.get')
@patch('app.domain.message_service.requests.post')
def test_send_whatsapp_message_text_success(
    mock_requests_post,
    mock_os_environ_get
):
    """Tests successful sending of a text WhatsApp message."""
    mock_os_environ_get.return_value = "DUMMY_KEY" # Mock WHATSAPP_API_KEY
    
    expected_response_json = {'messages': [{'id': 'message_id_123'}]}
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = expected_response_json
    mock_requests_post.return_value = mock_response
    
    recipient_phone = "+15551234567"
    message_body = "Hello there!"
    
    response = send_whatsapp_message(to=recipient_phone, message=message_body, template=False)
    
    mock_requests_post.assert_called_once()
    call_args, call_kwargs = mock_requests_post.call_args
    
    assert call_args[0] == "https://graph.facebook.com/v18.0/289534840903017/messages"
    assert "headers" in call_kwargs
    assert call_kwargs["headers"]["Authorization"] == "Bearer DUMMY_KEY"
    assert call_kwargs["headers"]["Content-Type"] == "application/json"
    
    assert "data" in call_kwargs
    sent_data = json.loads(call_kwargs["data"]) # Data is sent as a JSON string
    
    expected_data_payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": recipient_phone,
        "preview_url": False,
        "type": "text",
        "text": {"body": message_body}
    }
    assert sent_data == expected_data_payload
    assert response == expected_response_json

@patch('app.domain.message_service.os.environ.get')
@patch('app.domain.message_service.requests.post')
def test_send_whatsapp_message_template_success(
    mock_requests_post,
    mock_os_environ_get
):
    """Tests successful sending of a template WhatsApp message."""
    mock_os_environ_get.return_value = "DUMMY_KEY"
    
    expected_response_json = {'messages': [{'id': 'template_msg_id_456'}]}
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = expected_response_json
    mock_requests_post.return_value = mock_response
    
    recipient_phone = "+15557654321"
    
    response = send_whatsapp_message(to=recipient_phone, message="This is ignored for template", template=True)
    
    mock_requests_post.assert_called_once()
    call_args, call_kwargs = mock_requests_post.call_args
    
    assert call_args[0] == "https://graph.facebook.com/v18.0/289534840903017/messages"
    assert "headers" in call_kwargs
    assert call_kwargs["headers"]["Authorization"] == "Bearer DUMMY_KEY"
    assert call_kwargs["headers"]["Content-Type"] == "application/json"
    
    assert "data" in call_kwargs
    sent_data = json.loads(call_kwargs["data"])
    
    expected_data_payload = {
        "messaging_product": "whatsapp",
        "to": recipient_phone,
        "type": "template",
        "template": {
            "name": "hello_world", # As per function's hardcoded template
            "language": {"code": "en_US"}
        }
    }
    assert sent_data == expected_data_payload
    assert response == expected_response_json

@patch('app.domain.message_service.os.environ.get')
@patch('app.domain.message_service.requests.post')
def test_send_whatsapp_message_api_error(
    mock_requests_post,
    mock_os_environ_get
):
    """Tests behavior when WhatsApp API call fails."""
    mock_os_environ_get.return_value = "DUMMY_KEY"
    
    error_response_json = {"error": {"message": "API limit reached", "type": "OAuthException"}}
    mock_response = MagicMock()
    mock_response.status_code = 400 # Example error code
    mock_response.json.return_value = error_response_json
    mock_requests_post.return_value = mock_response
    
    recipient_phone = "+15559876543"
    message_body = "Test message for error"
    
    response = send_whatsapp_message(to=recipient_phone, message=message_body, template=False)
    
    mock_requests_post.assert_called_once() # Ensure it was called
    # We can also check call_args if needed, but the main point is the return value
    
    assert response == error_response_json

# Tests for respond_and_send_message
@patch('app.domain.message_service.send_whatsapp_message')
@patch('app.domain.message_service.demo_agent.run')
def test_respond_and_send_message_success(
    mock_agent_run,
    mock_send_whatsapp_message,
    caplog # For optional print statement checking
):
    """Tests the successful flow of respond_and_send_message."""
    agent_response_text = "Agent response message"
    mock_agent_run.return_value = agent_response_text
    
    whatsapp_success_response = {'status': 'sent', 'message_id': 'wa_msg_123'}
    mock_send_whatsapp_message.return_value = whatsapp_success_response
    
    mock_user = User(id="user_id_001", phone="+1234567890", first_name="Test", last_name="User", role="basic")
    user_message_text = "Hello agent, please help."
    
    # The function prints the return value of send_whatsapp_message
    # and does not return anything itself.
    respond_and_send_message(user_message=user_message_text, user=mock_user)
    
    mock_agent_run.assert_called_once_with(user_message_text, mock_user.id)
    mock_send_whatsapp_message.assert_called_once_with(
        to=mock_user.phone,
        message=agent_response_text,
        template=False
    )
    
    # Optional: Check caplog for print statements
    # The function definition is: print(f"Message sent: {response}")
    assert f"Message sent: {whatsapp_success_response}" in caplog.text
    assert f"User message: {user_message_text}" in caplog.text
    assert f"Agent response: {agent_response_text}" in caplog.text

@patch('app.domain.message_service.send_whatsapp_message')
@patch('app.domain.message_service.demo_agent.run')
def test_respond_and_send_message_agent_fails(
    mock_agent_run,
    mock_send_whatsapp_message
):
    """Tests behavior when the agent's run method fails."""
    agent_failure_exception = Exception("Agent failed spectacularly")
    mock_agent_run.side_effect = agent_failure_exception
    
    mock_user = User(id="user_id_002", phone="+1987654321", first_name="Another", last_name="TestUser", role="admin")
    user_message_text = "This will fail."
    
    with pytest.raises(Exception, match="Agent failed spectacularly") as excinfo:
        respond_and_send_message(user_message=user_message_text, user=mock_user)
        
    assert excinfo.value == agent_failure_exception # Check it's the same exception
    mock_agent_run.assert_called_once_with(user_message_text, mock_user.id)
    mock_send_whatsapp_message.assert_not_called()

@patch('app.domain.message_service.send_whatsapp_message')
@patch('app.domain.message_service.demo_agent.run')
def test_respond_and_send_message_whatsapp_fails(
    mock_agent_run,
    mock_send_whatsapp_message
):
    """Tests behavior when send_whatsapp_message fails."""
    agent_response_text = "Agent worked fine."
    mock_agent_run.return_value = agent_response_text
    
    send_failure_exception = Exception("Send failed due to network issue")
    mock_send_whatsapp_message.side_effect = send_failure_exception
    
    mock_user = User(id="user_id_003", phone="+1122334455", first_name="YetAnother", last_name="Person", role="basic")
    user_message_text = "Hoping this send fails."
    
    with pytest.raises(Exception, match="Send failed due to network issue") as excinfo:
        respond_and_send_message(user_message=user_message_text, user=mock_user)
        
    assert excinfo.value == send_failure_exception # Check it's the same exception
    mock_agent_run.assert_called_once_with(user_message_text, mock_user.id)
    mock_send_whatsapp_message.assert_called_once_with(
        to=mock_user.phone,
        message=agent_response_text,
        template=False
    )
