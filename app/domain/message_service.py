import os
import json
import requests
from typing import BinaryIO

from app.domain.agents.demo_agent import demo_agent
from app.schema import Audio, User
from openai import OpenAI
from security import safe_requests



WHATSAPP_API_KEY = "YOUR_API_KEY"
llm = OpenAI()

def transcribe_audio_file(audio_file: BinaryIO) -> str:
    if not audio_file:
        return "No audio file provided"
    try:
        transcription = llm.audio.transcriptions.create(
            file=audio_file,
            model="whisper-1",
            response_format="text"
        )
        return transcription
    except Exception as e:
        raise ValueError("Error transcribing audio") from e


def transcribe_audio(audio: Audio) -> str:
    file_path = download_file_from_facebook(audio.id, "audio", audio.mime_type)
    with open(file_path, 'rb') as audio_binary:
        transcription = transcribe_audio_file(audio_binary)
    try:
        os.remove(file_path)
    except Exception as e:
        print(f"Failed to delete file: {e}")
    return transcription


def download_file_from_facebook(file_id: str, file_type: str, mime_type: str) -> str | None:
    # First GET request to retrieve the download URL
    url = f"https://graph.facebook.com/v19.0/{file_id}"
    headers = {"Authorization": f"Bearer {WHATSAPP_API_KEY}"}
    response = safe_requests.get(url, headers=headers)

    if response.status_code == 200:
        download_url = response.json().get('url')

        # Second GET request to download the file
        response = safe_requests.get(download_url, headers=headers)

        if response.status_code == 200:
            file_extension = mime_type.split('/')[-1].split(';')[0]  # Extract file extension from mime_type
            file_path = f"{file_id}.{file_extension}"
            with open(file_path, 'wb') as file:
                file.write(response.content)
            if file_type == "image" or file_type == "audio":
                return file_path

        raise ValueError(f"Failed to download file. Status code: {response.status_code}")
    raise ValueError(f"Failed to retrieve download URL. Status code: {response.status_code}")


def authenticate_user_by_phone_number(phone_number: str) -> User | None:
    allowed_users = [
        {"id": 1, "phone": "+1234567890", "first_name": "John", "last_name": "Doe", "role": "basic"},
        {"id": 2, "phone": "+0987654321", "first_name": "Jane", "last_name": "Smith", "role": "basic"}
    ]

    for user in allowed_users:
        if user["phone"] == phone_number:
            return User(**user)
    return None


def send_whatsapp_message(to, message, template=True):
    url = f"https://graph.facebook.com/v18.0/289534840903017/messages"
    headers = {
        "Authorization": f"Bearer " + WHATSAPP_API_KEY,
        "Content-Type": "application/json"
    }
    if not template:
        data = {
            "messaging_product": "whatsapp",
            "preview_url": False,
            "recipient_type": "individual",
            "to": to,
            "type": "text",
            "text": {
                "body": message
            }
        }
    else:
        data = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "template",
            "template": {
                "name": "hello_world",
                "language": {
                    "code": "en_US"
                }
            }
        }

    response = requests.post(url, headers=headers, data=json.dumps(data), timeout=60)
    return response.json()


def respond_and_send_message(user_message: str, user: User):
    agent = demo_agent
    response = agent.run(user_message, user.id)
    send_whatsapp_message(user.phone, response, template=False)
    print(f"Sent message to user {user.first_name} {user.last_name} ({user.phone})")
    print(f"Message: {response}")


if __name__ == "__main__":
    demo_agent.run("What are my expenses", 1)

