import os
import sys
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from sqlmodel import create_engine

from app.persistence.db import engine


@pytest.fixture(scope="session")
def monkeypatch_safe_requests():
    with patch("app.utils.safe_requests.get") as mock_get, patch(
        "app.utils.safe_requests.post"
    ) as mock_post:
        yield mock_get, mock_post


@pytest.fixture
def mock_requests():
    with patch("requests.get") as mock_get, patch("requests.post") as mock_post:
        yield mock_get, mock_post


@pytest.fixture
def mock_openai_llm():
    with patch("openai.OpenAI") as mock_openai:
        yield mock_openai


@pytest.fixture
def mock_genai_client():
    with patch("google.generativeai.GenerativeModel") as mock_genai:
        yield mock_genai


@pytest.fixture
def tmp_path_file(tmp_path):
    file_path = tmp_path / "test_file.txt"
    file_path.write_text("test content")
    return file_path


@pytest.fixture(scope="session")
def sample_payloads():
    return {
        "text": {"body": "Hello"},
        "image": {"mime_type": "image/jpeg"},
        "audio": {"mime_type": "audio/ogg"},
    }


@pytest.fixture
def deterministic_datetime(monkeypatch):
    class MockDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2023, 1, 1, 12, 0, 0)

    monkeypatch.setattr("datetime.datetime", MockDateTime)


@pytest.fixture(scope="session", autouse=True)
def set_pythonhashseed():
    os.environ["PYTHONHASHSEED"] = "0"
    random.seed(0)


@pytest.fixture(scope="session")
def in_memory_db_engine():
    return create_engine("sqlite:///:memory:")


@pytest.fixture(scope="function")
def populate_db(in_memory_db_engine, mock_data):
    with patch("app.persistence.db.engine", in_memory_db_engine):
        # Create tables and insert data
        from sqlmodel import SQLModel

        from app.models.mock_data import MOCK_DATA

        SQLModel.metadata.create_all(in_memory_db_engine)
        with in_memory_db_engine.connect() as connection:
            for table, data in MOCK_DATA.items():
                connection.execute(table.insert().values(data))
        yield
        SQLModel.metadata.drop_all(in_memory_db_engine)


@pytest.fixture
def mock_env_vars(monkeypatch):
    monkeypatch.setenv("WHATSAPP_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "test-key")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "test-key")
    monkeypatch.setenv("FIREBASE_SERVICE_ACCOUNT_KEY", "{}")
