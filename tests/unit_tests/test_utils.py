"""Utility function tests"""

import pytest
from unittest.mock import patch, MagicMock
from storm_research.utils import load_chat_model
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_anthropic import ChatAnthropic


class TestLoadChatModel:
    """Tests for load_chat_model function"""

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-api-key"})
    def test_load_openai_model(self):
        """Test loading OpenAI model"""
        model = load_chat_model("openai/gpt-4.1")
        assert isinstance(model, ChatOpenAI)
        assert model.model_name == "gpt-4.1"

    def test_load_anthropic_model(self):
        """Test loading Anthropic model"""
        model = load_chat_model("anthropic/claude-3-5-haiku-latest")
        assert isinstance(model, ChatAnthropic)
        assert model.model == "claude-3-5-haiku-latest"

    @patch.dict(
        "os.environ",
        {
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_API_KEY": "test-api-key",
        },
    )
    def test_load_azure_model(self):
        """Test loading Azure OpenAI model"""
        model = load_chat_model("azure/gpt-4.1")
        assert isinstance(model, AzureChatOpenAI)
        assert model.deployment_name == "gpt-4.1"
        assert model.openai_api_version == "2024-12-01-preview"
        assert model.temperature == 0.1

    @patch.dict("os.environ", {}, clear=True)
    def test_azure_missing_endpoint(self):
        """Test error when Azure OpenAI endpoint is missing"""
        with pytest.raises(ValueError) as exc_info:
            load_chat_model("azure/gpt-4.1")

        assert "AZURE_OPENAI_ENDPOINT" in str(exc_info.value)
        assert "AZURE_OPENAI_API_KEY" in str(exc_info.value)

    def test_invalid_model_string(self):
        """Test invalid model string format"""
        with pytest.raises(ValueError) as exc_info:
            load_chat_model("invalid-format")

        assert "provider/model-name" in str(exc_info.value)

    def test_unsupported_provider(self):
        """Test unsupported provider"""
        with pytest.raises(ValueError) as exc_info:
            load_chat_model("unsupported/model")

        assert "Unsupported provider: unsupported" in str(exc_info.value)
