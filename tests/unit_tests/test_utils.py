"""유틸리티 함수 테스트"""

import pytest
from unittest.mock import patch, MagicMock
from storm_research.utils import load_chat_model
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_anthropic import ChatAnthropic


class TestLoadChatModel:
    """load_chat_model 함수 테스트"""
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-api-key'})
    def test_load_openai_model(self):
        """OpenAI 모델 로드 테스트"""
        model = load_chat_model("openai/gpt-4")
        assert isinstance(model, ChatOpenAI)
        assert model.model_name == "gpt-4"
    
    def test_load_anthropic_model(self):
        """Anthropic 모델 로드 테스트"""
        model = load_chat_model("anthropic/claude-3-5-sonnet-20240620")
        assert isinstance(model, ChatAnthropic)
        assert model.model == "claude-3-5-sonnet-20240620"
    
    @patch.dict('os.environ', {
        'AZURE_OPENAI_ENDPOINT': 'https://test.openai.azure.com/',
        'AZURE_OPENAI_API_KEY': 'test-api-key'
    })
    def test_load_azure_model(self):
        """Azure OpenAI 모델 로드 테스트"""
        model = load_chat_model("azure/gpt-4.1")
        assert isinstance(model, AzureChatOpenAI)
        assert model.deployment_name == "gpt-4.1"
        assert model.openai_api_version == "2024-12-01-preview"
        assert model.temperature == 0.1
    
    @patch.dict('os.environ', {}, clear=True)
    def test_azure_missing_endpoint(self):
        """Azure OpenAI 엔드포인트 누락 시 에러 테스트"""
        with pytest.raises(ValueError) as exc_info:
            load_chat_model("azure/gpt-4.1")
        
        assert "AZURE_OPENAI_ENDPOINT" in str(exc_info.value)
        assert "AZURE_OPENAI_API_KEY" in str(exc_info.value)
    
    def test_invalid_model_string(self):
        """잘못된 모델 문자열 형식 테스트"""
        with pytest.raises(ValueError) as exc_info:
            load_chat_model("invalid-format")
        
        assert "provider/model-name" in str(exc_info.value)
    
    def test_unsupported_provider(self):
        """지원하지 않는 프로바이더 테스트"""
        with pytest.raises(ValueError) as exc_info:
            load_chat_model("unsupported/model")
        
        assert "지원하지 않는 프로바이더: unsupported" in str(exc_info.value)