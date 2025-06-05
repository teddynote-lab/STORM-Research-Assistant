"""STORM Research Assistant의 유틸리티 함수들

이 모듈은 프로젝트 전반에서 사용되는 공통 유틸리티 함수들을 제공합니다.
"""

from typing import Union, Optional
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


def load_chat_model(model_string: str) -> BaseChatModel:
    """모델 문자열을 파싱하여 적절한 Chat 모델을 로드
    
    Args:
        model_string: "provider/model-name" 형식의 문자열
        
    Returns:
        초기화된 Chat 모델
        
    Raises:
        ValueError: 지원하지 않는 프로바이더인 경우
    """
    # 프로바이더와 모델명 분리
    try:
        provider, model_name = model_string.split("/", 1)
    except ValueError:
        raise ValueError(
            f"모델 문자열은 'provider/model-name' 형식이어야 합니다. 입력값: {model_string}"
        )
    
    # 프로바이더별 모델 초기화
    if provider == "openai":
        return ChatOpenAI(model=model_name)
    elif provider == "anthropic":
        return ChatAnthropic(model=model_name)
    else:
        raise ValueError(f"지원하지 않는 프로바이더: {provider}")


def extract_text_from_message(
    message: Union[AIMessage, HumanMessage, SystemMessage, str]
) -> str:
    """다양한 메시지 타입에서 텍스트 추출
    
    Args:
        message: 텍스트를 추출할 메시지
        
    Returns:
        추출된 텍스트
    """
    if isinstance(message, str):
        return message
    elif isinstance(message, (AIMessage, HumanMessage, SystemMessage)):
        return message.content
    else:
        return str(message)


def format_analyst_description(analyst) -> str:
    """분석가 정보를 보기 좋게 포맷팅
    
    Args:
        analyst: Analyst 객체
        
    Returns:
        포맷팅된 분석가 설명
    """
    return (
        f"👤 **{analyst.name}**\n"
        f"   - 역할: {analyst.role}\n"
        f"   - 소속: {analyst.affiliation}\n"
        f"   - 전문분야: {analyst.description}"
    )


def format_section_header(section_name: str) -> str:
    """섹션 헤더를 일관된 형식으로 포맷팅
    
    Args:
        section_name: 섹션 이름
        
    Returns:
        포맷팅된 헤더
    """
    return f"\n\n## {section_name}\n\n"


def truncate_text(text: str, max_length: int = 1000) -> str:
    """텍스트를 지정된 길이로 자르고 말줄임표 추가
    
    Args:
        text: 자를 텍스트
        max_length: 최대 길이
        
    Returns:
        잘린 텍스트
    """
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."


def clean_source_citation(source: str) -> str:
    """소스 인용을 깔끔하게 정리
    
    Args:
        source: 원본 소스 문자열
        
    Returns:
        정리된 소스 문자열
    """
    # Document 태그 제거
    source = source.replace('<Document source="', '')
    source = source.replace('"/>', '')
    source = source.replace('</Document>', '')
    
    # 중복된 공백 제거
    source = ' '.join(source.split())
    
    return source.strip()


def generate_thread_id() -> str:
    """체크포인터용 고유 thread ID 생성
    
    Returns:
        UUID 기반 thread ID
    """
    import uuid
    return str(uuid.uuid4())