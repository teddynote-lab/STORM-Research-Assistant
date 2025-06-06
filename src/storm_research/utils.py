"""STORM Research Assistant Utility Functions

This module provides common utility functions used throughout the project.
"""

import os
from typing import Union, Optional
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


def load_chat_model(model_string: str) -> BaseChatModel:
    """Parse model string and load appropriate Chat model
    
    Args:
        model_string: String in "provider/model-name" format
        
    Returns:
        Initialized Chat model
        
    Raises:
        ValueError: If provider is not supported
    """
    # Separate provider and model name
    try:
        provider, model_name = model_string.split("/", 1)
    except ValueError:
        raise ValueError(
            f"Model string must be in 'provider/model-name' format. Input: {model_string}"
        )
    
    # Initialize model by provider
    if provider == "openai":
        return ChatOpenAI(model=model_name)
    elif provider == "anthropic":
        return ChatAnthropic(model=model_name)
    elif provider == "azure":
        # Azure OpenAI configuration
        azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        azure_api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        
        if not azure_endpoint or not azure_api_key:
            raise ValueError(
                "To use Azure OpenAI, AZURE_OPENAI_ENDPOINT and "
                "AZURE_OPENAI_API_KEY environment variables must be set."
            )
        
        return AzureChatOpenAI(
            deployment_name=model_name,
            api_version="2024-12-01-preview",
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
            temperature=0.1
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def extract_text_from_message(
    message: Union[AIMessage, HumanMessage, SystemMessage, str]
) -> str:
    """Extract text from various message types
    
    Args:
        message: Message to extract text from
        
    Returns:
        Extracted text
    """
    if isinstance(message, str):
        return message
    elif isinstance(message, (AIMessage, HumanMessage, SystemMessage)):
        return message.content
    else:
        return str(message)


def format_analyst_description(analyst) -> str:
    """Format analyst information for display
    
    Args:
        analyst: Analyst object
        
    Returns:
        Formatted analyst description
    """
    return (
        f"👤 **{analyst.name}**\n"
        f"   - Role: {analyst.role}\n"
        f"   - Affiliation: {analyst.affiliation}\n"
        f"   - Expertise: {analyst.description}"
    )


def format_section_header(section_name: str) -> str:
    """Format section header in consistent style
    
    Args:
        section_name: Section name
        
    Returns:
        Formatted header
    """
    return f"\n\n## {section_name}\n\n"


def truncate_text(text: str, max_length: int = 1000) -> str:
    """Truncate text to specified length and add ellipsis
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."


def clean_source_citation(source: str) -> str:
    """Clean up source citations
    
    Args:
        source: Original source string
        
    Returns:
        Cleaned source string
    """
    # Remove Document tags
    source = source.replace('<Document source="', '')
    source = source.replace('"/>', '')
    source = source.replace('</Document>', '')
    
    # Remove duplicate spaces
    source = ' '.join(source.split())
    
    return source.strip()


def generate_thread_id() -> str:
    """Generate unique thread ID for checkpointer
    
    Returns:
        UUID-based thread ID
    """
    import uuid
    return str(uuid.uuid4())