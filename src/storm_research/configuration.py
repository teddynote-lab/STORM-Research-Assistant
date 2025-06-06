"""STORM Research Assistant Configuration Management

This module manages runtime configuration for the research assistant system.
"""

from dataclasses import dataclass, field
from typing import Optional
from langchain_core.runnables import RunnableConfig


@dataclass
class Configuration:
    """STORM Research Assistant Configuration Class

    Defines configuration options available in LangGraph Studio.
    """

    # Model Settings
    model: str = field(
        default="azure/gpt-4.1",
        metadata={
            "description": "LLM model to use (provider/model format)",
            "examples": [
                "azure/gpt-4.1",
                "openai/gpt-4.1",
                "openai/gpt-4.1-mini",
                "anthropic/claude-opus-4-20250514",
                "anthropic/claude-3-7-sonnet-latest",
                "anthropic/claude-3-5-haiku-latest",
            ],
        },
    )

    # Research Settings
    max_analysts: int = field(
        default=3,
        metadata={
            "description": "Maximum number of analysts to generate",
            "range": [1, 10],
        },
    )

    max_interview_turns: int = field(
        default=3,
        metadata={
            "description": "Maximum conversation turns per interview",
            "range": [1, 10],
        },
    )

    # Search Settings
    tavily_max_results: int = field(
        default=3,
        metadata={
            "description": "Maximum number of Tavily search results",
            "range": [1, 10],
        },
    )

    arxiv_max_docs: int = field(
        default=3,
        metadata={
            "description": "Maximum number of ArXiv search documents",
            "range": [1, 10],
        },
    )

    # Parallel Processing Settings
    parallel_interviews: bool = field(
        default=True, metadata={"description": "Whether to run interviews in parallel"}
    )

    # Checkpointer Settings
    enable_checkpointing: bool = field(
        default=True, metadata={"description": "Whether to enable state checkpointing"}
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create Configuration instance by extracting settings from RunnableConfig

        Args:
            config: Runtime configuration passed from LangGraph

        Returns:
            Configuration instance with applied settings
        """
        configurable = config.get("configurable", {}) if config else {}

        # Create default instance to get default values
        defaults = cls()

        return cls(
            model=configurable.get("model", defaults.model),
            max_analysts=configurable.get("max_analysts", defaults.max_analysts),
            max_interview_turns=configurable.get(
                "max_interview_turns", defaults.max_interview_turns
            ),
            tavily_max_results=configurable.get(
                "tavily_max_results", defaults.tavily_max_results
            ),
            arxiv_max_docs=configurable.get("arxiv_max_docs", defaults.arxiv_max_docs),
            parallel_interviews=configurable.get(
                "parallel_interviews", defaults.parallel_interviews
            ),
            enable_checkpointing=configurable.get(
                "enable_checkpointing", defaults.enable_checkpointing
            ),
        )
