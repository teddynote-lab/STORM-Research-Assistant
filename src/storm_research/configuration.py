"""STORM Research Assistant의 설정 관리

이 모듈은 연구 보조 시스템의 런타임 설정을 관리합니다.
"""

from dataclasses import dataclass, field
from typing import Optional
from langchain_core.runnables import RunnableConfig


@dataclass
class Configuration:
    """STORM Research Assistant의 설정 클래스

    LangGraph Studio에서 사용할 수 있는 설정 옵션들을 정의합니다.
    """

    # 모델 설정
    model: str = field(
        default="azure/gpt-4.1",
        metadata={
            "description": "사용할 LLM 모델 (provider/model 형식)",
            "examples": [
                "azure/gpt-4.1",
                "openai/gpt-4.1",
                "anthropic/claude-3-5-sonnet-20240620",
            ],
        },
    )

    # 연구 설정
    max_analysts: int = field(
        default=3, metadata={"description": "생성할 최대 분석가 수", "range": [1, 10]}
    )

    max_interview_turns: int = field(
        default=3,
        metadata={"description": "인터뷰당 최대 대화 턴 수", "range": [1, 10]},
    )

    # 검색 설정
    tavily_max_results: int = field(
        default=3,
        metadata={"description": "Tavily 검색 결과 최대 개수", "range": [1, 10]},
    )

    arxiv_max_docs: int = field(
        default=3,
        metadata={"description": "ArXiv 검색 문서 최대 개수", "range": [1, 10]},
    )

    # 병렬 처리 설정
    parallel_interviews: bool = field(
        default=True, metadata={"description": "인터뷰를 병렬로 실행할지 여부"}
    )

    # 체크포인터 설정
    enable_checkpointing: bool = field(
        default=True, metadata={"description": "상태 체크포인팅 활성화 여부"}
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """RunnableConfig에서 설정을 추출하여 Configuration 인스턴스 생성

        Args:
            config: LangGraph에서 전달되는 런타임 설정

        Returns:
            설정이 적용된 Configuration 인스턴스
        """
        configurable = config.get("configurable", {}) if config else {}

        # 기본 인스턴스를 생성하여 기본값 가져오기
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
