"""STORM Research Assistant의 상태 정의

이 모듈은 연구 프로세스의 각 단계에서 사용되는 상태를 정의합니다.
"""

import operator
from dataclasses import dataclass, field
from typing import List, Annotated, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from typing import Sequence
from typing_extensions import Annotated

# ====================== 데이터 모델 ======================


class Analyst(BaseModel):
    """분석가의 속성과 메타데이터를 정의하는 클래스

    각 분석가는 고유한 관점과 전문성을 가지고 있습니다.
    """

    # 주요 소속 정보
    affiliation: str = Field(description="분석가의 주요 소속 기관")
    # 이름
    name: str = Field(description="분석가의 이름")
    # 역할
    role: str = Field(description="주제와 관련된 분석가의 역할")
    # 중점, 우려 사항 및 동기에 대한 설명
    description: str = Field(description="분석가의 관심사, 우려사항, 동기에 대한 설명")

    @property
    def persona(self) -> str:
        """분석가의 페르소나를 문자열로 반환"""
        return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n"


class Perspectives(BaseModel):
    """분석가들의 집합을 나타내는 클래스"""

    analysts: List[Analyst] = Field(
        description="역할과 소속을 포함한 분석가들의 종합 목록"
    )


class SearchQuery(BaseModel):
    """검색 쿼리를 위한 데이터 클래스"""

    search_query: str = Field(None, description="정보 검색을 위한 검색 쿼리")


# ====================== 상태 정의 ======================


@dataclass
class InputState:
    """그래프 입력을 위한 스키마"""

    # 연구 주제
    messages: Annotated[Sequence[AnyMessage], add_messages] = field(
        default_factory=list
    )


@dataclass
class OutputState:
    """그래프 출력을 위한 스키마"""

    # 완성된 최종 보고서
    final_report: str


class GenerateAnalystsState(InputState):
    """분석가 생성 단계의 상태"""

    # 연구 주제
    topic: str
    # 생성할 분석가의 최대 수
    max_analysts: int
    # 사용자로부터 받은 피드백
    human_analyst_feedback: Optional[str]
    # 생성된 분석가 목록
    analysts: List[Analyst]


class InterviewState(MessagesState):
    """인터뷰 단계의 상태

    MessagesState를 상속받아 대화 내역을 자동으로 관리합니다.
    """

    # 대화 턴수
    max_num_turns: int
    # 소스 문서를 포함하는 컨텍스트 리스트
    context: Annotated[list, operator.add]
    # 현재 인터뷰 중인 분석가
    analyst: Analyst
    # 인터뷰 내용을 저장하는 문자열
    interview: str
    # 작성된 보고서 섹션 리스트
    sections: list


class ResearchGraphState(TypedDict):
    """전체 연구 프로세스의 내부 상태"""

    # 연구 주제
    topic: str
    # 생성할 분석가의 최대 수
    max_analysts: int
    # 사용자의 분석가 피드백
    human_analyst_feedback: Optional[str]
    # 생성된 분석가 목록
    analysts: List[Analyst]
    # 각 분석가가 작성한 섹션들
    sections: Annotated[list, operator.add]
    # 최종 보고서의 서론
    introduction: str
    # 최종 보고서의 본문 내용
    content: str
    # 최종 보고서의 결론
    conclusion: str
    # 완성된 최종 보고서
    final_report: str

    # 연구 주제
    messages: Annotated[Sequence[AnyMessage], add_messages] = field(
        default_factory=list
    )
