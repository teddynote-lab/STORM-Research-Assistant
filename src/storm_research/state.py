"""STORM Research Assistant State Definitions

This module defines the states used at each stage of the research process.
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

# ====================== Data Models ======================


class Analyst(BaseModel):
    """Class defining analyst attributes and metadata

    Each analyst has unique perspectives and expertise.
    """

    # Primary affiliation information
    affiliation: str = Field(description="Analyst's primary organization")
    # Name
    name: str = Field(description="Analyst's name")
    # Role
    role: str = Field(description="Analyst's role related to the topic")
    # Description of focus, concerns, and motivations
    description: str = Field(description="Description of analyst's interests, concerns, and motivations")

    @property
    def persona(self) -> str:
        """Return analyst's persona as string"""
        return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n"


class Perspectives(BaseModel):
    """Class representing a collection of analysts"""

    analysts: List[Analyst] = Field(
        description="Comprehensive list of analysts including roles and affiliations"
    )


class SearchQuery(BaseModel):
    """Data class for search queries"""

    search_query: str = Field(None, description="Search query for information retrieval")


# ====================== State Definitions ======================


@dataclass
class InputState:
    """Schema for graph input"""

    # Research topic
    messages: Annotated[Sequence[AnyMessage], add_messages] = field(
        default_factory=list
    )


@dataclass
class OutputState:
    """Schema for graph output"""

    # Completed final report
    final_report: str


class GenerateAnalystsState(InputState):
    """State for analyst generation phase"""

    # Research topic
    topic: str
    # Maximum number of analysts to generate
    max_analysts: int
    # Feedback received from user
    human_analyst_feedback: Optional[str]
    # Generated analyst list
    analysts: List[Analyst]


class InterviewState(MessagesState):
    """State for interview phase

    Inherits from MessagesState to automatically manage conversation history.
    """

    # Conversation turns
    max_num_turns: int
    # Context list containing source documents
    context: Annotated[list, operator.add]
    # Analyst currently being interviewed
    analyst: Analyst
    # String storing interview content
    interview: str
    # List of written report sections
    sections: list


class ResearchGraphState(TypedDict):
    """Internal state of the entire research process"""

    # Research topic
    topic: str
    # Maximum number of analysts to generate
    max_analysts: int
    # User's analyst feedback
    human_analyst_feedback: Optional[str]
    # Generated analyst list
    analysts: List[Analyst]
    # Sections written by each analyst
    sections: Annotated[list, operator.add]
    # Introduction of final report
    introduction: str
    # Body content of final report
    content: str
    # Conclusion of final report
    conclusion: str
    # Completed final report
    final_report: str

    # Research topic
    messages: Annotated[Sequence[AnyMessage], add_messages] = field(
        default_factory=list
    )
