"""STORM Research Assistant의 메인 그래프 정의

이 모듈은 연구 프로세스를 조율하는 LangGraph 그래프를 정의합니다.
"""

from typing import List, Literal, cast
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
    get_buffer_string,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END

# from langgraph.checkpoint.memory import InMemorySaver  # LangGraph API가 자동으로 처리
from langgraph.constants import Send

from storm_research.state import (
    InterviewState,
    InputState,
    OutputState,
    ResearchGraphState,
    Analyst,
    Perspectives,
    SearchQuery,
)
from storm_research.prompts import (
    ANALYST_INSTRUCTIONS,
    QUESTION_INSTRUCTIONS,
    ANSWER_INSTRUCTIONS,
    SEARCH_INSTRUCTIONS,
    SECTION_WRITER_INSTRUCTIONS,
    REPORT_WRITER_INSTRUCTIONS,
    INTRO_CONCLUSION_INSTRUCTIONS,
)
from storm_research.configuration import Configuration
from storm_research.tools import get_search_tools
from storm_research.utils import load_chat_model, generate_thread_id


# ====================== 분석가 생성 노드 ======================


async def create_analysts(state: ResearchGraphState, config: RunnableConfig) -> dict:
    """연구 주제에 맞는 분석가 페르소나를 생성

    각 분석가는 고유한 관점과 전문성을 가지고 연구에 기여합니다.
    """
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.model)

    topic = state.get("topic", "")
    max_analysts = state.get("max_analysts", configuration.max_analysts)

    # 구조화된 출력을 위해 모델 설정
    structured_model = model.with_structured_output(Perspectives)

    # 프롬프트 구성
    system_message = ANALYST_INSTRUCTIONS.format(
        topic=topic,
        human_analyst_feedback="",  # 사용자 피드백은 비어있음
        max_analysts=max_analysts,
    )

    # 분석가 생성
    result = await structured_model.ainvoke(
        [
            SystemMessage(content=system_message),
            HumanMessage(content="Generate the set of analysts."),
        ]
    )

    return {"analysts": result.analysts}


# ====================== 인터뷰 노드 ======================


async def generate_question(state: InterviewState, config: RunnableConfig) -> dict:
    """분석가가 전문가에게 질문 생성

    분석가의 페르소나와 이전 대화 내용을 바탕으로
    통찰력 있는 질문을 생성합니다.
    """
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.model)

    analyst = state["analyst"]
    messages = state["messages"]

    # 질문 생성 프롬프트 구성
    system_message = QUESTION_INSTRUCTIONS.format(goals=analyst.persona)

    # 질문 생성
    question = await model.ainvoke([SystemMessage(content=system_message)] + messages)

    return {"messages": [question]}


async def search_web(state: InterviewState, config: RunnableConfig) -> dict:
    """웹에서 관련 정보 검색

    대화 내용을 분석하여 적절한 검색 쿼리를 생성하고
    웹에서 관련 정보를 검색합니다.
    """
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.model)
    search_tools = get_search_tools(config)

    # 검색 쿼리 생성
    structured_model = model.with_structured_output(SearchQuery)
    search_query = await structured_model.ainvoke(
        [SystemMessage(content=SEARCH_INSTRUCTIONS)] + state["messages"]
    )

    # 웹 검색 수행
    search_results = await search_tools.search_web(search_query.search_query)

    return {"context": [search_results]}


async def search_arxiv(state: InterviewState, config: RunnableConfig) -> dict:
    """ArXiv에서 학술 논문 검색

    대화 내용을 분석하여 적절한 검색 쿼리를 생성하고
    ArXiv에서 관련 논문을 검색합니다.
    """
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.model)
    search_tools = get_search_tools(config)

    # 검색 쿼리 생성
    structured_model = model.with_structured_output(SearchQuery)
    search_query = await structured_model.ainvoke(
        [SystemMessage(content=SEARCH_INSTRUCTIONS)] + state["messages"]
    )

    # ArXiv 검색 수행
    search_results = await search_tools.search_arxiv(search_query.search_query)

    return {"context": [search_results]}


async def generate_answer(state: InterviewState, config: RunnableConfig) -> dict:
    """전문가 역할로 질문에 답변 생성

    검색된 컨텍스트를 바탕으로 전문가 입장에서
    상세하고 정확한 답변을 생성합니다.
    """
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.model)

    analyst = state["analyst"]
    messages = state["messages"]
    context = state["context"]

    # 답변 생성 프롬프트 구성
    system_message = ANSWER_INSTRUCTIONS.format(goals=analyst.persona, context=context)

    # 답변 생성
    answer = await model.ainvoke([SystemMessage(content=system_message)] + messages)

    # 전문가 답변임을 표시
    answer.name = "expert"

    return {"messages": [answer]}


async def save_interview(state: InterviewState) -> dict:
    """완료된 인터뷰 내용 저장

    대화 내용을 문자열로 변환하여 저장합니다.
    """
    messages = state["messages"]
    interview_content = get_buffer_string(messages)

    return {"interview": interview_content}


def route_messages(
    state: InterviewState, name: str = "expert"
) -> Literal["ask_question", "save_interview"]:
    """인터뷰 진행 상황에 따라 다음 단계 결정

    최대 턴 수에 도달했거나 인터뷰가 완료되면 저장하고,
    그렇지 않으면 추가 질문을 생성합니다.
    """
    messages = state["messages"]
    max_num_turns = state.get("max_num_turns", 3)

    # 전문가 답변 횟수 확인
    num_responses = len(
        [m for m in messages if isinstance(m, AIMessage) and m.name == name]
    )

    # 최대 턴 수 도달 확인
    if num_responses >= max_num_turns:
        return "save_interview"

    # 인터뷰 종료 신호 확인
    last_question = messages[-2]
    if "Thank you so much for your help" in last_question.content:
        return "save_interview"

    return "ask_question"


async def write_section(state: InterviewState, config: RunnableConfig) -> dict:
    """인터뷰 내용을 바탕으로 보고서 섹션 작성

    분석가의 관점에서 인터뷰 내용을 정리하여
    보고서의 한 섹션을 작성합니다.
    """
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.model)

    context = state["context"]
    analyst = state["analyst"]

    # 섹션 작성 프롬프트 구성
    system_message = SECTION_WRITER_INSTRUCTIONS.format(focus=analyst.description)

    # 섹션 작성
    section = await model.ainvoke(
        [
            SystemMessage(content=system_message),
            HumanMessage(content=f"Use this source to write your section: {context}"),
        ]
    )

    return {"sections": [section.content]}


# ====================== 보고서 작성 노드 ======================


def initiate_all_interviews(state: ResearchGraphState) -> List[Send]:
    """모든 분석가의 인터뷰를 동시에 시작

    각 분석가별로 독립적인 인터뷰 프로세스를 시작합니다.
    """
    topic = state.get("topic", "")

    # 각 분석가별로 인터뷰 시작
    return [
        Send(
            "conduct_interview",
            {
                "analyst": analyst,
                "messages": [
                    HumanMessage(
                        content=f"So you said you were writing an article on {topic}?"
                    )
                ],
                "max_num_turns": state.get("max_num_turns", 3),
            },
        )
        for analyst in state["analysts"]
    ]


async def write_report(state: ResearchGraphState, config: RunnableConfig) -> dict:
    """모든 섹션을 통합하여 보고서 본문 작성

    각 분석가가 작성한 섹션들을 종합하여
    일관성 있는 보고서로 통합합니다.
    """
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.model)

    sections = state["sections"]
    topic = state.get("topic", "")

    # 모든 섹션 연결
    formatted_sections = "\n\n".join(sections)

    # 보고서 작성 프롬프트 구성
    system_message = REPORT_WRITER_INSTRUCTIONS.format(
        topic=topic, context=formatted_sections
    )

    # 보고서 작성
    report = await model.ainvoke(
        [
            SystemMessage(content=system_message),
            HumanMessage(content="Write a report based upon these memos."),
        ]
    )

    return {"content": report.content}


async def write_introduction(state: ResearchGraphState, config: RunnableConfig) -> dict:
    """보고서 서론 작성

    전체 연구 내용을 요약하고 독자의 관심을 끄는
    매력적인 서론을 작성합니다.
    """
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.model)

    sections = state["sections"]
    topic = state.get("topic", "")

    # 모든 섹션 연결
    formatted_sections = "\n\n".join(sections)

    # 서론 작성 프롬프트 구성
    instructions = INTRO_CONCLUSION_INSTRUCTIONS.format(
        topic=topic, formatted_str_sections=formatted_sections
    )

    # 서론 작성
    intro = await model.ainvoke(
        [instructions, HumanMessage(content="Write the report introduction")]
    )

    return {"introduction": intro.content}


async def write_conclusion(state: ResearchGraphState, config: RunnableConfig) -> dict:
    """보고서 결론 작성

    연구의 주요 발견사항을 요약하고
    향후 연구 방향을 제시하는 결론을 작성합니다.
    """
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.model)

    sections = state["sections"]
    topic = state.get("topic", "")

    # 모든 섹션 연결
    formatted_sections = "\n\n".join(sections)

    # 결론 작성 프롬프트 구성
    instructions = INTRO_CONCLUSION_INSTRUCTIONS.format(
        topic=topic, formatted_str_sections=formatted_sections
    )

    # 결론 작성
    conclusion = await model.ainvoke(
        [instructions, HumanMessage(content="Write the report conclusion")]
    )

    return {"conclusion": conclusion.content}


async def finalize_report(state: ResearchGraphState) -> dict:
    """최종 보고서 조립

    서론, 본문, 결론을 하나로 합쳐
    완성된 보고서를 생성합니다.
    """
    content = state["content"]

    # "## Insights" 제목 제거
    if content.startswith("## Insights"):
        content = content.strip("## Insights")

    # Sources 섹션 분리
    sources = None
    if "## Sources" in content:
        try:
            content, sources = content.split("\n## Sources\n")
        except:
            sources = None

    # 최종 보고서 조립
    final_report = (
        state["introduction"]
        + "\n\n---\n\n## Main Idea\n\n"
        + content
        + "\n\n---\n\n"
        + state["conclusion"]
    )

    # Sources 섹션 추가
    if sources is not None:
        final_report += "\n\n## Sources\n" + sources

    return {"final_report": final_report}


# ====================== 그래프 빌드 함수 ======================


def build_interview_graph():
    """인터뷰 서브그래프 생성

    단일 분석가의 인터뷰 프로세스를 관리하는
    서브그래프를 생성합니다.
    """
    builder = StateGraph(InterviewState)

    # 노드 추가
    builder.add_node("ask_question", generate_question)
    builder.add_node("search_web", search_web)
    builder.add_node("search_arxiv", search_arxiv)
    builder.add_node("answer_question", generate_answer)
    builder.add_node("save_interview", save_interview)
    builder.add_node("write_section", write_section)

    # 엣지 정의
    builder.add_edge(START, "ask_question")
    builder.add_edge("ask_question", "search_web")
    builder.add_edge("ask_question", "search_arxiv")
    builder.add_edge("search_web", "answer_question")
    builder.add_edge("search_arxiv", "answer_question")
    builder.add_conditional_edges(
        "answer_question", route_messages, ["ask_question", "save_interview"]
    )
    builder.add_edge("save_interview", "write_section")
    builder.add_edge("write_section", END)

    # LangGraph API가 체크포인터를 자동으로 관리
    interview_graph = builder.compile().with_config(run_name="Conduct Interview")

    return interview_graph


def build_research_graph():
    """메인 연구 그래프 생성

    전체 연구 프로세스를 조율하는
    메인 그래프를 생성합니다.
    """
    # 인터뷰 서브그래프 생성
    interview_graph = build_interview_graph()

    # 메인 그래프 빌더 - 입력/출력 스키마 지정
    builder = StateGraph(ResearchGraphState, input=InputState, output=OutputState)

    # 노드 추가
    builder.add_node("create_analysts", create_analysts)
    builder.add_node("conduct_interview", interview_graph)
    builder.add_node("write_report", write_report)
    builder.add_node("write_introduction", write_introduction)
    builder.add_node("write_conclusion", write_conclusion)
    builder.add_node("finalize_report", finalize_report)

    # 엣지 정의
    builder.add_edge(START, "create_analysts")
    builder.add_conditional_edges(
        "create_analysts", initiate_all_interviews, ["conduct_interview"]
    )

    # 보고서 작성 단계
    builder.add_edge("conduct_interview", "write_report")
    builder.add_edge("conduct_interview", "write_introduction")
    builder.add_edge("conduct_interview", "write_conclusion")

    # 최종 보고서 생성
    builder.add_edge(
        ["write_conclusion", "write_report", "write_introduction"], "finalize_report"
    )
    builder.add_edge("finalize_report", END)

    # LangGraph API가 체크포인터를 자동으로 관리
    graph = builder.compile()

    return graph


# ====================== 메인 그래프 인스턴스 ======================

# LangGraph Studio에서 사용할 그래프 인스턴스
graph = build_research_graph()
