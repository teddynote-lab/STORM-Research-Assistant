"""STORM Research Assistant의 도구 정의

이 모듈은 연구 프로세스에서 사용되는 각종 도구들을 정의합니다.
"""

from typing import Optional
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.retrievers import ArxivRetriever
from langchain_core.runnables import RunnableConfig
from storm_research.configuration import Configuration


class SearchTools:
    """연구를 위한 검색 도구들을 관리하는 클래스"""
    
    def __init__(self, config: Optional[RunnableConfig] = None):
        """검색 도구 초기화
        
        Args:
            config: 런타임 설정
        """
        self.configuration = Configuration.from_runnable_config(config)
        
        # Tavily 검색 도구 초기화
        self.tavily_search = TavilySearchResults(
            max_results=self.configuration.tavily_max_results
        )
        
        # ArXiv 검색 도구 초기화
        self.arxiv_retriever = ArxivRetriever(
            load_max_docs=self.configuration.arxiv_max_docs,
            load_all_available_meta=True,
            get_full_documents=True,
        )
    
    async def search_web(self, query: str) -> str:
        """웹에서 정보를 검색
        
        Args:
            query: 검색 쿼리
            
        Returns:
            포맷팅된 검색 결과
        """
        try:
            # Tavily API를 사용하여 웹 검색
            search_results = await self.tavily_search.ainvoke(query)
            
            # 결과를 문서 형식으로 포맷팅
            formatted_results = []
            for doc in search_results:
                formatted_doc = (
                    f'<Document href="{doc["url"]}"/>\n'
                    f'{doc["content"]}\n'
                    f'</Document>'
                )
                formatted_results.append(formatted_doc)
            
            return "\n\n---\n\n".join(formatted_results)
            
        except Exception as e:
            return f"<Error>웹 검색 중 오류 발생: {str(e)}</Error>"
    
    async def search_arxiv(self, query: str) -> str:
        """ArXiv에서 학술 논문 검색
        
        Args:
            query: 검색 쿼리
            
        Returns:
            포맷팅된 검색 결과
        """
        try:
            # ArXiv에서 논문 검색
            arxiv_results = await self.arxiv_retriever.ainvoke(query)
            
            # 결과를 문서 형식으로 포맷팅
            formatted_results = []
            for doc in arxiv_results:
                metadata = doc.metadata
                formatted_doc = (
                    f'<Document source="{metadata["entry_id"]}" '
                    f'date="{metadata.get("Published", "")}" '
                    f'authors="{metadata.get("Authors", "")}"/>\n'
                    f'<Title>\n{metadata["Title"]}\n</Title>\n\n'
                    f'<Summary>\n{metadata["Summary"]}\n</Summary>\n\n'
                    f'<Content>\n{doc.page_content}\n</Content>\n'
                    f'</Document>'
                )
                formatted_results.append(formatted_doc)
            
            return "\n\n---\n\n".join(formatted_results)
            
        except Exception as e:
            return f"<Error>ArXiv 검색 중 오류 발생: {str(e)}</Error>"


# 도구 인스턴스 생성 함수
def get_search_tools(config: Optional[RunnableConfig] = None) -> SearchTools:
    """설정에 따른 검색 도구 인스턴스 반환
    
    Args:
        config: 런타임 설정
        
    Returns:
        SearchTools 인스턴스
    """
    return SearchTools(config)