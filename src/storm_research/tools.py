"""STORM Research Assistant Tool Definitions

This module defines various tools used in the research process.
"""

from typing import Optional
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.retrievers import ArxivRetriever
from langchain_core.runnables import RunnableConfig
from storm_research.configuration import Configuration


class SearchTools:
    """Class managing search tools for research"""
    
    def __init__(self, config: Optional[RunnableConfig] = None):
        """Initialize search tools
        
        Args:
            config: Runtime configuration
        """
        self.configuration = Configuration.from_runnable_config(config)
        
        # Initialize Tavily search tool
        self.tavily_search = TavilySearchResults(
            max_results=self.configuration.tavily_max_results
        )
        
        # Initialize ArXiv search tool
        self.arxiv_retriever = ArxivRetriever(
            load_max_docs=self.configuration.arxiv_max_docs,
            load_all_available_meta=True,
            get_full_documents=True,
        )
    
    async def search_web(self, query: str) -> str:
        """Search for information on the web
        
        Args:
            query: Search query
            
        Returns:
            Formatted search results
        """
        try:
            # Search the web using Tavily API
            search_results = await self.tavily_search.ainvoke(query)
            
            # Format results as documents
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
            return f"<Error>Error occurred during web search: {str(e)}</Error>"
    
    async def search_arxiv(self, query: str) -> str:
        """Search for academic papers on ArXiv
        
        Args:
            query: Search query
            
        Returns:
            Formatted search results
        """
        try:
            # Search papers on ArXiv
            arxiv_results = await self.arxiv_retriever.ainvoke(query)
            
            # Format results as documents
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
            return f"<Error>Error occurred during ArXiv search: {str(e)}</Error>"


# Tool instance creation function
def get_search_tools(config: Optional[RunnableConfig] = None) -> SearchTools:
    """Return search tool instance based on configuration
    
    Args:
        config: Runtime configuration
        
    Returns:
        SearchTools instance
    """
    return SearchTools(config)