import asyncio
import aiohttp
import time
from typing import List, Dict, Any
from bs4 import BeautifulSoup
from ddgs import DDGS
from .base_tool import BaseTool
from utils import ColorPrint


class AsyncSearchTool(BaseTool):
    """High-performance async search tool with connection pooling and caching"""

    def __init__(self, config: dict):
        self.config = config
        self.session = None
        self._cache = {}  # Simple in-memory cache
        self.cache_ttl = 300  # 5 minutes cache TTL

    @property
    def name(self) -> str:
        return "search_web_async"

    @property
    def description(self) -> str:
        return "Search the web asynchronously using DuckDuckGo for current information with high performance"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query to find information on the web",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of search results to return",
                    "default": 5,
                },
            },
            "required": ["query"],
        }

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session with connection pooling"""
        if self.session is None or self.session.closed:
            connector = aiohttp.TCPConnector(
                limit=50,  # Total connection pool size
                limit_per_host=10,  # Per-host connection limit
                ttl_dns_cache=300,  # DNS cache TTL
                use_dns_cache=True,
                keepalive_timeout=30,
            )

            timeout = aiohttp.ClientTimeout(total=15, connect=5)
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    "User-Agent": self.config.get("search", {}).get(
                        "user_agent", "Mozilla/5.0 (compatible; AsyncSearchBot)"
                    )
                },
            )
        return self.session

    def _get_cache_key(self, query: str, max_results: int) -> str:
        """Generate cache key for search results"""
        import hashlib

        return hashlib.md5(f"{query}_{max_results}".encode()).hexdigest()

    def _is_cache_valid(self, timestamp: float) -> bool:
        """Check if cache entry is still valid"""
        return time.time() - timestamp < self.cache_ttl

    async def _fetch_page_content(
        self, session: aiohttp.ClientSession, url: str
    ) -> str:
        """Fetch and parse page content asynchronously"""
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    html = await response.text()

                    # Parse HTML with BeautifulSoup
                    soup = BeautifulSoup(html, "html.parser")

                    # Remove script and style elements
                    for script in soup(["script", "style", "nav", "footer", "header"]):
                        script.decompose()

                    # Get text content
                    text = soup.get_text()
                    # Clean up whitespace
                    text = " ".join(text.split())

                    # Limit content length for performance
                    return text[:1500] + "..." if len(text) > 1500 else text
                else:
                    return f"HTTP {response.status}: Could not fetch content"

        except asyncio.TimeoutError:
            return "Timeout: Could not fetch content within time limit"
        except Exception as e:
            return f"Error fetching content: {str(e)}"

    async def execute(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search the web asynchronously with caching and connection pooling"""
        start_time = time.time()

        # Check cache first
        cache_key = self._get_cache_key(query, max_results)
        if cache_key in self._cache:
            cached_data, timestamp = self._cache[cache_key]
            if self._is_cache_valid(timestamp):
                ColorPrint.cache(f"Cache hit for search query: {query}")
                return cached_data

        try:
            # Perform DuckDuckGo search (this is still sync, but fast)
            ddgs = DDGS()
            search_results = list(ddgs.text(query, max_results=max_results))

            if not search_results:
                return [{"error": "No search results found"}]

            # Get HTTP session
            session = await self._get_session()

            # Create tasks for fetching page content concurrently
            content_tasks = []
            for result in search_results:
                task = self._fetch_page_content(session, result["href"])
                content_tasks.append(task)

            # Fetch all page contents concurrently with timeout
            try:
                page_contents = await asyncio.wait_for(
                    asyncio.gather(*content_tasks, return_exceptions=True),
                    timeout=20.0,  # Overall timeout for all requests
                )
            except asyncio.TimeoutError:
                # If overall timeout, use partial results
                page_contents = ["Timeout: Could not fetch content"] * len(
                    search_results
                )

            # Build simplified results
            simplified_results = []
            for i, result in enumerate(search_results):
                content = (
                    page_contents[i]
                    if i < len(page_contents)
                    else "Could not fetch content"
                )

                # Handle exceptions from gather
                if isinstance(content, Exception):
                    content = f"Error: {str(content)}"

                simplified_results.append(
                    {
                        "title": result["title"],
                        "url": result["href"],
                        "snippet": result["body"],
                        "content": content,
                        "relevance_score": 1.0 - (i * 0.1),  # Simple relevance scoring
                    }
                )

            # Cache the results
            self._cache[cache_key] = (simplified_results, time.time())

            # Clean old cache entries (simple cleanup)
            if len(self._cache) > 100:
                # Remove oldest entries
                sorted_cache = sorted(self._cache.items(), key=lambda x: x[1][1])
                for key, _ in sorted_cache[:20]:  # Remove 20 oldest
                    del self._cache[key]

            execution_time = time.time() - start_time
            ColorPrint.performance(
                f"Search completed in {execution_time:.2f}s for query: {query}"
            )

            return simplified_results

        except Exception as e:
            ColorPrint.error(f"Search failed for query '{query}': {str(e)}")
            return [{"error": f"Search failed: {str(e)}"}]

    async def close(self):
        """Close the HTTP session"""
        if self.session and not self.session.closed:
            await self.session.close()

    def __del__(self):
        """Cleanup on deletion"""
        if self.session and not self.session.closed:
            # Schedule cleanup if event loop is running
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.session.close())
            except RuntimeError:
                pass  # Event loop not running
