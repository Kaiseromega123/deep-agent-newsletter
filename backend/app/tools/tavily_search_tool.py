from tavily import TavilyClient
from app.core.config import TAVILY_API_KEY


class TavilySearchTool:
    def __init__(self):
        self.client = TavilyClient(api_key=TAVILY_API_KEY)

    def search(self, query: str, tavily_topic: str = "general"):
        response = self.client.search(
            query=query,
            topic=tavily_topic,
            max_results=5,
            country="spain",
            include_raw_content=False,  # no descargar contenido aquí, lo hace extract_pages_batch
        )

        results = response.get("results", [])
        print("NUM SEARCH RESULTS tavily search tool:", len(results))

        clean_results = []
        for item in results:
            clean_results.append({
                "title": item.get("title"),
                "url": item.get("url"),
                "content": item.get("content"),  # snippet corto (~200 chars)
            })

        return clean_results