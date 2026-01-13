
import requests
from typing import Dict, Any, Optional
import logging

try:
    import requests
except ImportError:
    requests = None

from .config import DEFAULT_TAVILY_ENDPOINT

logger = logging.getLogger(__name__)

class TavilyClient:
    """
    Tavily Search API Client.
    """
    def __init__(self, api_key: str, endpoint: str = DEFAULT_TAVILY_ENDPOINT, timeout: int = 15):
        if requests is None:
            raise ImportError("`requests` library is not installed. Please install it with `pip install requests`.")
        self.api_key = api_key
        self.endpoint = endpoint
        self.timeout = timeout

    def search(self, query: str, num_results: int = 5, domain: Optional[str] = None, lang: Optional[str] = None) -> Dict[str, Any]:
        """
        Executes a Tavily search with full content retrieval.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "query": query, 
            "max_results": num_results,
            "include_raw_content": True, # Request full text content
            "search_depth": "advanced"
        }
        
        if domain:
            payload["domain"] = domain
        if lang:
            payload["language"] = lang

        try:
            resp = requests.post(self.endpoint, headers=headers, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"Tavily HTTP error: {e}")
            return {"error": f"HTTP error: {e}"}
        except ValueError as e:
            logger.error(f"Tavily JSON error: {e}")
            return {"error": f"JSON parse error: {e}", "raw": resp.text if 'resp' in locals() else None}
        except Exception as e:
            logger.error(f"Tavily unexpected error: {e}")
            return {"error": str(e)}
