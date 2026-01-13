
import abc
import json
import logging
from typing import Dict, Any, Optional

try:
    import google.generativeai as genai
except ImportError:
    genai = None

# Configure logging
logger = logging.getLogger(__name__)

class LLMClient(abc.ABC):
    """LLM client interface."""
    @abc.abstractmethod
    def call(self, prompt: str) -> Dict[str, Any]:
        pass

class GeminiClient(LLMClient):
    """Google Gemini Client with JSON repair capabilities."""
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-pro"): 
        if genai is None:
            raise ImportError("`google-generativeai` library is not installed. Please install it with `pip install google-generativeai`.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.generation_config = genai.GenerationConfig(
            response_mime_type="application/json"
        )

    def _extract_json(self, text: str) -> Optional[str]:
        """Extract JSON block from text."""
        if not text:
            return None
        
        start_brace = text.find('{')
        start_bracket = text.find('[')
        
        if start_brace == -1 and start_bracket == -1:
            return None
            
        if start_brace == -1:
            start = start_bracket
        elif start_bracket == -1:
            start = start_brace
        else:
            start = min(start_brace, start_bracket)
            
        end_brace = text.rfind('}')
        end_bracket = text.rfind(']')
        
        if end_brace == -1 and end_bracket == -1:
            return None
            
        end = max(end_brace, end_bracket)
        
        if end <= start:
            return None
            
        potential_json = text[start:end+1]
        return potential_json

    def _get_json_repair_prompt(self, malformed_text: str) -> str:
        """Prompt to repair malformed JSON."""
        return f"""
        # Instruction
        You were previously asked to output in JSON format, but the following text was generated which could not be parsed.

        # Malformed Text
        ```
        {malformed_text}
        ```

        # Task
        Correct the format to be **strictly valid JSON** (starting with `{{` or `[`), containing the **exact same information**.
        Output ONLY the JSON.
        """

    def call(self, prompt: str, is_retry: bool = False) -> Dict[str, Any]:
        """
        prompt -> LLM call -> JSON cleaning -> JSON parsing.
        Retries once if parsing fails.
        """
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            text = getattr(response, "text", None) or getattr(response, "response", None) or str(response)
            
            cleaned_text = self._extract_json(text)
            
            if cleaned_text:
                try:
                    return json.loads(cleaned_text) 
                except Exception as e_clean:
                    logger.warning(f"[GeminiClient] JSON Parse Failed (after clean): {e_clean}")
                    
                    if is_retry:
                        logger.error(f"[GeminiClient] JSON Repair Retry Failed.")
                        return {"raw_text": text, "parse_error": f"Retry failed: {e_clean}"}
                    else:
                        logger.info(f"[GeminiClient] Retrying with repair prompt...")
                        repair_prompt = self._get_json_repair_prompt(text)
                        return self.call(repair_prompt, is_retry=True)
            else:
                logger.warning(f"[GeminiClient] No JSON block found in response.")
                
                if is_retry:
                    logger.error(f"[GeminiClient] No JSON block found after retry.")
                    return {"raw_text": text, "parse_error": "Retry failed: No JSON block found"}
                else:
                    logger.info(f"[GeminiClient] Retrying with repair prompt...")
                    repair_prompt = self._get_json_repair_prompt(text)
                    return self.call(repair_prompt, is_retry=True)
                
        except Exception as e:
            logger.error(f"[GeminiClient] API Error: {e}")
            if is_retry:
                return {"error": f"API call failed during retry: {e}"}
            else:
                return {"error": str(e)}
