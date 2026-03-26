import json
from google import genai
from google.genai import errors, types
from app.core.config import GEMINI_API_KEY, GEMINI_MODEL

# Límite de contenido por artículo para el análisis.
# 3000 chars es suficiente para captar tablas, datos clave y contexto.
# Bajar de 8000 a 3000 reduce tokens ~60% → respuesta mucho más rápida.
_ANALYZE_CAP = 3000


class NewsAnalyzerAgent:
    def __init__(self):
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model = GEMINI_MODEL

    def analyze_many(self, items: list[dict]) -> dict:
        compact_items = []

        for item in items:
            # Solo usamos content limpio — raw_content va directo al frontend para tablas
            content_text = (item.get("content") or item.get("clean_content") or "").strip()

            compact_items.append({
                "id": item.get("id"),
                "title": item.get("title", ""),
                "content": content_text[:_ANALYZE_CAP]
            })

        prompt = f"""Analiza estos artículos y devuelve análisis en ESPAÑOL.
Devuelve SOLO JSON válido. Conserva el id original.
page_type: "comparison" si es comparativa, "article" si no.
Si no hay pricing/features, devuelve listas vacías.
tables_detected: devuelve siempre lista vacía (las tablas se procesan aparte).

INPUT:
{json.dumps(compact_items, ensure_ascii=False)}"""

        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "title": {"type": "string"},
                            "page_type": {"type": "string"},
                            "important_points": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "summary": {"type": "string"},
                            "key_facts": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "pricing_info": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "feature_comparison": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "tables_detected": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Tabla markdown completa con pipes |"
                            },
                            "missing_structured_data": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": [
                            "id",
                            "title",
                            "page_type",
                            "important_points",
                            "summary",
                            "key_facts",
                            "pricing_info",
                            "feature_comparison",
                            "tables_detected",
                            "missing_structured_data"
                        ]
                    }
                }
            },
            "required": ["items"]
        }

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                )
            )

            raw_text = response.text.strip()
            return json.loads(raw_text)

        except errors.ClientError as e:
            return {
                "items": [],
                "error": f"Error de Gemini API: {e}"
            }
        except Exception as e:
            return {
                "items": [],
                "error": f"Error inesperado: {e}"
            }