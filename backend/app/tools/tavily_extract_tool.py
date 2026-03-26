import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tavily import TavilyClient
from app.core.config import TAVILY_API_KEY


class TavilyExtractTool:
    def __init__(self):
        self.client = TavilyClient(api_key=TAVILY_API_KEY)

    def _clean_content(self, text: str) -> str:
        if not text:
            return ""
        noise_patterns = [
            r"Skip to Main Content", r"Skip to\.\.\.", r"Advertisement",
            r"What to Read Next", r"Most Popular News", r"Most Popular Opinion",
            r"Most Popular", r"Further Reading", r"About Us",
            r"Copyright ©.*", r"All Rights Reserved.*", r"Newsletter Sign-up",
            r"Subscribe", r"Sign In", r"Print Edition",
            r"Latest Headlines", r"Videos", r"Audio", r"Puzzles",
        ]
        cleaned = text
        for pattern in noise_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        return cleaned.strip()

    def _extract_single(self, url: str, query: str | None = None) -> dict:
        """Extrae una sola URL. Devuelve dict con contenido o dict vacio si falla."""
        try:
            response = self.client.extract(
                urls=[url],
                extract_depth="basic",
                format="markdown",
                query=query,
            )
            results = response.get("results", [])
            if results:
                item = results[0]
                raw_content = item.get("raw_content", "")
                result = {
                    "url": item.get("url") or url,
                    "title": item.get("title"),
                    "raw_content": raw_content,
                    "clean_content": self._clean_content(raw_content),
                }
                print(f"[extract-single] OK {url[:60]} -> {len(raw_content)} chars")
                return result
            else:
                print(f"[extract-single] Sin resultados para {url[:60]}")
        except Exception as e:
            print(f"[extract-single] Error {url[:60]}: {e}")
        return {"url": url, "title": None, "raw_content": "", "clean_content": ""}

    def extract(self, url: str, query: str | None = None):
        """Extrae una URL."""
        return self._extract_single(url, query=query)

    def extract_many(self, urls: list[str], query: str | None = None) -> list[dict]:
        """Extrae multiples URLs en paralelo, de 1 en 1 para obtener contenido completo."""
        if not urls:
            return []

        print(f"[extract-many] Extrayendo {len(urls)} URLs en paralelo (1 a 1)...")

        output = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Mapear future -> indice original para mantener el orden
            future_to_idx = {}
            for i, url in enumerate(urls):
                future = executor.submit(self._extract_single, url, query)
                future_to_idx[future] = i

            # Recoger resultados preservando orden
            results_by_idx = {}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results_by_idx[idx] = future.result()
                except Exception as e:
                    print(f"[extract-many] Error en URL #{idx+1}: {e}")
                    results_by_idx[idx] = {
                        "url": urls[idx],
                        "title": None,
                        "raw_content": "",
                        "clean_content": "",
                    }

            # Ordenar por indice original
            for i in range(len(urls)):
                output.append(results_by_idx.get(i, {
                    "url": urls[i],
                    "title": None,
                    "raw_content": "",
                    "clean_content": "",
                }))

        ok_count = sum(1 for o in output if len(o.get("raw_content", "")) > 0)
        print(f"[extract-many] Completado: {ok_count}/{len(urls)} URLs con contenido")

        return output