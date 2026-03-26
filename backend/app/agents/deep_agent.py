from __future__ import annotations

import json
import operator
import uuid
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from app.tools.tavily_search_tool import TavilySearchTool
from app.tools.tavily_extract_tool import TavilyExtractTool
from app.agents.news_analyzer_agent import NewsAnalyzerAgent
from app.core.config import GEMINI_MODEL


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


class DeepAgent:
    def __init__(self):
        self.search_tool_impl = TavilySearchTool()
        self.extract_tool_impl = TavilyExtractTool()
        self.analyzer_agent_impl = NewsAnalyzerAgent()

        @tool
        def search_news(query: str) -> str:
            """Busca noticias en Tavily sobre un tema dado."""
            print("Usando herramienta de busqueda")
            results = self.search_tool_impl.search(
                query=query,
                tavily_topic="general"
            )
            # Guardar los resultados de búsqueda para poder completar después
            _tid = getattr(self, "_current_thread_id", "default")
            if not hasattr(self, "_search_results"):
                self._search_results = {}
            self._search_results[_tid] = results

            urls_only = [{"url": r["url"], "title": r["title"]} for r in results]
            return json.dumps({
                "urls": urls_only,
                "count": len(urls_only),
                "instruction": f"DEBES pasar TODAS las {len(urls_only)} URLs a extract_pages_batch. No filtres ninguna."
            }, ensure_ascii=False)

        @tool
        def extract_pages_batch(urls_array: list[str], query: str | None = None) -> str:
            """Extrae el contenido de VARIAS URLs a la vez (más rápido que una por una).
            urls_array debe ser una LISTA de strings, p.ej.: ["https://...", "https://..."]
            """
            print("Usando herramienta de extraccion en batch")
            
            # En caso de que el LLM mande un string con formato de lista en lugar de la lista en sí
            if isinstance(urls_array, str):
                try:
                    urls = json.loads(urls_array)
                except Exception:
                    try:
                        import ast
                        urls = ast.literal_eval(urls_array)
                    except Exception:
                        return json.dumps({"error": "urls_array no es una lista válida"}, ensure_ascii=False)
            else:
                urls = urls_array

            print(f"[extract] URLs recibidas para extraer: {len(urls)}")
            if query:
                print(f"[extract] LLM pasó query='{query}' — IGNORANDO para obtener contenido completo")
            for i, u in enumerate(urls, 1):
                print(f"[extract]   {i}. {u[:80]}")

            # IMPORTANTE: no pasar query a extract_many — con query, Tavily filtra/trunca el contenido
            extracted = self.extract_tool_impl.extract_many(urls=urls, query=None)

            print(f"[extract] Tavily devolvió {len(extracted)} resultados de {len(urls)} URLs")
            for i, item in enumerate(extracted, 1):
                raw_len = len(item.get("raw_content") or "")
                clean_len = len(item.get("clean_content") or "")
                raw = item.get("raw_content") or ""
                has_pipes = raw.count("|")
                table_lines = [l for l in raw.split("\n") if l.strip().startswith("|") and l.strip().endswith("|")]
                print(f"[extract]   {i}. {item.get('url', '???')[:60]} | raw={raw_len} clean={clean_len} | pipes={has_pipes} table_lines={len(table_lines)}")
                if table_lines:
                    print(f"[extract]     Ejemplo tabla: {table_lines[0][:120]}")

            # ── FIX: crear entries de fallback para URLs que Tavily no pudo extraer ──
            _tid = getattr(self, "_current_thread_id", "default")
            search_results = getattr(self, "_search_results", {}).get(_tid, [])
            extracted_urls = {item.get("url", "") for item in extracted}

            def _n(u):
                u = (u or "").strip()
                if "://" in u:
                    s, r = u.split("://", 1); u = s.lower() + "://" + r
                return u.rstrip("/")

            extracted_urls_norm = {_n(u) for u in extracted_urls}

            for sr in search_results:
                sr_url = sr.get("url", "")
                if _n(sr_url) not in extracted_urls_norm and sr_url:
                    snippet = sr.get("content", "") or ""
                    fallback_item = {
                        "url": sr_url,
                        "title": sr.get("title") or sr_url,
                        "raw_content": snippet,
                        "clean_content": snippet,
                    }
                    extracted.append(fallback_item)
                    extracted_urls_norm.add(_n(sr_url))
                    print(f"[extract] FALLBACK añadido: {sr_url[:60]} (snippet {len(snippet)} chars)")

            # Guardamos el contenido completo en memoria interna
            if not hasattr(self, "_extract_store"):
                self._extract_store = {}
            by_url = {item["url"]: item for item in extracted if item.get("url")}
            indexed = {**by_url, **{_n(k): v for k, v in by_url.items()}}
            for i, item in enumerate(extracted, 1):
                indexed[str(i)] = item  # fallback por posición
            self._extract_store[_tid] = indexed

            print(f"[extract] Store guardado con {len(indexed)} keys: {[k for k in indexed if k.isdigit()]}")

            summary = [
                {
                    "url": item.get("url"),
                    "title": item.get("title") or item.get("url"),
                    "content_chars": len(item.get("raw_content") or ""),
                }
                for item in extracted
            ]
            return json.dumps({"extracted": summary, "count": len(summary)}, ensure_ascii=False)

        @tool
        def analyze_news_batch(items_json: str) -> str:
            
            """
            Analiza una lista de páginas extraídas.
            items_json debe ser un JSON string con una lista de items (solo url y title):
            [{"id": 1, "url": "https://...", "title": "..."}]
            ¡AVISO! NO envíes el campo 'content' ni 'raw_content', será inyectado por el sistema desde la extracción previa.
            """
            print("Usando herramienta de analisis")
            try:
                items = json.loads(items_json)
            except Exception:
                return json.dumps(
                    {"items": [], "error": "items_json no es JSON válido"},
                    ensure_ascii=False
                )

            print(f"[analyze] Items recibidos para analizar: {len(items)}")
            for it in items:
                print(f"[analyze]   id={it.get('id')} url={it.get('url', '???')[:60]}")

            # Enriquecer cada item con el contenido completo guardado en _extract_store
            _tid = getattr(self, "_current_thread_id", "default")
            store = getattr(self, "_extract_store", {}).get(_tid, {})

            print(f"[analyze] Store tiene {len(store)} keys, posiciones: {sorted(k for k in store if k.isdigit())}")

            def _n(u):
                u = (u or "").strip()
                if "://" in u:
                    s, r = u.split("://", 1); u = s.lower() + "://" + r
                return u.rstrip("/")

            enriched = []
            for item in items:
                url = item.get("url", "")
                stored = store.get(url) or store.get(_n(url)) or {}
                content = item.get("content") or stored.get("clean_content") or ""
                enriched.append({
                    **item,
                    "content": content,
                })
                print(f"[analyze]   enrich id={item.get('id')}: content_len={len(content)} from_store={'yes' if stored else 'NO'}")

            # El analyzer solo recibe content limpio, nunca raw_content
            print(f"[analyze] Llamando a analyzer con {len(enriched)} items...")
            analysis = self.analyzer_agent_impl.analyze_many(enriched)
            print(f"[analyze] Analyzer devolvió {len(analysis.get('items', []))} items analizados")

            # Guardar análisis indexados por URL para poder completar resultados faltantes
            if not hasattr(self, "_analysis_store"):
                self._analysis_store = {}
            analysis_by_url = {}
            for anal_item in analysis.get("items", []):
                # Buscar URL correspondiente en enriched por id
                matching = [e for e in enriched if e.get("id") == anal_item.get("id")]
                if matching:
                    a_url = matching[0].get("url", "")
                    analysis_by_url[a_url] = anal_item
                    analysis_by_url[_n(a_url)] = anal_item
            self._analysis_store[_tid] = analysis_by_url

            json_output = json.dumps(analysis, ensure_ascii=False)
            return json_output + "\n\n[SYSTEM: Análisis completado. DEVUELVE EL JSON FINAL AHORA.]"

        self.tools = [search_news, extract_pages_batch, analyze_news_batch]
        self.tools_by_name = {tool.name: tool for tool in self.tools}

        self.model = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            temperature=0.2,
            # Desactivar thinking — sin esto 2.5-flash puede tardar minutos razonando
            thinking_budget=0,
        ).bind_tools(self.tools)

        self.system = """Eres un agente de investigación con memoria por thread. Responde en ESPAÑOL.

HAY DOS FLUJOS POSIBLES — elige el correcto según el input:

FLUJO A — El usuario manda un TOPIC (buscar noticias):
1. search_news → busca noticias y obtén URLs
2. extract_pages_batch → extrae el contenido de TODAS las URLs en UNA SOLA llamada (pasa todas las URLs juntas en urls_array)
3. analyze_news_batch → analiza TODOS los items en UNA SOLA llamada
4. Devuelve el JSON final

FLUJO B — El usuario manda una URL directa:
1. extract_pages_batch → extrae el contenido con urls_array=["<url>"]
2. analyze_news_batch → analiza el item extraído en UNA SOLA llamada
3. Devuelve el JSON final

REGLAS ABSOLUTAS:
- NUNCA te saltes analyze_news_batch. Siempre es el penúltimo paso antes del JSON final.
- NUNCA te saltes extract_pages_batch cuando hay URLs disponibles.
- extract_pages_batch acepta TODAS las URLs a la vez — NO la llames múltiples veces, pásalas todas en una sola llamada.
- Para analyze_news_batch, construye el items_json con: id (empieza en 1), title (usa el título del extract o la URL si no hay), url. NO incluyas content ni raw_content — el sistema los inyecta automáticamente desde la extracción.
- Si el usuario solo conversa o hace preguntas de seguimiento: responde sin herramientas usando el contexto del hilo.
- No inventes datos. Usa siempre las herramientas.
- En cada result, el campo raw_content puedes devolverlo vacío (""), el sistema lo rellenará automáticamente con el contenido original de la extracción para que el frontend pueda renderizar tablas.
- En el FLUJO A incluye TODOS los resultados extraídos en el JSON final (mínimo 5). NO filtres ni reduzcas la lista de resultados.
- ¡MUY IMPORTANTE! DESPUÉS DE RECIBIR LA RESPUESTA DE `analyze_news_batch`, DEBES GENERAR INMEDIATAMENTE EL JSON FINAL. ¡NO VUELVAS A LLAMAR A NINGUNA HERRAMIENTA!

El JSON final debe seguir este formato exacto:
{"topic":"string","results":[{"id":1,"title":"string","url":"string","content":"string","raw_content":"","analysis":{"page_type":"string","important_points":[],"summary":"string","key_facts":[],"pricing_info":[],"feature_comparison":[],"tables_detected":[],"missing_structured_data":[],"error":null}}],"error":null,"assistant_message":"string"}
Si no aplica búsqueda ni extracción, devuelve results vacío y responde en assistant_message.
Responde SOLO con JSON válido en el mensaje final. No añadas texto fuera del JSON.""".strip()

        self.memory = MemorySaver()

        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_model)
        graph.add_node("action", self.take_action)

        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "action", False: END},
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")

        self.graph = graph.compile(checkpointer=self.memory)
        self.default_topic = "conversation"

    def call_model(self, state: AgentState):
        messages = state["messages"]
        final_messages = [SystemMessage(content=self.system)] + messages
        response = self.model.invoke(final_messages)
        return {"messages": [response]}

    def exists_action(self, state: AgentState):
        last_message = state["messages"][-1]
        tool_calls = getattr(last_message, "tool_calls", [])
        return len(tool_calls) > 0

    def take_action(self, state: AgentState):
        tool_calls = state["messages"][-1].tool_calls
        results = []

        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            # ── FIX: Si el LLM llama a extract_pages_batch, forzar TODAS las URLs ──
            # El LLM a veces filtra y solo pasa 2 de 5 URLs. Aquí inyectamos las que falten.
            if tool_name == "extract_pages_batch":
                _tid = getattr(self, "_current_thread_id", "default")
                search_results = getattr(self, "_search_results", {}).get(_tid, [])
                if search_results:
                    all_search_urls = [r["url"] for r in search_results if r.get("url")]
                    # Obtener las URLs que el LLM decidió pasar
                    llm_urls = tool_args.get("urls_array", [])
                    if isinstance(llm_urls, str):
                        try:
                            llm_urls = json.loads(llm_urls)
                        except Exception:
                            llm_urls = []

                    # Normalizar para comparar
                    def _norm_url(u):
                        u = (u or "").strip()
                        if "://" in u:
                            s, r = u.split("://", 1)
                            u = s.lower() + "://" + r
                        return u.rstrip("/")

                    llm_set = {_norm_url(u) for u in llm_urls}
                    missing = [u for u in all_search_urls if _norm_url(u) not in llm_set]

                    if missing:
                        print(f"[take_action] LLM pasó {len(llm_urls)} URLs, search tenía {len(all_search_urls)}. Añadiendo {len(missing)} faltantes.")
                        if isinstance(tool_args.get("urls_array"), str):
                            # Si era string, reconstruir como lista completa
                            tool_args["urls_array"] = all_search_urls
                        else:
                            tool_args["urls_array"] = list(llm_urls) + missing
                    else:
                        print(f"[take_action] LLM pasó todas las {len(llm_urls)} URLs correctamente.")

            # ── FIX: Si el LLM llama a analyze_news_batch, forzar TODOS los items extraídos ──
            if tool_name == "analyze_news_batch":
                _tid = getattr(self, "_current_thread_id", "default")
                store = getattr(self, "_extract_store", {}).get(_tid, {})
                # Obtener todos los items extraídos por posición
                all_extracted = []
                for pos_key in sorted((k for k in store if k.isdigit()), key=int):
                    all_extracted.append(store[pos_key])

                if all_extracted:
                    try:
                        llm_items = json.loads(tool_args.get("items_json", "[]"))
                    except Exception:
                        llm_items = []

                    original_count = len(llm_items)

                    def _norm_url(u):
                        u = (u or "").strip()
                        if "://" in u:
                            s, r = u.split("://", 1)
                            u = s.lower() + "://" + r
                        return u.rstrip("/")

                    llm_urls_set = {_norm_url(it.get("url", "")) for it in llm_items}
                    next_id = max((it.get("id", 0) for it in llm_items), default=0) + 1

                    for ext_item in all_extracted:
                        ext_url = ext_item.get("url", "")
                        if _norm_url(ext_url) not in llm_urls_set and ext_url:
                            llm_items.append({
                                "id": next_id,
                                "url": ext_url,
                                "title": ext_item.get("title") or ext_url,
                            })
                            llm_urls_set.add(_norm_url(ext_url))
                            next_id += 1

                    if len(llm_items) > original_count:
                        print(f"[take_action] analyze: LLM tenía {original_count} items, completado a {len(llm_items)}")
                        tool_args["items_json"] = json.dumps(llm_items, ensure_ascii=False)

            if tool_name not in self.tools_by_name:
                results.append(
                    ToolMessage(
                        tool_call_id=tool_call["id"],
                        name=tool_name,
                        content=json.dumps(
                            {"error": f"Tool no encontrada: {tool_name}"},
                            ensure_ascii=False
                        ),
                    )
                )
                continue

            try:
                result = self.tools_by_name[tool_name].invoke(tool_args)
            except Exception as e:
                result = json.dumps(
                    {"error": f"Error ejecutando {tool_name}: {str(e)}"},
                    ensure_ascii=False
                )

            results.append(
                ToolMessage(
                    tool_call_id=tool_call["id"],
                    name=tool_name,
                    content=str(result),
                )
            )

        return {"messages": results}

    def _build_user_prompt(
        self,
        topic: str | None = None,
        url: str | None = None,
        message: str | None = None,
    ) -> str:
        if message:
            return (
                f"El usuario ha enviado este mensaje o consulta:\n"
                f"mensaje = {message.strip()}\n\n"
                f"Si este mensaje requiere buscar información nueva, Ejecuta OBLIGATORIAMENTE estos pasos en orden:\n"
                f"1. search_news para buscar información sobre el mensaje\n"
                f"2. extract_pages_batch con TODAS las URLs a la vez en una sola llamada. urls_array debe ser un JSON array, ejemplo: [\"https://url1\", \"https://url2\"]\n"
                f"3. analyze_news_batch con todos los items extraídos en una sola llamada\n"
                f"4. Devuelve el JSON final con los resultados.\n"
                f"Si es solo una charla o no necesita búsqueda, responde directamente en el campo assistant_message del JSON final."
            )

        if url:
            url_json = json.dumps([url])
            return (
                f"El usuario quiere procesar esta URL directamente:\n"
                f"url = {url}\n\n"
                f"Ejecuta OBLIGATORIAMENTE estos pasos en orden:\n"
                f"1. extract_pages_batch con urls_array={url_json}\n"
                f"2. analyze_news_batch pasando: id=1, title del extract, url=\"{url}\"\n"
                f"   (NO incluyas content ni raw_content, el sistema los inyecta)\n"
                f"3. Devuelve el JSON final con los resultados del análisis."
            )

        user_topic = (topic or "").strip() or "unknown"
        return (
            f"El usuario quiere investigar este topic:\n"
            f"topic = {user_topic}\n\n"
            f"Ejecuta OBLIGATORIAMENTE estos pasos en orden:\n"
            f"1. search_news para buscar noticias sobre el topic\n"
            f"2. extract_pages_batch con TODAS las URLs a la vez en una sola llamada. urls_array debe ser un JSON array, ejemplo: [\"https://url1\", \"https://url2\"]\n"
            f"3. analyze_news_batch con todos los items extraídos en una sola llamada\n"
            f"4. Devuelve el JSON final con los resultados."
        )

    def run(
        self,
        topic: str | None = None,
        url: str | None = None,
        message: str | None = None,
        thread_id: str | None = None,
    ):
        prompt = self._build_user_prompt(topic=topic, url=url, message=message)

        if not thread_id:
            thread_id = str(uuid.uuid4())

        config = {"configurable": {"thread_id": thread_id}}

        self._current_thread_id = thread_id  # Para que extract_pages_batch sepa en qué thread está
        result = self.graph.invoke(
            {"messages": [HumanMessage(content=prompt)]},
            config=config,
        )

        last_message = result["messages"][-1]
        raw_content = last_message.content

        # Gemini 2.5-flash devuelve content como lista de bloques
        if isinstance(raw_content, list):
            content = "\n".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in raw_content
            ).strip()
        else:
            content = str(raw_content).strip()

        # Quitar posibles code fences markdown (```json ... ```)
        if content.startswith("```"):
            lines = content.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            content = "\n".join(lines).strip()

        try:
            parsed = json.loads(content)
        except Exception:
            try:
                import json_repair
                parsed = json_repair.loads(content)
                if not isinstance(parsed, dict):
                    raise ValueError("Parsed content is not a dict")
            except Exception:
                parsed = {
                    "thread_id": thread_id,
                    "topic": topic or url or self.default_topic,
                    "results": [],
                    "error": None,
                    "assistant_message": content,
                }
                return parsed

        parsed["thread_id"] = thread_id
        parsed.setdefault("topic", topic or url or self.default_topic)
        parsed.setdefault("results", [])
        parsed.setdefault("error", None)
        parsed.setdefault("assistant_message", None)

        # Inyectar el raw_content completo almacenado en memoria al resultado final para el frontend
        store = getattr(self, "_extract_store", {}).get(thread_id, {})

        def _norm(u: str) -> str:
            """Normaliza URL para comparación: strip, lowercase scheme+host, quita trailing slash."""
            u = (u or "").strip()
            if "://" in u:
                scheme, rest = u.split("://", 1)
                u = scheme.lower() + "://" + rest
            return u.rstrip("/")

        # store tiene triple índice: URL original, normalizada, y posición (str)
        print(f"[inject] store tiene {len(store)} keys para thread {thread_id}")
        for idx, result_item in enumerate(parsed.get("results", []), 1):
            item_url = result_item.get("url", "")
            stored = (store.get(item_url)
                      or store.get(_norm(item_url))
                      or store.get(str(idx)))  # fallback por posición
            if stored:
                raw = stored.get("raw_content", "")
                if raw:
                    result_item["raw_content"] = raw
                    table_lines = [l for l in raw.split("\n") if l.strip().startswith("|") and l.strip().endswith("|")]
                    print(f"[inject] OK #{idx}: {item_url[:60]} ({len(raw)} chars, tables={len(table_lines)} lines)")
                else:
                    print(f"[inject] WARN raw vacío #{idx}: {item_url[:60]}")
            else:
                print(f"[inject] WARN no en store #{idx}: {item_url[:60]}")

        # ── FIX: completar resultados que el LLM omitió ──
        existing_urls = {_norm(r.get("url", "")) for r in parsed.get("results", [])}
        _analysis_items = getattr(self, "_analysis_store", {}).get(thread_id, {})

        print(f"[complete] LLM devolvió {len(parsed.get('results', []))} results")
        print(f"[complete] existing_urls: {existing_urls}")
        print(f"[complete] Store position keys: {sorted(k for k in store if k.isdigit())}")
        print(f"[complete] Analysis store tiene {len(_analysis_items)} keys")

        for pos_key in sorted((k for k in store if k.isdigit()), key=int):
            item = store[pos_key]
            item_url = item.get("url", "")
            norm_url = _norm(item_url)
            is_existing = norm_url in existing_urls
            print(f"[complete]   pos={pos_key} url={item_url[:60]} already_in_results={is_existing}")
            if is_existing:
                continue
            if not item_url:
                continue

            analysis_data = _analysis_items.get(norm_url) or _analysis_items.get(item_url) or {}
            new_result = {
                "id": len(parsed["results"]) + 1,
                "title": item.get("title") or item_url,
                "url": item_url,
                "content": (item.get("clean_content") or "")[:500],
                "raw_content": item.get("raw_content", ""),
                "analysis": analysis_data if analysis_data else {
                    "page_type": "article",
                    "important_points": [],
                    "summary": (item.get("clean_content") or "")[:300],
                    "key_facts": [],
                    "pricing_info": [],
                    "feature_comparison": [],
                    "tables_detected": [],
                    "missing_structured_data": [],
                    "error": None,
                },
            }
            parsed["results"].append(new_result)
            existing_urls.add(norm_url)
            print(f"[complete] ADDED missing result #{new_result['id']}: {item_url[:60]}")

        print(f"[complete] Final results count: {len(parsed.get('results', []))}")

        # Re-numerar ids secuencialmente
        for i, r in enumerate(parsed.get("results", []), 1):
            r["id"] = i

        # Fix: si el modelo devolvió JSON pero sin resultados ni mensaje útil,
        # generamos un assistant_message de fallback para que el frontend no quede vacío.
        if not parsed.get("results") and not parsed.get("assistant_message"):
            parsed["assistant_message"] = (
                "El agente completó el flujo pero no obtuvo resultados. "
                "Puede que la URL no sea accesible o que el contenido extraído esté vacío. "
                "Intenta con otra URL o reformula tu búsqueda."
            )

        return parsed