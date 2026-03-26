from fastapi import APIRouter
from app.schemas.agent import AgentRequest, AgentResponse
from app.services.agent_service import AgentService
import traceback

router = APIRouter(prefix="/agent", tags=["agent"])
agent_service = AgentService()


@router.get("/test")
def test_agent():
    return {"message": "Agent funcionando"}


def _parse_input(request: AgentRequest) -> dict:
    """Si solo llega message, detecta si es URL y enruta. El texto libre
    siempre se mantiene como message para preservar el contexto conversacional."""
    topic = request.topic
    url = request.url
    message = request.message

    if message and not topic and not url:
        text = message.strip()
        if text.startswith("http://") or text.startswith("https://"):
            # Es una URL: enrutar a url
            url = text
            message = None
        # Si es texto libre, lo dejamos como message siempre.
        # Así el agente puede usar el historial del thread para responder.

    return {"topic": topic, "url": url, "message": message}


@router.post("/run", response_model=AgentResponse)
def run_agent(request: AgentRequest):
    parsed = _parse_input(request)
    try:
        result = agent_service.run(
            topic=parsed["topic"],
            url=parsed["url"],
            message=parsed["message"],
            thread_id=request.thread_id,
        )
        print(f"[run_agent] Resultado OK, keys: {list(result.keys())}")
        return result
    except Exception as e:
        print(f"[run_agent] ERROR: {e}")
        traceback.print_exc()
        raise