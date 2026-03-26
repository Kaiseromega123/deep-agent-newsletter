from app.agents.deep_agent import DeepAgent


class AgentService:
    def __init__(self):
        self.agent = DeepAgent()

    def run(
        self,
        topic: str | None = None,
        url: str | None = None,
        message: str | None = None,
        thread_id: str | None = None,
    ) -> dict:
        return self.agent.run(
            topic=topic, url=url, message=message, thread_id=thread_id
        )