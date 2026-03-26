from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.agent import router as agent_router

app = FastAPI(title="Deep Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(agent_router)


@app.get("/")
def home():
    return {"message": "API funcionando"}