import os
from dotenv import load_dotenv, find_dotenv

# Encuentra el .env aunque esté en la carpeta superior
load_dotenv(find_dotenv(usecwd=True), override=True)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_ANALYZER_MODEL = "gemini-2.0-flash"