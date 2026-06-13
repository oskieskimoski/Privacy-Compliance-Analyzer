import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    RAW_DATA_DIR: Path = BASE_DIR / "data" / "raw"

    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"


settings = Settings()

settings.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
