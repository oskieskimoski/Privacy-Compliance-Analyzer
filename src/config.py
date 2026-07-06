import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    RAW_DATA_DIR: Path = BASE_DIR / "data" / "raw"

    CHUNK_SIZE: int = 1500
    CHUNK_OVERLAP: int = 300
    VECTOR_DB_DIR: Path = BASE_DIR / "data" / "vector_db"
    EMBEDDING_MODEL_NAME: str = "intfloat/multilingual-e5-small"


settings = Settings()

settings.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
