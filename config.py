from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass


def require_non_empty(value: str, name: str) -> str:
    if not value or not value.strip():
        raise ValueError(f"Missing required config: {name}. Set it in environment or in a .env file.")
    return value


@dataclass(frozen=True)
class OpenAIConfig:
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    temperature: float = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
    max_tokens: int = int(os.getenv("OPENAI_MAX_TOKENS", "800"))


@dataclass(frozen=True)
class MySQLConfig:
    username: str = os.getenv("MYSQL_USERNAME", "root")
    password: str = os.getenv("MYSQL_PASSWORD", "")
    host: str = os.getenv("MYSQL_HOST", "127.0.0.1")
    port: int = int(os.getenv("MYSQL_PORT", "3306"))
    database: str = os.getenv("MYSQL_DATABASE", "Chinook")

    def sqlalchemy_uri(self) -> str:
        return f"mysql+mysqlconnector://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
