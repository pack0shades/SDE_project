"""Typed view of config.toml, the single place for models and knobs."""

import tomllib
from functools import cache
from pathlib import Path

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


class Agent(BaseModel):
    model: str
    temperature: float


class Regulation(BaseModel):
    name: str
    file: Path  # relative to ROOT
    focus: str
    hints: list[str] = []


class Embedding(BaseModel):
    model: str
    query_prefix: str


class Rerank(BaseModel):
    model: str


class Retrieval(BaseModel):
    k: int
    mmr_keep: int
    mmr_lambda: float
    top_n: int
    max_chunk_tokens: int


class Pipeline(BaseModel):
    batch_size: int
    max_rounds: int
    web_search: bool


class Settings(BaseModel):
    embedding: Embedding
    rerank: Rerank
    retrieval: Retrieval
    pipeline: Pipeline
    agents: dict[str, Agent]
    regulations: dict[str, Regulation]


@cache
def settings() -> Settings:
    return Settings.model_validate(tomllib.loads((ROOT / "config.toml").read_text()))
