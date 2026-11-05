"""The only module that talks to LLMs. Every call is cached on disk and recorded in a trace."""

import hashlib
import json
import re
import time
from collections.abc import Callable
from pathlib import Path

import litellm
from pydantic import TypeAdapter, ValidationError

from .config import DATA, settings

litellm.suppress_debug_info = True

CACHE = DATA / "cache"
# Free tiers fail transiently: 429 quotas, 503 "high demand", hangs. Other 4xx are real errors.
RETRYABLE = (
    litellm.RateLimitError,
    litellm.ServiceUnavailableError,
    litellm.InternalServerError,
    litellm.APIConnectionError,  # includes litellm.Timeout
)
DELAYS = (2, 4, 8, 16, 32, 64)


class EmptyResponse(Exception):
    """Gemini 3.x can spend its whole budget thinking and return no text."""


def _cache_path(agent: str, messages: list[dict]) -> Path:
    a = settings().agents[agent]
    key = json.dumps([a.model, a.temperature, messages], sort_keys=True)
    return CACHE / f"{hashlib.sha256(key.encode()).hexdigest()}.txt"


def _call(agent: str, messages: list[dict]) -> str:
    a = settings().agents[agent]
    for delay in (*DELAYS, None):
        try:
            r = litellm.completion(model=a.model, temperature=a.temperature, messages=messages, timeout=300)
            text = r.choices[0].message.content
            if text and text.strip():
                return text
            err: Exception = EmptyResponse(f"{a.model} returned no text")
        except RETRYABLE as e:
            if "PerDay" in str(e) or "per day" in str(e).lower():  # a daily quota will not clear in 2 minutes
                raise
            err = e
        if delay is None:
            raise err
        time.sleep(delay)
    raise AssertionError("unreachable")


def complete(agent: str, messages: list[dict], trace: list[dict]) -> str:
    """Text completion for `agent` (a key under [agents] in config.toml)."""
    path = _cache_path(agent, messages)
    cached = path.exists()
    if cached:
        text = path.read_text()
    else:
        text = _call(agent, messages)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    trace.append({"agent": agent, "model": settings().agents[agent].model, "messages": messages,
                  "response": text, "cached": cached})
    return text


def extract_json[T](text: str, adapter: TypeAdapter[T]) -> T:
    """First JSON value in `text` that fits `adapter`, ignoring code fences and prose.

    Models often quote other JSON (an id list, say) in their reasoning before the real answer.
    """
    decoder = json.JSONDecoder(strict=False)  # models put raw newlines in strings
    broken: ValueError | None = None
    invalid: tuple[int, ValidationError] | None = None
    for n, m in enumerate(re.finditer(r"[\[{]", text)):
        start = m.start()
        try:
            value, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError as e:
            if broken is None and e.pos - start > 20:  # a real JSON attempt, not a bracket in prose
                broken = ValueError(f"invalid JSON ({e.msg}) near {text[max(start, e.pos - 80):e.pos + 20]!r}; "
                                    "escape or avoid double quotes inside string values")
            continue
        # one object where a one-item list was asked for, but only if it is the reply's first JSON value
        for candidate in ([value, [value]] if n == 0 and isinstance(value, dict) else [value]):
            try:
                return adapter.validate_python(candidate)
            except ValidationError as e:  # keep the biggest candidate's error: that one is the intended answer
                if invalid is None or end - start > invalid[0]:
                    invalid = (end - start, e)
    raise broken or (invalid[1] if invalid else ValueError("no JSON value found in the reply"))


def complete_json[T](agent: str, messages: list[dict], schema: type[T], trace: list[dict],
                     validate: Callable[[T], None] | None = None) -> T:
    """Completion parsed into `schema` (a model or e.g. list[Model]); one repair retry on bad output.

    `validate` may raise ValueError for checks the schema cannot express.
    """
    adapter = TypeAdapter(schema)
    msgs = list(messages)
    for attempt in range(2):
        text = complete(agent, msgs, trace)
        try:
            value = extract_json(text, adapter)
            if validate:
                validate(value)
            return value
        except ValueError as e:  # includes JSONDecodeError and pydantic ValidationError
            _cache_path(agent, msgs).unlink(missing_ok=True)  # never replay a bad answer on re-run
            if attempt:
                raise
            msgs += [{"role": "assistant", "content": text},
                     {"role": "user", "content": f"Your reply was invalid: {e}\nReturn only the corrected JSON."}]
    raise AssertionError("unreachable")
