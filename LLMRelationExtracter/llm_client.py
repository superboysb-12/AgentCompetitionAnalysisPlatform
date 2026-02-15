"""
Shared LangChain LLM caller utilities for relation extraction.

This module centralizes:
1) Process-wide LLM concurrency control for outbound requests.
2) Unified retry/backoff behavior for network and parse failures.
3) Strict JSON-schema response handling for ChatOpenAI calls.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

_GLOBAL_LLM_CONCURRENCY_LOCK = threading.Lock()
_GLOBAL_LLM_CONCURRENCY_LIMIT: Optional[int] = None
_GLOBAL_LLM_CONCURRENCY_SEMAPHORE: Optional[threading.BoundedSemaphore] = None
_GLOBAL_LLM_CALL_COUNT_LOCK = threading.Lock()
_GLOBAL_LLM_CALL_COUNT = 0
_GLOBAL_CHAT_OPENAI_LOCK = threading.Lock()
_GLOBAL_CHAT_OPENAI_INSTANCES: Dict[tuple, Any] = {}
_GLOBAL_CHAT_OPENAI_IDS: set[int] = set()


def configure_global_llm_concurrency(max_concurrent_calls: int) -> int:
    """
    Configure process-wide global LLM call concurrency.
    Keep the first initialized limit for process consistency.
    """
    global _GLOBAL_LLM_CONCURRENCY_LIMIT, _GLOBAL_LLM_CONCURRENCY_SEMAPHORE
    safe_limit = max(1, int(max_concurrent_calls))
    with _GLOBAL_LLM_CONCURRENCY_LOCK:
        if _GLOBAL_LLM_CONCURRENCY_LIMIT is None:
            _GLOBAL_LLM_CONCURRENCY_LIMIT = safe_limit
            _GLOBAL_LLM_CONCURRENCY_SEMAPHORE = threading.BoundedSemaphore(safe_limit)
            return safe_limit
        return _GLOBAL_LLM_CONCURRENCY_LIMIT


def get_global_llm_concurrency() -> Optional[int]:
    with _GLOBAL_LLM_CONCURRENCY_LOCK:
        return _GLOBAL_LLM_CONCURRENCY_LIMIT


def _get_global_llm_semaphore(max_concurrent_calls: int) -> threading.BoundedSemaphore:
    global _GLOBAL_LLM_CONCURRENCY_SEMAPHORE
    configure_global_llm_concurrency(max_concurrent_calls)
    with _GLOBAL_LLM_CONCURRENCY_LOCK:
        if _GLOBAL_LLM_CONCURRENCY_SEMAPHORE is None:
            # Defensive fallback; should already be set by configure call above.
            _GLOBAL_LLM_CONCURRENCY_SEMAPHORE = threading.BoundedSemaphore(
                max(1, int(max_concurrent_calls))
            )
        return _GLOBAL_LLM_CONCURRENCY_SEMAPHORE


def mark_global_llm_call_completed(schema_name: str) -> int:
    global _GLOBAL_LLM_CALL_COUNT
    with _GLOBAL_LLM_CALL_COUNT_LOCK:
        _GLOBAL_LLM_CALL_COUNT += 1
        current = _GLOBAL_LLM_CALL_COUNT
    print(f"[llm-call] completed={current} schema={schema_name}", flush=True)
    return current


def get_shared_chat_openai(
    *,
    api_key: str,
    base_url: str,
    model: str,
    temperature: float = 0.0,
    timeout: float = 300.0,
) -> Any:
    """
    Return a process-wide singleton ChatOpenAI instance for the same config.
    """
    key = (
        str(api_key or ""),
        str(base_url or ""),
        str(model or ""),
        float(temperature),
        float(timeout),
    )
    with _GLOBAL_CHAT_OPENAI_LOCK:
        llm = _GLOBAL_CHAT_OPENAI_INSTANCES.get(key)
        if llm is None:
            llm = ChatOpenAI(
                api_key=api_key,
                base_url=base_url,
                model=model,
                temperature=temperature,
                timeout=timeout,
            )
            _GLOBAL_CHAT_OPENAI_INSTANCES[key] = llm
            _GLOBAL_CHAT_OPENAI_IDS.add(id(llm))
        return llm


def is_shared_chat_openai_instance(llm: Any) -> bool:
    """
    Check whether the given llm object is managed by singleton factory.
    """
    with _GLOBAL_CHAT_OPENAI_LOCK:
        return id(llm) in _GLOBAL_CHAT_OPENAI_IDS


@asynccontextmanager
async def global_llm_concurrency_guard(max_concurrent_calls: int):
    """
    Process-wide LLM concurrency guard.
    Every request (including retries) must pass this guard.
    """
    sem = _get_global_llm_semaphore(max_concurrent_calls)
    await asyncio.to_thread(sem.acquire)
    try:
        yield
    finally:
        sem.release()


def extract_error_code(exc: Exception) -> Optional[str]:
    """
    Best-effort extraction of error code/status from provider/network exceptions.
    """
    if exc is None:
        return None

    for attr in ("status_code", "code", "errno"):
        value = getattr(exc, attr, None)
        if value not in (None, ""):
            return str(value)

    response = getattr(exc, "response", None)
    if response is not None:
        status_code = getattr(response, "status_code", None)
        if status_code not in (None, ""):
            return str(status_code)

        data = None
        try:
            json_method = getattr(response, "json", None)
            if callable(json_method):
                data = json_method()
        except Exception:  # noqa: BLE001
            data = None

        if isinstance(data, dict):
            err = data.get("error")
            if isinstance(err, dict):
                for key in ("code", "type", "status"):
                    value = err.get(key)
                    if value not in (None, ""):
                        return str(value)
            for key in ("code", "status"):
                value = data.get(key)
                if value not in (None, ""):
                    return str(value)

    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            for key in ("code", "type", "status"):
                value = err.get(key)
                if value not in (None, ""):
                    return str(value)
        for key in ("code", "status"):
            value = body.get(key)
            if value not in (None, ""):
                return str(value)

    return None


class LangChainJsonLLMClient:
    """
    Unified wrapper for ChatOpenAI JSON-schema calls.

    Supports:
    - network-call retries
    - parse retries
    - exponential backoff
    - optional process-wide concurrency gate
    """

    def __init__(
        self,
        llm: Any,
        *,
        logger: Optional[logging.Logger] = None,
        max_retries: int = 8,
        retry_delay: float = 2.0,
        retry_backoff_factor: float = 2.5,
        retry_max_delay: float = 120.0,
        hard_timeout: float = 180.0,
        llm_global_concurrency: Optional[int] = None,
        print_call_counter: bool = False,
        slow_call_threshold: float = 30.0,
        recycle_on_connection_error: bool = True,
        recycle_after_calls: int = 0,
        recycle_min_interval: float = 5.0,
    ) -> None:
        self.llm = llm
        self.logger = logger
        self.max_retries = max(0, int(max_retries))
        self.retry_delay = max(0.1, float(retry_delay))
        self.retry_backoff_factor = max(1.0, float(retry_backoff_factor))
        self.retry_max_delay = max(self.retry_delay, float(retry_max_delay))
        self.hard_timeout = float(hard_timeout)
        self.print_call_counter = bool(print_call_counter)
        self.slow_call_threshold = max(0.0, float(slow_call_threshold))
        self.recycle_on_connection_error = bool(recycle_on_connection_error)
        self.recycle_after_calls = max(0, int(recycle_after_calls))
        self.recycle_min_interval = max(0.0, float(recycle_min_interval))
        self.llm_global_concurrency = (
            configure_global_llm_concurrency(llm_global_concurrency)
            if llm_global_concurrency is not None
            else None
        )
        self._local_call_count_lock = threading.Lock()
        self._local_call_count = 0
        self._last_recycle_ts = 0.0
        self._recycle_lock: Optional[asyncio.Lock] = None

    async def call_json(
        self,
        messages: Sequence[Dict[str, str]],
        schema: Dict,
        schema_name: str,
    ) -> Dict:
        lc_messages = self._to_langchain_messages(messages)

        parse_attempt = 0
        while True:
            response = await self._ainvoke_with_retry(
                lc_messages,
                schema_name,
                schema,
            )
            try:
                content = self._response_to_text(response)
                if not content:
                    raise ValueError(f"Empty response for {schema_name}")
                return json.loads(content)
            except Exception as exc:  # noqa: BLE001
                if parse_attempt >= self.max_retries:
                    if isinstance(exc, ValueError):
                        preview = content[:200] if "content" in locals() else ""
                        raise ValueError(
                            f"Invalid JSON for {schema_name}: {preview}"
                        ) from exc
                    raise
                delay = self._compute_retry_delay(parse_attempt)
                if self.logger is not None:
                    self.logger.warning(
                        "LLM response parse failed for %s, retry %s/%s in %.2fs: %s",
                        schema_name,
                        parse_attempt + 1,
                        self.max_retries,
                        delay,
                        exc,
                    )
                error_code = extract_error_code(exc)
                print(
                    f"[llm-retry] schema={schema_name} parse_retry={parse_attempt + 1}/{self.max_retries} "
                    f"delay={delay:.2f}s code={error_code or 'N/A'} "
                    f"err={exc.__class__.__name__}: {str(exc)[:180]}",
                    flush=True,
                )
                parse_attempt += 1
                await asyncio.sleep(delay)

    def _bind_json_schema_llm(self, schema: Dict, schema_name: str) -> Any:
        return self.llm.bind(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                },
            }
        )

    def _to_langchain_messages(self, messages: Sequence[Dict[str, str]]) -> List[Any]:
        lc_messages: List[Any] = []
        for msg in messages or []:
            role = str((msg or {}).get("role") or "").strip().lower()
            content = str((msg or {}).get("content") or "")
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))
        return lc_messages

    def _response_to_text(self, response: Any) -> str:
        raw_content = getattr(response, "content", "")
        if isinstance(raw_content, str):
            return raw_content.strip()
        if isinstance(raw_content, list):
            text_parts: List[str] = []
            for part in raw_content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str):
                        text_parts.append(text)
                elif isinstance(part, str):
                    text_parts.append(part)
            return "".join(text_parts).strip()
        return str(raw_content or "").strip()

    def _compute_retry_delay(self, attempt: int) -> float:
        return min(
            self.retry_max_delay,
            self.retry_delay * (self.retry_backoff_factor ** max(0, attempt)),
        )

    async def _ainvoke_with_retry(
        self,
        lc_messages: List[Any],
        schema_name: str,
        schema: Dict,
    ) -> Any:
        attempt = 0
        while True:
            try:
                llm = self._bind_json_schema_llm(schema, schema_name)
                call_started = time.monotonic()
                if self.llm_global_concurrency is not None:
                    async with global_llm_concurrency_guard(self.llm_global_concurrency):
                        if self.hard_timeout > 0:
                            response = await asyncio.wait_for(
                                llm.ainvoke(lc_messages),
                                timeout=self.hard_timeout,
                            )
                        else:
                            response = await llm.ainvoke(lc_messages)
                else:
                    if self.hard_timeout > 0:
                        response = await asyncio.wait_for(
                            llm.ainvoke(lc_messages),
                            timeout=self.hard_timeout,
                        )
                    else:
                        response = await llm.ainvoke(lc_messages)
                if self.print_call_counter:
                    mark_global_llm_call_completed(schema_name)
                call_elapsed = time.monotonic() - call_started
                if (
                    self.slow_call_threshold > 0
                    and call_elapsed >= self.slow_call_threshold
                ):
                    print(
                        f"[llm-call] slow schema={schema_name} elapsed={call_elapsed:.1f}s",
                        flush=True,
                    )
                if self.recycle_after_calls > 0:
                    should_recycle = False
                    with self._local_call_count_lock:
                        self._local_call_count += 1
                        if self._local_call_count % self.recycle_after_calls == 0:
                            should_recycle = True
                    if should_recycle:
                        await self._recycle_socket_transport(
                            f"periodic_call_count={self._local_call_count}"
                        )
                return response
            except Exception as exc:  # noqa: BLE001
                if attempt >= self.max_retries:
                    raise
                delay = self._compute_retry_delay(attempt)
                if self.recycle_on_connection_error and self._looks_like_connection_error(exc):
                    await self._recycle_socket_transport(
                        f"connection_error:{exc.__class__.__name__}"
                    )
                if self.logger is not None:
                    self.logger.warning(
                        "LLM call failed for %s, retry %s/%s in %.2fs: %s",
                        schema_name,
                        attempt + 1,
                        self.max_retries,
                        delay,
                        exc,
                    )
                error_code = extract_error_code(exc)
                print(
                    f"[llm-retry] schema={schema_name} retry={attempt + 1}/{self.max_retries} "
                    f"delay={delay:.2f}s code={error_code or 'N/A'} "
                    f"err={exc.__class__.__name__}: {str(exc)[:180]}",
                    flush=True,
                )
                attempt += 1
                await asyncio.sleep(delay)

    def _looks_like_connection_error(self, exc: Exception) -> bool:
        name = exc.__class__.__name__.lower()
        text = str(exc or "").lower()
        name_tokens = (
            "apiconnectionerror",
            "connecterror",
            "connectionerror",
            "readerror",
            "remoteprotocolerror",
        )
        text_tokens = (
            "connection error",
            "connection reset",
            "server disconnected",
            "broken pipe",
            "connection aborted",
            "network is unreachable",
            "name or service not known",
            "temporary failure in name resolution",
            "tlsv1",
            "ssl",
        )
        if any(token in name for token in name_tokens):
            return True
        return any(token in text for token in text_tokens)

    def _get_recycle_lock(self) -> asyncio.Lock:
        if self._recycle_lock is None:
            self._recycle_lock = asyncio.Lock()
        return self._recycle_lock

    async def _recycle_socket_transport(self, reason: str) -> None:
        now = time.monotonic()
        if now - self._last_recycle_ts < self.recycle_min_interval:
            return
        lock = self._get_recycle_lock()
        async with lock:
            now2 = time.monotonic()
            if now2 - self._last_recycle_ts < self.recycle_min_interval:
                return
            await self._close_llm_clients()
            self._last_recycle_ts = time.monotonic()
            if self.logger is not None:
                self.logger.info("LLM socket recycle done: reason=%s", reason)

    async def _close_llm_clients(self) -> None:
        llm = self.llm
        root_async_client = getattr(llm, "root_async_client", None)
        if root_async_client is not None:
            close_fn = getattr(root_async_client, "close", None)
            if callable(close_fn):
                try:
                    maybe_coro = close_fn()
                    if asyncio.iscoroutine(maybe_coro):
                        await maybe_coro
                except Exception as exc:  # noqa: BLE001
                    if self.logger is not None:
                        self.logger.debug(
                            "Ignore async client close error during recycle: %s",
                            exc,
                        )

        root_client = getattr(llm, "root_client", None)
        if root_client is not None:
            close_fn = getattr(root_client, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception as exc:  # noqa: BLE001
                    if self.logger is not None:
                        self.logger.debug(
                            "Ignore sync client close error during recycle: %s",
                            exc,
                        )
