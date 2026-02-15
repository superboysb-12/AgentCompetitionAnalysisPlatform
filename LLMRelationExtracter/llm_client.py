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
import weakref
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
from urllib.parse import urlparse

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

try:
    import httpx
except Exception:  # noqa: BLE001
    httpx = None

_GLOBAL_LLM_CONCURRENCY_LOCK = threading.Lock()
_GLOBAL_LLM_CONCURRENCY_LIMIT: Optional[int] = None
_GLOBAL_LLM_LOOP_SEMAPHORES: "weakref.WeakKeyDictionary[Any, asyncio.Semaphore]" = (
    weakref.WeakKeyDictionary()
)
_GLOBAL_LLM_RPM_LOCK = threading.Lock()
_GLOBAL_LLM_RPM_LIMIT: Optional[float] = None
_GLOBAL_LLM_LOOP_RPM_LIMITERS: "weakref.WeakKeyDictionary[Any, _LoopRpmLimiter]"  # type: ignore[name-defined]
_GLOBAL_LLM_CALL_COUNT_LOCK = threading.Lock()
_GLOBAL_LLM_CALL_COUNT = 0
_GLOBAL_CHAT_OPENAI_LOCK = threading.Lock()
_GLOBAL_CHAT_OPENAI_INSTANCES: Dict[tuple, Any] = {}
_GLOBAL_CHAT_OPENAI_IDS: set[int] = set()
_GLOBAL_CHAT_OPENAI_KEY_BY_ID: Dict[int, tuple] = {}
_GLOBAL_JSON_LLM_CLIENT_LOCK = threading.Lock()
_GLOBAL_JSON_LLM_CLIENTS: Dict[tuple, "LangChainJsonLLMClient"] = {}
_GLOBAL_HTTPX_WARNING_PRINTED = False
_GLOBAL_RETRY_LOG_LOCK = threading.Lock()


class _LoopRpmLimiter:
    """
    Per-event-loop paced dispatcher.
    It serializes requests and distributes starts evenly by fixed interval.
    When upstream has no pending callers, pacing is paused until demand resumes.
    """

    def __init__(self, rpm: float) -> None:
        safe_rpm = max(1.0, float(rpm))
        self.interval = 60.0 / safe_rpm
        self._lock = asyncio.Lock()
        self._next_allowed = 0.0

    async def acquire_slot(self) -> None:
        await self._lock.acquire()
        try:
            now = time.monotonic()
            wait_seconds = self._next_allowed - now
            if wait_seconds > 0:
                await asyncio.sleep(wait_seconds)
            current = time.monotonic()
            base = max(current, self._next_allowed)
            self._next_allowed = base + self.interval
        except BaseException:
            self._lock.release()
            raise

    def release_slot(self) -> None:
        if self._lock.locked():
            self._lock.release()


_GLOBAL_LLM_LOOP_RPM_LIMITERS = weakref.WeakKeyDictionary()


def configure_global_llm_concurrency(max_concurrent_calls: int) -> int:
    """
    Configure process-wide global LLM call concurrency.
    Keep the first initialized limit for process consistency.
    """
    global _GLOBAL_LLM_CONCURRENCY_LIMIT
    safe_limit = max(1, int(max_concurrent_calls))
    with _GLOBAL_LLM_CONCURRENCY_LOCK:
        if _GLOBAL_LLM_CONCURRENCY_LIMIT is None:
            _GLOBAL_LLM_CONCURRENCY_LIMIT = safe_limit
            return safe_limit
        return _GLOBAL_LLM_CONCURRENCY_LIMIT


def get_global_llm_concurrency() -> Optional[int]:
    with _GLOBAL_LLM_CONCURRENCY_LOCK:
        return _GLOBAL_LLM_CONCURRENCY_LIMIT


def _get_global_llm_semaphore(max_concurrent_calls: int) -> asyncio.Semaphore:
    effective_limit = configure_global_llm_concurrency(max_concurrent_calls)
    loop = asyncio.get_running_loop()
    with _GLOBAL_LLM_CONCURRENCY_LOCK:
        sem = _GLOBAL_LLM_LOOP_SEMAPHORES.get(loop)
        if sem is None:
            sem = asyncio.Semaphore(effective_limit)
            _GLOBAL_LLM_LOOP_SEMAPHORES[loop] = sem
        return sem


def configure_global_llm_rpm(max_rpm: float) -> float:
    """
    Configure process-wide global LLM RPM pacing.
    Keep the first initialized limit for process consistency.
    """
    global _GLOBAL_LLM_RPM_LIMIT
    safe_rpm = max(1.0, float(max_rpm))
    with _GLOBAL_LLM_RPM_LOCK:
        if _GLOBAL_LLM_RPM_LIMIT is None:
            _GLOBAL_LLM_RPM_LIMIT = safe_rpm
            return safe_rpm
        return _GLOBAL_LLM_RPM_LIMIT


def get_global_llm_rpm() -> Optional[float]:
    with _GLOBAL_LLM_RPM_LOCK:
        return _GLOBAL_LLM_RPM_LIMIT


def _get_global_llm_rpm_limiter(max_rpm: float) -> _LoopRpmLimiter:
    effective_rpm = configure_global_llm_rpm(max_rpm)
    loop = asyncio.get_running_loop()
    with _GLOBAL_LLM_RPM_LOCK:
        limiter = _GLOBAL_LLM_LOOP_RPM_LIMITERS.get(loop)
        if limiter is None:
            limiter = _LoopRpmLimiter(effective_rpm)
            _GLOBAL_LLM_LOOP_RPM_LIMITERS[loop] = limiter
        return limiter


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
    http_max_connections: Optional[int] = None,
    http_max_keepalive_connections: Optional[int] = None,
    http_keepalive_expiry: Optional[float] = None,
) -> Any:
    """
    Return a process-wide singleton ChatOpenAI instance for the same config.
    """
    max_connections = _normalize_pool_int(http_max_connections, default=200)
    max_keepalive_connections = _normalize_pool_int(
        http_max_keepalive_connections,
        default=min(max_connections, 100),
    )
    keepalive_expiry = _normalize_pool_float(http_keepalive_expiry, default=30.0)
    key = (
        str(api_key or ""),
        str(base_url or ""),
        str(model or ""),
        float(temperature),
        float(timeout),
        int(max_connections),
        int(max_keepalive_connections),
        float(keepalive_expiry),
    )
    with _GLOBAL_CHAT_OPENAI_LOCK:
        llm = _GLOBAL_CHAT_OPENAI_INSTANCES.get(key)
        created = False
        if llm is None:
            llm = _build_chat_openai_from_key(key)
            _GLOBAL_CHAT_OPENAI_INSTANCES[key] = llm
            _GLOBAL_CHAT_OPENAI_IDS.add(id(llm))
            created = True
        _GLOBAL_CHAT_OPENAI_KEY_BY_ID[id(llm)] = key
        action = "create" if created else "reuse"
        print(
            "[llm-client] shared-chat-openai "
            f"action={action} id={id(llm)} model={model} pool={max_connections}/{max_keepalive_connections}/{keepalive_expiry}",
            flush=True,
        )
        return llm


def is_shared_chat_openai_instance(llm: Any) -> bool:
    """
    Check whether the given llm object is managed by singleton factory.
    """
    with _GLOBAL_CHAT_OPENAI_LOCK:
        return id(llm) in _GLOBAL_CHAT_OPENAI_IDS


async def _close_chat_openai_clients(
    llm: Any,
    *,
    logger: Optional[logging.Logger] = None,
) -> None:
    root_async_client = getattr(llm, "root_async_client", None)
    if root_async_client is not None:
        close_fn = getattr(root_async_client, "close", None)
        if callable(close_fn):
            try:
                maybe_coro = close_fn()
                if asyncio.iscoroutine(maybe_coro):
                    await maybe_coro
            except Exception as exc:  # noqa: BLE001
                if logger is not None:
                    logger.debug(
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
                if logger is not None:
                    logger.debug(
                        "Ignore sync client close error during recycle: %s",
                        exc,
                    )


def _schedule_close_chat_openai_clients(
    llm: Any,
    *,
    logger: Optional[logging.Logger] = None,
    delay_seconds: float = 120.0,
) -> None:
    safe_delay = max(0.0, float(delay_seconds))

    def _runner() -> None:
        try:
            asyncio.run(_close_chat_openai_clients(llm, logger=logger))
        except Exception as exc:  # noqa: BLE001
            if logger is not None:
                logger.debug("Ignore deferred client close error: %s", exc)

    timer = threading.Timer(safe_delay, _runner)
    timer.daemon = True
    timer.start()


def _normalize_pool_int(value: Optional[int], *, default: int) -> int:
    if value is None:
        return max(1, int(default))
    return max(1, int(value))


def _normalize_pool_float(value: Optional[float], *, default: float) -> float:
    if value is None:
        return max(1.0, float(default))
    return max(1.0, float(value))


def _build_httpx_clients(
    *,
    max_connections: int,
    max_keepalive_connections: int,
    keepalive_expiry: float,
    timeout: float,
) -> Dict[str, Any]:
    global _GLOBAL_HTTPX_WARNING_PRINTED
    if httpx is None:
        if not _GLOBAL_HTTPX_WARNING_PRINTED:
            print(
                "[llm-client] httpx unavailable; fallback to OpenAI default transport",
                flush=True,
            )
            _GLOBAL_HTTPX_WARNING_PRINTED = True
        return {}
    limits = httpx.Limits(
        max_connections=max_connections,
        max_keepalive_connections=max_keepalive_connections,
        keepalive_expiry=keepalive_expiry,
    )
    timeout_cfg = httpx.Timeout(timeout)
    return {
        "http_client": httpx.Client(limits=limits, timeout=timeout_cfg),
        "http_async_client": httpx.AsyncClient(limits=limits, timeout=timeout_cfg),
    }


def _build_chat_openai_from_key(key: tuple) -> Any:
    (
        api_key,
        base_url,
        model,
        temperature,
        timeout,
        max_connections,
        max_keepalive_connections,
        keepalive_expiry,
    ) = key
    transport_kwargs = _build_httpx_clients(
        max_connections=int(max_connections),
        max_keepalive_connections=int(max_keepalive_connections),
        keepalive_expiry=float(keepalive_expiry),
        timeout=float(timeout),
    )
    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
        timeout=timeout,
        **transport_kwargs,
    )


async def reset_openai_client(
    llm: Any,
    *,
    reason: str,
    logger: Optional[logging.Logger] = None,
    defer_close_seconds: float = 120.0,
) -> Any:
    """
    Reset the underlying OpenAI clients.
    If the llm comes from shared factory, replace the global singleton instance.
    """
    replacement = llm
    old_shared_llm: Optional[Any] = None
    shared_key = None

    with _GLOBAL_CHAT_OPENAI_LOCK:
        shared_key = _GLOBAL_CHAT_OPENAI_KEY_BY_ID.get(id(llm))
        if shared_key is not None and _GLOBAL_CHAT_OPENAI_INSTANCES.get(shared_key) is llm:
            replacement = _build_chat_openai_from_key(shared_key)
            _GLOBAL_CHAT_OPENAI_INSTANCES[shared_key] = replacement
            _GLOBAL_CHAT_OPENAI_IDS.discard(id(llm))
            _GLOBAL_CHAT_OPENAI_KEY_BY_ID.pop(id(llm), None)
            _GLOBAL_CHAT_OPENAI_IDS.add(id(replacement))
            _GLOBAL_CHAT_OPENAI_KEY_BY_ID[id(replacement)] = shared_key
            old_shared_llm = llm

    print(
        f"[llm-client] openai-client-reset reason={reason} shared={old_shared_llm is not None}",
        flush=True,
    )

    if old_shared_llm is not None:
        safe_defer = max(0.0, float(defer_close_seconds))
        if safe_defer > 0:
            _schedule_close_chat_openai_clients(
                old_shared_llm,
                logger=logger,
                delay_seconds=safe_defer,
            )
            print(
                f"[llm-client] openai-client-close deferred={safe_defer:.1f}s",
                flush=True,
            )
        else:
            await _close_chat_openai_clients(old_shared_llm, logger=logger)
        return replacement

    # Non-shared instance fallback: close and clear in-place handles.
    await _close_chat_openai_clients(llm, logger=logger)
    for attr in ("client", "async_client", "root_client", "root_async_client"):
        if hasattr(llm, attr):
            try:
                setattr(llm, attr, None)
            except Exception:  # noqa: BLE001
                continue
    return llm


@asynccontextmanager
async def global_llm_concurrency_guard(max_concurrent_calls: int):
    """
    Process-wide LLM concurrency guard.
    Every request (including retries) must pass this guard.
    """
    sem = _get_global_llm_semaphore(max_concurrent_calls)
    await sem.acquire()
    try:
        yield
    finally:
        sem.release()


@asynccontextmanager
async def global_llm_rpm_guard(max_rpm: float):
    """
    Process-wide RPM pacing guard.
    Starts are evenly distributed at interval = 60 / rpm.
    This guard is serial (one in-flight request globally), and pauses when idle.
    """
    limiter = _get_global_llm_rpm_limiter(max_rpm)
    await limiter.acquire_slot()
    try:
        yield
    finally:
        limiter.release_slot()


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


def extract_error_diagnostics(exc: Exception, *, max_depth: int = 4) -> Dict[str, Any]:
    """
    Best-effort diagnostics for network/provider exceptions.
    Includes chained causes and request endpoint metadata.
    """
    diagnostics: Dict[str, Any] = {}
    if exc is None:
        return diagnostics

    chain: List[Dict[str, str]] = []
    current: Optional[BaseException] = exc
    visited: set[int] = set()
    for _ in range(max(1, int(max_depth))):
        if current is None:
            break
        cid = id(current)
        if cid in visited:
            break
        visited.add(cid)
        chain.append(
            {
                "type": current.__class__.__name__,
                "message": str(current)[:240],
            }
        )
        nxt = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
        if isinstance(nxt, BaseException):
            current = nxt
            continue
        break

    if chain:
        diagnostics["error_chain"] = chain
        diagnostics["root_error_type"] = chain[-1]["type"]
        diagnostics["root_error"] = chain[-1]["message"]

    request = getattr(exc, "request", None)
    response = getattr(exc, "response", None)
    if request is None and response is not None:
        request = getattr(response, "request", None)
    if request is not None:
        method = getattr(request, "method", None)
        url_obj = getattr(request, "url", None)
        url_text = str(url_obj) if url_obj is not None else ""
        if method:
            diagnostics["request_method"] = str(method)
        if url_text:
            diagnostics["request_url"] = url_text
            try:
                diagnostics["request_host"] = urlparse(url_text).netloc
            except Exception:  # noqa: BLE001
                pass

    if response is not None:
        status_code = getattr(response, "status_code", None)
        if status_code not in (None, ""):
            diagnostics["response_status_code"] = status_code

    errno_value = getattr(exc, "errno", None)
    if errno_value not in (None, ""):
        diagnostics["errno"] = str(errno_value)

    return diagnostics


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
        llm_global_rpm: Optional[float] = None,
        print_call_counter: bool = False,
        slow_call_threshold: float = 30.0,
        recycle_on_connection_error: bool = True,
        recycle_after_calls: int = 0,
        recycle_min_interval: float = 5.0,
        recycle_defer_close_seconds: float = 120.0,
        retry_event_log_path: Optional[str] = None,
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
        self.recycle_defer_close_seconds = max(0.0, float(recycle_defer_close_seconds))
        self.retry_event_log_path = str(retry_event_log_path or "").strip() or None
        self.llm_global_rpm = (
            configure_global_llm_rpm(llm_global_rpm)
            if llm_global_rpm is not None and float(llm_global_rpm) > 0
            else None
        )
        self.llm_global_concurrency = (
            configure_global_llm_concurrency(llm_global_concurrency)
            if (self.llm_global_rpm is None and llm_global_concurrency is not None)
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
                diagnostics = extract_error_diagnostics(exc)
                root_type = diagnostics.get("root_error_type", "")
                root_error = diagnostics.get("root_error", "")
                print(
                    f"[llm-retry] schema={schema_name} parse_retry={parse_attempt + 1}/{self.max_retries} "
                    f"delay={delay:.2f}s code={error_code or 'N/A'} "
                    f"err={exc.__class__.__name__}: {str(exc)[:180]} "
                    f"root={root_type}: {str(root_error)[:140]}",
                    flush=True,
                )
                self._append_retry_event(
                    schema_name=schema_name,
                    phase="parse_retry",
                    attempt=parse_attempt + 1,
                    delay=delay,
                    error_code=error_code,
                    exc=exc,
                    diagnostics=diagnostics,
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
                if self.llm_global_rpm is not None:
                    async with global_llm_rpm_guard(self.llm_global_rpm):
                        if self.hard_timeout > 0:
                            response = await asyncio.wait_for(
                                llm.ainvoke(lc_messages),
                                timeout=self.hard_timeout,
                            )
                        else:
                            response = await llm.ainvoke(lc_messages)
                elif self.llm_global_concurrency is not None:
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
                diagnostics = extract_error_diagnostics(exc)
                root_type = diagnostics.get("root_error_type", "")
                root_error = diagnostics.get("root_error", "")
                request_host = diagnostics.get("request_host", "")
                print(
                    f"[llm-retry] schema={schema_name} retry={attempt + 1}/{self.max_retries} "
                    f"delay={delay:.2f}s code={error_code or 'N/A'} "
                    f"err={exc.__class__.__name__}: {str(exc)[:180]} "
                    f"root={root_type}: {str(root_error)[:140]} host={request_host}",
                    flush=True,
                )
                self._append_retry_event(
                    schema_name=schema_name,
                    phase="call_retry",
                    attempt=attempt + 1,
                    delay=delay,
                    error_code=error_code,
                    exc=exc,
                    diagnostics=diagnostics,
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
            self.llm = await reset_openai_client(
                self.llm,
                reason=reason,
                logger=self.logger,
                defer_close_seconds=self.recycle_defer_close_seconds,
            )
            self._last_recycle_ts = time.monotonic()
            if self.logger is not None:
                self.logger.info("LLM socket recycle done: reason=%s", reason)

    async def _close_llm_clients(self) -> None:
        await _close_chat_openai_clients(self.llm, logger=self.logger)

    def _append_retry_event(
        self,
        *,
        schema_name: str,
        phase: str,
        attempt: int,
        delay: float,
        error_code: Optional[str],
        exc: Exception,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> None:
        log_path = self.retry_event_log_path
        if not log_path:
            return
        record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "schema": schema_name,
            "phase": phase,
            "attempt": int(attempt),
            "max_retries": int(self.max_retries),
            "delay_seconds": float(delay),
            "error_type": exc.__class__.__name__,
            "error": str(exc),
            "error_code": error_code,
            "diagnostics": diagnostics or {},
        }
        try:
            path = Path(log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with _GLOBAL_RETRY_LOG_LOCK:
                with path.open("a", encoding="utf-8") as fp:
                    fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as log_exc:  # noqa: BLE001
            if self.logger is not None:
                self.logger.debug("Ignore retry event log write error: %s", log_exc)


def build_json_llm_client_from_config(
    llm: Any,
    *,
    config: Mapping[str, Any],
    logger: Optional[logging.Logger] = None,
    llm_global_concurrency: Optional[int] = None,
    llm_global_rpm: Optional[float] = None,
    print_call_counter: bool = False,
    slow_call_threshold: float = 30.0,
) -> LangChainJsonLLMClient:
    """
    Build LangChainJsonLLMClient from shared config keys.
    """
    if llm_global_concurrency is None:
        raw_global = config.get("llm_global_concurrency", 10)
        llm_global_concurrency = int(raw_global) if raw_global is not None else None
    if llm_global_rpm is None:
        raw_rpm = config.get("llm_global_rpm", None)
        try:
            parsed_rpm = float(raw_rpm) if raw_rpm is not None else 0.0
        except Exception:  # noqa: BLE001
            parsed_rpm = 0.0
        if parsed_rpm > 0:
            llm_global_rpm = parsed_rpm
        elif llm_global_concurrency is not None:
            # Compatibility mapping: old "global_concurrency" now paces RPM.
            llm_global_rpm = float(llm_global_concurrency)

    max_retries = max(0, int(config.get("max_retries", 8)))
    retry_delay = float(config.get("retry_delay", 2.0))
    retry_backoff_factor = float(config.get("retry_backoff_factor", 2.5))
    retry_max_delay = float(config.get("retry_max_delay", 120.0))
    hard_timeout = float(config.get("llm_call_hard_timeout", 180.0))
    effective_print_counter = bool(print_call_counter)
    effective_slow_threshold = max(0.0, float(slow_call_threshold))
    recycle_on_connection_error = bool(
        config.get("llm_socket_recycle_on_connection_error", True)
    )
    recycle_after_calls = int(config.get("llm_socket_recycle_after_calls", 0))
    recycle_min_interval = float(config.get("llm_socket_recycle_min_interval", 5.0))
    recycle_defer_close_seconds = float(
        config.get("llm_socket_recycle_defer_close_seconds", 120.0)
    )
    retry_event_log_path = str(
        config.get("llm_retry_log_path", "logs/llm_retry_events.jsonl")
    )
    singleton_key = (
        id(llm),
        max_retries,
        retry_delay,
        retry_backoff_factor,
        retry_max_delay,
        hard_timeout,
        llm_global_concurrency,
        llm_global_rpm,
        effective_print_counter,
        effective_slow_threshold,
        recycle_on_connection_error,
        recycle_after_calls,
        recycle_min_interval,
        recycle_defer_close_seconds,
        retry_event_log_path,
    )

    with _GLOBAL_JSON_LLM_CLIENT_LOCK:
        shared = _GLOBAL_JSON_LLM_CLIENTS.get(singleton_key)
        if shared is not None:
            if shared.logger is None and logger is not None:
                shared.logger = logger
            print(
                "[llm-client] shared-json-client "
                f"action=reuse id={id(shared)} llm_id={id(shared.llm)} "
                f"mode={'rpm' if shared.llm_global_rpm is not None else 'concurrency'} "
                f"rpm={shared.llm_global_rpm or 0}",
                flush=True,
            )
            return shared

        shared = LangChainJsonLLMClient(
            llm,
            logger=logger,
            max_retries=max_retries,
            retry_delay=retry_delay,
            retry_backoff_factor=retry_backoff_factor,
            retry_max_delay=retry_max_delay,
            hard_timeout=hard_timeout,
            llm_global_concurrency=llm_global_concurrency,
            llm_global_rpm=llm_global_rpm,
            print_call_counter=effective_print_counter,
            slow_call_threshold=effective_slow_threshold,
            recycle_on_connection_error=recycle_on_connection_error,
            recycle_after_calls=recycle_after_calls,
            recycle_min_interval=recycle_min_interval,
            recycle_defer_close_seconds=recycle_defer_close_seconds,
            retry_event_log_path=retry_event_log_path,
        )
        _GLOBAL_JSON_LLM_CLIENTS[singleton_key] = shared
        print(
            "[llm-client] shared-json-client "
            f"action=create id={id(shared)} llm_id={id(shared.llm)} "
            f"mode={'rpm' if shared.llm_global_rpm is not None else 'concurrency'} "
            f"rpm={shared.llm_global_rpm or 0}",
            flush=True,
        )
        return shared
