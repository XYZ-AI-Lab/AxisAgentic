# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Session-scoped code-execution tools backed by an E2B sandbox.

A task attempt lazily provisions one E2B sandbox on the first code tool call, later
code tool calls in that attempt reuse it, and the orchestrator tears it down when
the attempt finishes. There is no sandbox lifecycle for the model to manage and
no MCP server framing, so the tools register as flat native function tools
exactly like the web-search tools in this package.

The preferred interface is split by semantics:

* ``python_exec(code=...)`` runs Python source in the sandbox's persistent
  notebook-like interpreter. Variables, imports, functions, and files persist
  across Python calls in the same task attempt.
* ``shell_exec(cmd=...)`` runs shell commands in the same sandbox and is intended
  for shell-native work such as ``ls``, ``cat``, ``make``, ``gcc/g++``,
  ``curl/wget``, and package installation.

A new attempt gets a fresh sandbox.

Transient E2B failures — notably HTTP 429 rate limits — are retried with capped
exponential backoff instead of being surfaced as the tool response, reusing the
same retry primitives as the summary-LLM path
(:class:`agentic.tools.web_search._retry.RetryConfig` + ``compute_backoff``). A
rate-limited call waits and retries until it genuinely succeeds or the retry
budget is exhausted.

Requires the ``e2b-code-interpreter`` package and a valid ``E2B_API_KEY``.
"""

from __future__ import annotations

import asyncio
import contextlib
import html
import importlib.util
import logging
import os
import re
import sys
from dataclasses import dataclass
from typing import Any

from agentic.contracts.messages import ToolResultStatus
from agentic.tools.base import CallableTool, ToolResult
from agentic.tools.web_search._retry import RetryConfig, compute_backoff

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SANDBOX_TIMEOUT = 600  # seconds
DEFAULT_E2B_TEMPLATE_ID = "1av7fdjfvcparqo8efq6"
MAX_RESULT_LEN = 20_000
MAX_ERROR_LEN = 4_000

E2B_IMPORT_NAME = "e2b_code_interpreter"
E2B_PACKAGE_NAME = "e2b-code-interpreter"
E2B_MISSING_MESSAGE = f"Error: {E2B_PACKAGE_NAME} package is not installed in the agentic runtime. Install it with: pip install {E2B_PACKAGE_NAME}"

DEFAULT_PYTHON_EXEC_DESCRIPTION = (
    "Execute Python source code in the shared stateful Python interpreter for this task attempt.\n\n"
    "Use this as the default tool for calculations, enumeration, symbolic math (sympy), "
    "numeric work (numpy/scipy), data processing, simulations, writing/reading files, "
    "and verifying intermediate results. All python_exec calls in the same task attempt "
    "reuse one notebook-like interpreter, so variables, imports, functions, installed "
    "packages, and files persist across calls. python_exec and shell_exec share the same "
    "sandbox filesystem and package environment: packages installed with shell_exec can "
    "be imported here, and files written by shell_exec can be read here. Do not wrap "
    "Python in shell heredocs and do not pass Python source to shell_exec.\n\n"
    "Args:\n"
    "    code: Python source code to execute.\n\n"
    "Returns:\n"
    "    XML containing the Python execution result."
)

DEFAULT_SHELL_EXEC_DESCRIPTION = (
    "Execute a shell command in the shared sandbox for this task attempt.\n\n"
    "Use this only when the task genuinely needs shell/program/filesystem semantics, "
    "such as ls/cat/touch, make, gcc/g++, curl/wget, pip install, or probing system "
    "programs. Prefer python_exec for Python source code. Do not pass Python source code to "
    "shell_exec; call python_exec(code=...) instead. Files, installed packages, and "
    "working-directory state persist across calls in the same task attempt. shell_exec "
    "and python_exec share the same sandbox filesystem and package environment: a package "
    "installed here can be imported by python_exec, and files written here can be read by "
    "python_exec.\n\n"
    "Args:\n"
    "    cmd: The shell command to execute in the shared sandbox.\n\n"
    "Returns:\n"
    "    XML containing exit_code, stdout, and stderr."
)

SIMPLE_PYTHON_EXEC_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "code": {
            "type": "string",
            "description": "Python source code to execute in the shared stateful Python interpreter.",
        },
    },
    "required": ["code"],
}

SIMPLE_SHELL_EXEC_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "cmd": {
            "type": "string",
            "description": "Shell command to execute in the shared sandbox.",
        },
    },
    "required": ["cmd"],
}

# Leading "<status>: ..." prefix used by the E2B REST error formatter.
_STATUS_PREFIX_RE = re.compile(r"^\s*(\d{3})\b")


# ---------------------------------------------------------------------------
# Options + availability helpers
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class E2BSandboxOptions:
    """Connection/runtime options for E2B-backed code sandbox tools.

    Args:
        api_key: E2B API key. Falls back to the ``E2B_API_KEY`` environment
            variable when ``None``.
        template_id: E2B sandbox template ID to provision.
        sandbox_timeout: Sandbox keep-alive timeout in seconds.
    """

    api_key: str | None = None
    template_id: str = DEFAULT_E2B_TEMPLATE_ID
    sandbox_timeout: int = DEFAULT_SANDBOX_TIMEOUT

    def resolved_api_key(self) -> str:
        return self.api_key or os.environ.get("E2B_API_KEY", "")


def default_code_exec_retry() -> RetryConfig:
    """Default transient-error retry policy for E2B sandbox calls.

    Mirrors the summary-LLM retry profile (patient 30s base backoff with jitter,
    retry on 408/429 and 5xx) so an E2B 429 rate limit is waited out and retried
    rather than surfaced to the model as the tool response.
    """
    return RetryConfig(
        max_attempts=5,
        backoff_base_seconds=30.0,
        backoff_multiplier=2.0,
        backoff_max_seconds=240.0,
        retryable_status_codes=frozenset({408, 429}),
        jitter=True,
        respect_retry_after=True,
        retry_on_server_errors=True,
    )


def is_e2b_available() -> bool:
    """Return whether the E2B code-interpreter SDK is importable."""
    if E2B_IMPORT_NAME in sys.modules:
        return True
    return importlib.util.find_spec(E2B_IMPORT_NAME) is not None


def validate_code_sandbox_environment(*, options: E2BSandboxOptions | None = None) -> list[str]:
    """Return configuration problems that would make code sandbox tools fail.

    The tool executes in-process (not in a separate MCP server), so dependency
    and API-key problems should be surfaced before a long evaluation starts
    instead of failing every tool call mid-run.
    """
    opts = options or E2BSandboxOptions()
    problems: list[str] = []
    if not opts.resolved_api_key():
        problems.append("E2B_API_KEY is not set")
    if not is_e2b_available():
        problems.append(f"{E2B_PACKAGE_NAME} is not installed")
    return problems


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _truncate(text: str, limit: int) -> str:
    if len(text) > limit:
        return text[:limit] + " [truncated due to length limit]"
    return text


def _format_shell_result(result: Any) -> str:
    exit_code = getattr(result, "exit_code", None)
    if exit_code is None:
        exit_code = 0
    stdout = getattr(result, "stdout", "")
    stderr = getattr(result, "stderr", "")
    return (
        "<shell_results>\n"
        f"<exit_code>{html.escape(str(exit_code))}</exit_code>\n"
        f"<stdout>{html.escape(str(stdout or ''))}</stdout>\n"
        f"<stderr>{html.escape(str(stderr or ''))}</stderr>\n"
        "</shell_results>"
    )


def _format_python_result(execution: Any) -> str:
    return f"<python_results>\n{html.escape(str(execution or ''))}\n</python_results>"


def _load_async_sandbox() -> Any:
    try:
        from e2b_code_interpreter import AsyncSandbox  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(E2B_MISSING_MESSAGE) from exc
    return AsyncSandbox


def _e2b_error_status(exc: BaseException) -> int | None:
    """Best-effort HTTP status classification for an E2B exception.

    Returns the HTTP-equivalent status code when one can be determined, so the
    shared :class:`RetryConfig` can decide retryability the same way as the
    summary-LLM path. Returns ``None`` for non-transient / unclassifiable errors.
    """
    # Typed rate-limit exception (covers REST 429 and gRPC resource_exhausted).
    # e2b is optional; fall back to message parsing if it cannot be imported.
    with contextlib.suppress(Exception):
        from e2b.exceptions import RateLimitException  # type: ignore[import-untyped]

        if isinstance(exc, RateLimitException):
            return 429

    message = str(exc)
    match = _STATUS_PREFIX_RE.match(message)
    if match:
        return int(match.group(1))
    lowered = message.lower()
    if "rate limit" in lowered or "too many requests" in lowered:
        return 429
    return None


def _is_sandbox_not_found(exc: BaseException) -> bool:
    """Return whether an E2B exception means the cached sandbox handle expired."""
    lowered = str(exc).lower()
    return "sandbox was not found" in lowered or "sandbox not found" in lowered


# ---------------------------------------------------------------------------
# Session runtime
# ---------------------------------------------------------------------------


class SessionScopedCodeExecutionRuntime:
    """Shared E2B sandbox runtime used by Python and shell tools."""

    def __init__(
        self,
        *,
        options: E2BSandboxOptions,
        retry: RetryConfig,
    ) -> None:
        self._options = options
        self._retry = retry
        self._sandbox: Any | None = None
        self._sandbox_id: str | None = None
        self._task_id: str | None = None
        self._startup_retries = 0
        # Serializes lazy sandbox creation. Tool calls within an attempt are
        # dispatched serially today (ToolManager.execute_tool_calls awaits each
        # call in turn), so this is a guard against a future concurrent
        # same-turn dispatch rather than a fix for an active race.
        self._sandbox_lock = asyncio.Lock()

    async def begin_task(self, *, task_id: str | None = None) -> None:
        self._task_id = task_id

    async def end_task(self, *, task_id: str | None = None) -> None:  # noqa: ARG002
        sandbox = self._sandbox
        self._sandbox = None
        self._sandbox_id = None
        self._task_id = None
        self._startup_retries = 0
        if sandbox is not None:
            with contextlib.suppress(Exception):
                await sandbox.kill()

    async def _ensure_sandbox(self) -> Any:
        if self._sandbox is not None:
            return self._sandbox

        async with self._sandbox_lock:
            # Re-check under the lock: a concurrent caller may have created the
            # sandbox while we waited to acquire it.
            if self._sandbox is not None:
                return self._sandbox

            api_key = self._options.resolved_api_key()
            if not api_key:
                raise RuntimeError("E2B_API_KEY is not set.")

            async_sandbox = _load_async_sandbox()
            for attempt in range(1, self._retry.max_attempts + 1):
                try:
                    sandbox = await async_sandbox.create(
                        template=self._options.template_id,
                        timeout=self._options.sandbox_timeout,
                        api_key=api_key,
                    )
                    self._sandbox = sandbox
                    self._sandbox_id = await self._read_sandbox_id(sandbox)
                    self._startup_retries = attempt - 1
                    return sandbox
                except Exception as exc:  # retry/surface sandbox creation failures
                    status = _e2b_error_status(exc)
                    retryable = status is not None and self._retry.is_retryable_status(status)
                    if retryable and attempt < self._retry.max_attempts:
                        delay = compute_backoff(attempt, self._retry)
                        logger.warning(
                            "code_sandbox: E2B sandbox create transient error (status~%s) on attempt %d/%d; waiting %.1fs before retry",
                            status,
                            attempt,
                            self._retry.max_attempts,
                            delay,
                        )
                        await asyncio.sleep(delay)
                        continue
                    self._startup_retries = attempt - 1
                    raise

            raise RuntimeError("Failed to create E2B sandbox after exhausting retries.")

    async def _discard_sandbox(self) -> None:
        sandbox = self._sandbox
        self._sandbox = None
        self._sandbox_id = None
        if sandbox is not None:
            with contextlib.suppress(Exception):
                await sandbox.kill()

    async def _refresh_sandbox_timeout(self, sandbox: Any) -> None:
        set_timeout = getattr(sandbox, "set_timeout", None)
        if set_timeout is None:
            return
        with contextlib.suppress(Exception):
            await set_timeout(self._options.sandbox_timeout)

    @staticmethod
    async def _read_sandbox_id(sandbox: Any) -> str | None:
        try:
            info = await sandbox.get_info()
        except Exception:
            return None
        sandbox_id = getattr(info, "sandbox_id", None)
        return str(sandbox_id) if sandbox_id else None

    def _metadata(self, *, execution_kind: str, e2b_retries: int = 0) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "sandbox_scope": "task_attempt",
            "code_execution_kind": execution_kind,
        }
        if self._sandbox_id:
            metadata["sandbox_id"] = self._sandbox_id
        if self._task_id:
            metadata["sandbox_task_id"] = self._task_id
        total_retries = self._startup_retries + e2b_retries
        if total_retries:
            metadata["e2b_retries"] = total_retries
        return metadata

    async def _execute(
        self,
        *,
        execution_kind: str,
        operation_label: str,
        runner: Any,
        formatter: Any,
    ) -> ToolResult:
        api_key = self._options.resolved_api_key()
        if not api_key:
            return ToolResult(content="Error: E2B_API_KEY is not set.", status=ToolResultStatus.FAILED)

        try:
            _load_async_sandbox()
        except RuntimeError:
            return ToolResult(content=E2B_MISSING_MESSAGE, status=ToolResultStatus.FAILED)

        try:
            sandbox = await self._ensure_sandbox()
        except Exception as exc:  # surface sandbox startup failures as tool errors
            details = _truncate(str(exc), MAX_ERROR_LEN)
            metadata = {"e2b_retries": self._startup_retries} if self._startup_retries else {}
            return ToolResult(
                content=f"[ERROR]: Failed to start shared E2B sandbox. Exception type: {type(exc).__name__}, Details: {details}",
                status=ToolResultStatus.FAILED,
                metadata=metadata,
            )

        for attempt in range(1, self._retry.max_attempts + 1):
            try:
                await self._refresh_sandbox_timeout(sandbox)
                execution = await runner(sandbox)
                metadata = self._metadata(execution_kind=execution_kind, e2b_retries=attempt - 1)
                return ToolResult(content=_truncate(formatter(execution), MAX_RESULT_LEN), metadata=metadata)
            except Exception as exc:  # surface/retry any sandbox failure
                status = _e2b_error_status(exc)
                sandbox_not_found = _is_sandbox_not_found(exc)
                retryable = sandbox_not_found or (status is not None and self._retry.is_retryable_status(status))
                if retryable and attempt < self._retry.max_attempts:
                    if sandbox_not_found:
                        await self._discard_sandbox()
                        try:
                            sandbox = await self._ensure_sandbox()
                        except Exception:
                            raise exc from None
                    delay = compute_backoff(attempt, self._retry)
                    logger.warning(
                        "%s: E2B transient error (status~%s) on attempt %d/%d; waiting %.1fs before retry",
                        execution_kind,
                        status,
                        attempt,
                        self._retry.max_attempts,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                details = _truncate(str(exc), MAX_ERROR_LEN)
                metadata = self._metadata(execution_kind=execution_kind, e2b_retries=attempt - 1)
                return ToolResult(
                    content=f"[ERROR]: Failed to run {operation_label}. Exception type: {type(exc).__name__}, Details: {details}",
                    status=ToolResultStatus.FAILED,
                    metadata=metadata,
                )

        return ToolResult(content=f"[ERROR]: Failed to run {operation_label} after exhausting retries.", status=ToolResultStatus.FAILED)

    async def shell_exec(self, cmd: str) -> ToolResult:
        async def _runner(sandbox: Any) -> Any:
            return await sandbox.commands.run(cmd)

        return await self._execute(
            execution_kind="shell",
            operation_label="command",
            runner=_runner,
            formatter=_format_shell_result,
        )

    async def python_exec(self, code: str) -> ToolResult:
        async def _runner(sandbox: Any) -> Any:
            run_code = getattr(sandbox, "run_code", None)
            if run_code is None:
                raise RuntimeError("E2B sandbox does not expose run_code; cannot execute Python without shell fallback.")
            return await run_code(code)

        return await self._execute(
            execution_kind="python",
            operation_label="Python code",
            runner=_runner,
            formatter=_format_python_result,
        )


class SessionScopedCodeExecutionTool(CallableTool):
    """Flat code execution tool backed by a shared E2B runtime."""

    def __init__(
        self,
        *,
        name: str,
        description: str,
        parameters: dict[str, Any],
        fn: Any,
        runtime: SessionScopedCodeExecutionRuntime,
        server_name: str | None,
        max_calls_per_task: int | None,
        max_consecutive_calls_with_same_args: int | None,
        emoji: str,
    ) -> None:
        self._runtime = runtime
        super().__init__(
            name=name,
            description=description,
            parameters=parameters,
            fn=fn,
            strict_mode=False,
            server_name=server_name,
            max_calls_per_task=max_calls_per_task,
            max_consecutive_calls_with_same_args=max_consecutive_calls_with_same_args,
            emoji=emoji,
        )

    async def begin_task(self, *, task_id: str | None = None) -> None:
        await self._runtime.begin_task(task_id=task_id)

    async def end_task(self, *, task_id: str | None = None) -> None:
        await self._runtime.end_task(task_id=task_id)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_python_exec_tool(
    *,
    runtime: SessionScopedCodeExecutionRuntime | None = None,
    name: str = "python_exec",
    description: str = DEFAULT_PYTHON_EXEC_DESCRIPTION,
    parameters: dict[str, Any] | None = None,
    options: E2BSandboxOptions | None = None,
    retry: RetryConfig | None = None,
    server_name: str | None = None,
    max_calls_per_task: int | None = None,
    max_consecutive_calls_with_same_args: int | None = 3,
) -> CallableTool:
    """Create the stateful Python interpreter tool.

    When ``runtime`` is supplied, the returned tool shares the same E2B sandbox
    with other tools using that runtime.
    """
    runtime = runtime or SessionScopedCodeExecutionRuntime(
        options=options or E2BSandboxOptions(),
        retry=retry or default_code_exec_retry(),
    )
    return SessionScopedCodeExecutionTool(
        name=name,
        description=description,
        parameters=parameters if parameters is not None else SIMPLE_PYTHON_EXEC_PARAMETERS,
        fn=runtime.python_exec,
        runtime=runtime,
        server_name=server_name,
        max_calls_per_task=max_calls_per_task,
        max_consecutive_calls_with_same_args=max_consecutive_calls_with_same_args,
        emoji="🐍",
    )


def create_shell_exec_tool(
    *,
    runtime: SessionScopedCodeExecutionRuntime | None = None,
    name: str = "shell_exec",
    description: str = DEFAULT_SHELL_EXEC_DESCRIPTION,
    parameters: dict[str, Any] | None = None,
    options: E2BSandboxOptions | None = None,
    retry: RetryConfig | None = None,
    server_name: str | None = None,
    max_calls_per_task: int | None = None,
    max_consecutive_calls_with_same_args: int | None = 3,
) -> CallableTool:
    """Create the low-frequency shell command tool."""
    runtime = runtime or SessionScopedCodeExecutionRuntime(
        options=options or E2BSandboxOptions(),
        retry=retry or default_code_exec_retry(),
    )
    return SessionScopedCodeExecutionTool(
        name=name,
        description=description,
        parameters=parameters if parameters is not None else SIMPLE_SHELL_EXEC_PARAMETERS,
        fn=runtime.shell_exec,
        runtime=runtime,
        server_name=server_name,
        max_calls_per_task=max_calls_per_task,
        max_consecutive_calls_with_same_args=max_consecutive_calls_with_same_args,
        emoji="🐚",
    )


def create_code_sandbox_tools(
    *,
    options: E2BSandboxOptions | None = None,
    retry: RetryConfig | None = None,
    server_name: str | None = None,
    max_calls_per_task: int | None = None,
    max_consecutive_calls_with_same_args: int | None = 3,
) -> list[CallableTool]:
    """Create ``python_exec`` and ``shell_exec`` sharing one task sandbox.

    ``python_exec`` is intentionally returned first because it is the default
    coding path; ``shell_exec`` is an escape hatch for shell-native tasks.
    """
    runtime = SessionScopedCodeExecutionRuntime(
        options=options or E2BSandboxOptions(),
        retry=retry or default_code_exec_retry(),
    )
    return [
        create_python_exec_tool(
            runtime=runtime,
            server_name=server_name,
            max_calls_per_task=max_calls_per_task,
            max_consecutive_calls_with_same_args=max_consecutive_calls_with_same_args,
        ),
        create_shell_exec_tool(
            runtime=runtime,
            server_name=server_name,
            max_calls_per_task=max_calls_per_task,
            max_consecutive_calls_with_same_args=max_consecutive_calls_with_same_args,
        ),
    ]
