# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Session-scoped code-execution tools backed by E2B sandboxes.

Provides:

* :func:`create_code_sandbox_tools` — register the preferred
  ``python_exec`` + ``shell_exec`` pair backed by one shared E2B sandbox per
  task attempt (no MCP server framing).
"""

from agentic.tools.code_sandbox.code_exec import (
    DEFAULT_E2B_TEMPLATE_ID,
    DEFAULT_PYTHON_EXEC_DESCRIPTION,
    DEFAULT_SANDBOX_TIMEOUT,
    DEFAULT_SHELL_EXEC_DESCRIPTION,
    E2B_PACKAGE_NAME,
    SIMPLE_PYTHON_EXEC_PARAMETERS,
    SIMPLE_SHELL_EXEC_PARAMETERS,
    E2BSandboxOptions,
    create_code_sandbox_tools,
    create_python_exec_tool,
    create_shell_exec_tool,
    default_code_exec_retry,
    is_e2b_available,
    validate_code_sandbox_environment,
)

__all__ = [
    "DEFAULT_E2B_TEMPLATE_ID",
    "DEFAULT_PYTHON_EXEC_DESCRIPTION",
    "DEFAULT_SANDBOX_TIMEOUT",
    "DEFAULT_SHELL_EXEC_DESCRIPTION",
    "E2B_PACKAGE_NAME",
    "SIMPLE_PYTHON_EXEC_PARAMETERS",
    "SIMPLE_SHELL_EXEC_PARAMETERS",
    "E2BSandboxOptions",
    "create_code_sandbox_tools",
    "create_python_exec_tool",
    "create_shell_exec_tool",
    "default_code_exec_retry",
    "is_e2b_available",
    "validate_code_sandbox_environment",
]
