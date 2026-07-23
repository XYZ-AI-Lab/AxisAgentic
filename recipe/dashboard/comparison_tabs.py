# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Config, prompt, and task-description comparison helpers."""

from __future__ import annotations

import contextlib
import json
from typing import TYPE_CHECKING, Any

import streamlit as st

from recipe.dashboard.discovery import _all_task_ids, _latest_path
from recipe.dashboard.loading import (
    _agentic_system_prompt,
    _agentic_system_prompt_source,
    _load_json_cached,
    _ori_messages,
    _ori_system_prompt,
)
from recipe.dashboard.rendering import _render_wrapped_text

if TYPE_CHECKING:
    from recipe.dashboard.sides import DashboardSide

# ---------------------------------------------------------------------------
# UI: Config tab
# ---------------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_structured_config(path: Any) -> dict[str, Any] | None:
    if not path.exists():
        return None
    if path.suffix == ".json":
        d = _load_json_cached(str(path))
        if "_load_error" in d:
            return None
        return d if isinstance(d, dict) else {"value": d}
    if path.suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError:
            return {"_raw_text": path.read_text(encoding="utf-8")}
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else {"value": loaded}
    return None


def _read_text_file(path: Any) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return ""


def _dump_config_text(config: dict[str, Any]) -> str:
    try:
        import yaml

        return yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    except ImportError:
        import json

        return json.dumps(config, indent=2, sort_keys=False, ensure_ascii=False)


def _side_effective_config(side: DashboardSide) -> tuple[dict[str, Any], list[str]]:
    if not side.run:
        return {}, []

    sources: list[str] = []
    merged: dict[str, Any] = {}

    effective_path = side.run / "run_config.effective.yaml"
    effective = _load_structured_config(effective_path)
    if effective is not None:
        sources.append("run_config.effective.yaml")
        sources_path = side.run / "run_config.sources.json"
        config_sources = _load_structured_config(sources_path)
        if config_sources is not None:
            sources.append("run_config.sources.json")
            return {"effective_config": effective, "config_sources": config_sources}, sources
        return effective, sources

    for hydra_name in (".hydra", ".hybra"):
        hydra_dir = side.run / hydra_name
        for name in ("config.yaml", "hydra.yaml", "overrides.yaml"):
            path = hydra_dir / name
            data = _load_structured_config(path)
            if data is not None:
                merged = _deep_merge(merged, data)
                sources.append(str(path.relative_to(side.run)))

    for name in ("run_config.input.yaml", "env_info.json", "run_config.json"):
        path = side.run / name
        data = _load_structured_config(path)
        if data is not None:
            merged = _deep_merge(merged, data)
            sources.append(name)

    metadata_path = side.run / "run_metadata.json"
    metadata = _load_structured_config(metadata_path)
    if metadata is not None:
        effective = metadata.get("effective_config")
        if isinstance(effective, dict):
            # The runner records this after CLI overrides have been resolved, so
            # it should take precedence over Hydra defaults and intermediate files.
            merged = _deep_merge(merged, effective)
        cli_args = metadata.get("cli_args")
        if isinstance(cli_args, dict):
            merged["cli_args"] = cli_args
        for key in ("entrypoint", "created_at", "git", "artifacts"):
            if key in metadata:
                merged[key] = metadata[key]
        sources.append("run_metadata.json")

    if not merged and side.kind == "original":
        for f in sorted(side.run.glob("task_*.json"))[:1]:
            env = _load_json_cached(str(f)).get("env_info", {})
            if env:
                merged = {"env_info": env}
                sources.append(f.name)
                break

    return merged, sources


def _side_effective_config_text(side: DashboardSide) -> tuple[str, list[str]]:
    """Return the effective run config as comparable text, preferring YAML."""
    if not side.run:
        return "", []

    effective_path = side.run / "run_config.effective.yaml"
    if effective_path.exists():
        return _read_text_file(effective_path), ["run_config.effective.yaml"]

    effective_config, sources = _side_effective_config(side)
    if not effective_config:
        return "", sources
    return _dump_config_text(effective_config), sources


def _render_config_comparison_sides(left: DashboardSide, right: DashboardSide) -> None:
    st.header("Config Comparison")
    left_text, left_sources = _side_effective_config_text(left)
    right_text, right_sources = _side_effective_config_text(right)

    c1, c2, c3 = st.columns(3)
    c1.metric(f"{left.label} length", f"{len(left_text):,} chars" if left_text else "-")
    c2.metric(f"{right.label} length", f"{len(right_text):,} chars" if right_text else "-")
    if left_text and right_text:
        c3.metric("Match", "Yes" if left_text.strip() == right_text.strip() else "No")

    if left_sources:
        st.caption(f"{left.label} config: " + ", ".join(f"`{source}`" for source in left_sources))
    if right_sources:
        st.caption(f"{right.label} config: " + ", ".join(f"`{source}`" for source in right_sources))

    view = st.radio("View", ["Diff", "Side by Side", left.label, right.label, "JSON"], horizontal=True, key="config_view")
    if view == "Diff":
        if left_text and right_text:
            _render_side_by_side_diff(
                left_text, right_text, left_label=left.label, right_label=right.label, identical_label="Effective configs are identical."
            )
        else:
            st.info("Both sides needed for diff.")
    elif view == "Side by Side":
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown(f"**{left.label}**")
            st.code(left_text, language="yaml") if left_text else st.info("Not available.")
        with col_r:
            st.markdown(f"**{right.label}**")
            st.code(right_text, language="yaml") if right_text else st.info("Not available.")
    elif view == left.label:
        st.code(left_text, language="yaml") if left_text else st.info("Not available.")
    elif view == right.label:
        st.code(right_text, language="yaml") if right_text else st.info("Not available.")
    else:
        cols = st.columns(2)
        for col, side in zip(cols, (left, right), strict=False):
            with col:
                st.subheader(side.label)
                effective_config, sources = _side_effective_config(side)
                if not effective_config:
                    st.info("No config found.")
                    continue
                st.caption("Loaded from: " + ", ".join(f"`{source}`" for source in sources))
                st.json(effective_config)


def _char_diff_highlight(left: str, right: str, del_color: str, ins_color: str) -> tuple[str, str]:
    """Return (left_html, right_html) with character-level diff spans.

    Spaces inside highlighted spans are rendered as ``\u2423`` (open-box ␣)
    so that whitespace differences are immediately visible.
    """
    import difflib
    import html as _html

    sm = difflib.SequenceMatcher(None, left, right, autojunk=False)
    l_parts: list[str] = []
    r_parts: list[str] = []

    def _esc_visible(text: str, *, inside_highlight: bool) -> str:
        """HTML-escape *text*, making spaces visible when inside a highlight span."""
        escaped = _html.escape(text)
        if inside_highlight:
            # Show each space as ␣ so whitespace diffs are obvious
            escaped = escaped.replace(" ", "\u2423")
        return escaped

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            l_parts.append(_esc_visible(left[i1:i2], inside_highlight=False))
            r_parts.append(_esc_visible(right[j1:j2], inside_highlight=False))
        elif tag == "replace":
            l_parts.append(f'<span style="background:{del_color};border-radius:2px;">{_esc_visible(left[i1:i2], inside_highlight=True)}</span>')
            r_parts.append(f'<span style="background:{ins_color};border-radius:2px;">{_esc_visible(right[j1:j2], inside_highlight=True)}</span>')
        elif tag == "delete":
            l_parts.append(f'<span style="background:{del_color};border-radius:2px;">{_esc_visible(left[i1:i2], inside_highlight=True)}</span>')
        elif tag == "insert":
            r_parts.append(f'<span style="background:{ins_color};border-radius:2px;">{_esc_visible(right[j1:j2], inside_highlight=True)}</span>')

    return "".join(l_parts), "".join(r_parts)


_DiffRow = tuple[int | None, str | None, int | None, str | None, str]


def _compute_diff_rows(left_text: str, right_text: str) -> list[_DiffRow]:
    """Compute side-by-side diff rows between *left_text* and *right_text*.

    Blank lines are only matched against other blank lines.
    Returns a list of ``(left_num, left_text, right_num, right_text, kind)``
    tuples where *kind* is ``'delete'``, ``'insert'``, ``'replace'``, or ``'sep'``.
    """
    import difflib

    left_lines = left_text.splitlines()
    right_lines = right_text.splitlines()

    _BLANK = "\x00BLANK\x00"
    left_norm = [_BLANK if not ln.strip() else ln for ln in left_lines]
    right_norm = [_BLANK if not ln.strip() else ln for ln in right_lines]

    sm = difflib.SequenceMatcher(None, left_norm, right_norm, autojunk=False)
    rows: list[_DiffRow] = []
    last_was_equal = False

    def _emit_replace(i1: int, i2: int, j1: int, j2: int) -> None:
        """Sub-align a replace block with the same blank-sentinel trick."""
        sub = difflib.SequenceMatcher(None, left_norm[i1:i2], right_norm[j1:j2], autojunk=False)
        for stag, a1, a2, b1, b2 in sub.get_opcodes():
            if stag == "equal":
                continue
            if stag == "replace":
                n = max(a2 - a1, b2 - b1)
                for k in range(n):
                    li = i1 + a1 + k if k < (a2 - a1) else None
                    ri = j1 + b1 + k if k < (b2 - b1) else None
                    rows.append(
                        (
                            li + 1 if li is not None else None,
                            left_lines[li] if li is not None else None,
                            ri + 1 if ri is not None else None,
                            right_lines[ri] if ri is not None else None,
                            "replace",
                        )
                    )
            elif stag == "delete":
                for k in range(a1, a2):
                    rows.append((i1 + k + 1, left_lines[i1 + k], None, None, "delete"))
            elif stag == "insert":
                for k in range(b1, b2):
                    rows.append((None, None, j1 + k + 1, right_lines[j1 + k], "insert"))

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            last_was_equal = True
            continue
        if last_was_equal and rows:
            rows.append((None, None, None, None, "sep"))
        last_was_equal = False

        if tag == "replace":
            _emit_replace(i1, i2, j1, j2)
        elif tag == "delete":
            for k in range(i1, i2):
                rows.append((k + 1, left_lines[k], None, None, "delete"))
        elif tag == "insert":
            for k in range(j1, j2):
                rows.append((None, None, k + 1, right_lines[k], "insert"))

    return rows


def _diff_rows_to_html(rows: list[_DiffRow], *, left_label: str = "Original", right_label: str = "Agentic") -> str:
    """Render diff rows into an HTML table string with char-level highlights."""
    import html as _html

    _DEL_BG = "#ffeef0"
    _INS_BG = "#e6ffec"
    _DEL_HL = "#f5c6cb"
    _INS_HL = "#a6f1c0"
    _SEP_BG = "#f6f8fa"
    _NUM_CSS = "color:#888;text-align:right;padding:0 4px;user-select:none;border-right:1px solid #ddd;min-width:3em;"
    _CELL_CSS = "padding:1px 6px;white-space:pre-wrap;word-wrap:break-word;"

    _BG_MAP = {
        "delete": (_DEL_BG, "transparent"),
        "insert": ("transparent", _INS_BG),
        "replace": (_DEL_BG, _INS_BG),
    }

    parts = [
        '<div style="overflow-x:auto;max-height:700px;overflow-y:auto;border:1px solid #ddd;border-radius:6px;">',
        '<table style="width:100%;border-collapse:collapse;font-family:monospace;font-size:13px;">',
        f'<tr><th style="{_NUM_CSS}background:#f6f8fa;">#</th>'
        f'<th style="{_CELL_CSS}background:#f6f8fa;font-weight:bold;">{left_label}</th>'
        f'<th style="{_NUM_CSS}background:#f6f8fa;border-left:2px solid #ccc;">#</th>'
        f'<th style="{_CELL_CSS}background:#f6f8fa;font-weight:bold;">{right_label}</th></tr>',
    ]

    for l_num, l_text, r_num, r_text, kind in rows:
        if kind == "sep":
            parts.append(
                f'<tr><td colspan="4" style="background:{_SEP_BG};text-align:center;'
                f'padding:2px;color:#aaa;font-size:11px;border-top:1px solid #eee;border-bottom:1px solid #eee;">'
                f"\u00b7\u00b7\u00b7</td></tr>"
            )
            continue

        l_bg, r_bg = _BG_MAP.get(kind, ("transparent", "transparent"))
        ln = str(l_num) if l_num is not None else ""
        rn = str(r_num) if r_num is not None else ""

        if kind == "replace" and l_text is not None and r_text is not None:
            lt, rt = _char_diff_highlight(l_text, r_text, _DEL_HL, _INS_HL)
        else:
            lt = _html.escape(l_text) if l_text is not None else ""
            rt = _html.escape(r_text) if r_text is not None else ""

        parts.append(
            f"<tr>"
            f'<td style="{_NUM_CSS}background:{l_bg};">{ln}</td>'
            f'<td style="{_CELL_CSS}background:{l_bg};">{lt}</td>'
            f'<td style="{_NUM_CSS}background:{r_bg};border-left:2px solid #ccc;">{rn}</td>'
            f'<td style="{_CELL_CSS}background:{r_bg};">{rt}</td>'
            f"</tr>"
        )

    parts.append("</table></div>")
    return "".join(parts)


def _render_side_by_side_diff(
    left_text: str,
    right_text: str,
    *,
    left_label: str = "Original",
    right_label: str = "Agentic",
    identical_label: str = "System prompts are identical.",
) -> None:
    """Render a side-by-side diff showing only lines with differences.

    Empty/blank lines are only matched against other empty lines,
    never against content lines, so the alignment stays intuitive.
    Character-level differences (including spaces) are highlighted inline.
    """
    rows = _compute_diff_rows(left_text, right_text)
    if not rows:
        st.success(identical_label)
        return

    n_del = sum(1 for r in rows if r[4] == "delete")
    n_ins = sum(1 for r in rows if r[4] == "insert")
    n_chg = sum(1 for r in rows if r[4] == "replace")
    st.caption(f"Changed: {n_chg}  |  Removed: {n_del}  |  Added: {n_ins}")
    st.markdown(_diff_rows_to_html(rows, left_label=left_label, right_label=right_label), unsafe_allow_html=True)


def _side_system_prompt(side: DashboardSide) -> str:
    if not side.run:
        return ""
    if side.kind == "original":
        for attempts in side.index.values():
            prompt = _ori_system_prompt(_load_json_cached(_latest_path(attempts)))
            if prompt:
                return prompt
        for f in sorted(side.run.glob("task_*.json")):
            prompt = _ori_system_prompt(_load_json_cached(str(f)))
            if prompt:
                return prompt
        return ""
    return _agentic_system_prompt(side.run)


def _side_system_prompt_source(side: DashboardSide) -> str:
    if not side.run:
        return ""
    if side.kind == "original":
        return "Loaded from first original task log with `main_agent_message_history.system_prompt`"
    return _agentic_system_prompt_source(side.run)


def _pretty_print_json_string_lines(text: str) -> tuple[str, int]:
    """Expand whole-line JSON strings, such as chat-template tool schemas.

    This is a dashboard-only view transform. The logged rendered prompt remains
    byte-for-byte unchanged; valid compact JSON lines are shown as fenced JSON
    blocks so tool schemas are easier to scan.
    """
    formatted_lines: list[str] = []
    changed = 0
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("{", "[")) and stripped.endswith(("}", "]")):
            with contextlib.suppress(json.JSONDecodeError, TypeError, ValueError):
                parsed = json.loads(stripped)
                if isinstance(parsed, (dict, list)):
                    indent = line[: len(line) - len(line.lstrip())]
                    pretty = json.dumps(parsed, indent=2, ensure_ascii=False)
                    if formatted_lines and formatted_lines[-1].strip():
                        formatted_lines.append("")
                    formatted_lines.append(f"{indent}```json")
                    formatted_lines.extend(f"{indent}{pretty_line}" if pretty_line else pretty_line for pretty_line in pretty.splitlines())
                    formatted_lines.append(f"{indent}```")
                    formatted_lines.append("")
                    changed += 1
                    continue
        formatted_lines.append(line)
    return "\n".join(formatted_lines).rstrip(), changed


def _render_system_prompt_sides(left: DashboardSide, right: DashboardSide) -> None:
    st.header("System Prompt")
    st.caption("For chat-template runs, this loads `extracted_system_block` from `chat_template_render.json`.")
    left_prompt = _side_system_prompt(left)
    right_prompt = _side_system_prompt(right)
    if not left_prompt and not right_prompt:
        st.info("No system prompts found.")
        return
    left_source = _side_system_prompt_source(left)
    right_source = _side_system_prompt_source(right)
    if left_source:
        st.caption(f"{left.label} system prompt: {left_source}")
    if right_source:
        st.caption(f"{right.label} system prompt: {right_source}")

    left_prompt_display, left_json_lines = _pretty_print_json_string_lines(left_prompt)
    right_prompt_display, right_json_lines = _pretty_print_json_string_lines(right_prompt)
    if left_json_lines or right_json_lines:
        st.caption(
            "Display formatting: whole lines that are valid JSON objects or arrays are expanded into fenced JSON blocks "
            "so chat-template tool schemas are easier to inspect. The log artifact itself is unchanged."
        )
        st.caption(f"Pretty-printed JSON lines: {left.label}={left_json_lines}, {right.label}={right_json_lines}")
    _render_text_comparison_sides(left_prompt_display, right_prompt_display, left.label, right.label, radio_key="sp_view")


def _render_text_comparison_sides(left_text: str, right_text: str, left_label: str, right_label: str, *, radio_key: str) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric(f"{left_label} length", f"{len(left_text):,} chars" if left_text else "-")
    c2.metric(f"{right_label} length", f"{len(right_text):,} chars" if right_text else "-")
    if left_text and right_text:
        c3.metric("Match", "✅ Yes" if left_text.strip() == right_text.strip() else "❌ No")

    view = st.radio("View", ["Side by Side", "Diff", left_label, right_label], horizontal=True, key=radio_key)
    if view == "Side by Side":
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown(f"**{left_label}**")
            _render_wrapped_text(left_text) if left_text else st.info("Not available.")
        with col_r:
            st.markdown(f"**{right_label}**")
            _render_wrapped_text(right_text) if right_text else st.info("Not available.")
    elif view == "Diff":
        if left_text and right_text:
            _render_side_by_side_diff(left_text, right_text, left_label=left_label, right_label=right_label)
        else:
            st.info("Both sides needed for diff.")
    elif view == left_label:
        _render_wrapped_text(left_text) if left_text else st.info("Not available.")
    else:
        _render_wrapped_text(right_text) if right_text else st.info("Not available.")


def _extract_first_user_message(data: dict[str, Any], *, agentic: bool) -> str:
    """Return the content of the first user message from a task log."""
    msgs = data.get("conversation", []) if agentic else _ori_messages(data)
    for m in msgs:
        if m.get("role") == "user":
            return str(m.get("content", ""))
    return ""


def _extract_first_user_message_for_side(data: dict[str, Any], side: DashboardSide) -> str:
    return _extract_first_user_message(data, agentic=(side.kind == "agentic"))


def _render_task_description_sides(left: DashboardSide, right: DashboardSide) -> None:
    st.header("Task Description")
    all_ids = _all_task_ids(left.index, right.index)
    if not all_ids:
        st.info("No tasks found.")
        return

    selected = st.selectbox("Select task", all_ids, key="taskdesc_task")
    if not selected:
        return

    left_desc = ""
    right_desc = ""
    if selected in left.index:
        left_desc = _extract_first_user_message_for_side(_load_json_cached(_latest_path(left.index[selected])), left)
    if selected in right.index:
        right_desc = _extract_first_user_message_for_side(_load_json_cached(_latest_path(right.index[selected])), right)

    if not left_desc and not right_desc:
        st.info("No task descriptions found.")
        return

    _render_text_comparison_sides(left_desc, right_desc, left.label, right.label, radio_key="td_view")
