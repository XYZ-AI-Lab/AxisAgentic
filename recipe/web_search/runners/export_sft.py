# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic.sft_export import SwiftAgentExportConfig
from recipe.web_search.sft_export import DEFAULT_ACCEPTED_STATUSES, WebSearchSFTExportConfig, export_web_search_run_to_sft


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export web-search canonical traces into derived SFT datasets.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing web-search-benchmark traces.")
    parser.add_argument("--output", default=None, help="Output JSONL path. Defaults to <run-dir>/sft_exports/swift_agent.jsonl.")
    parser.add_argument("--format", default="swift_agent", choices=["swift_agent"], help="Derived SFT format.")
    parser.add_argument("--max-traces", type=int, default=None, help="Maximum number of exported samples.")
    parser.add_argument(
        "--status",
        nargs="*",
        default=list(DEFAULT_ACCEPTED_STATUSES),
        help="Accepted trace statuses. Use 'all' to disable status filtering.",
    )
    parser.add_argument("--include-metadata", action="store_true", help="Include source metadata in each JSONL row.")
    parser.add_argument("--no-reasoning", action="store_true", help="Do not render reasoning_content as <think> blocks.")
    parser.add_argument("--strict", action="store_true", help="Fail on the first invalid trace instead of recording errors.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    statuses = tuple() if args.status == ["all"] else tuple(args.status)
    config = WebSearchSFTExportConfig(
        format=args.format,
        accepted_statuses=statuses,
        max_traces=args.max_traces,
        include_metadata=args.include_metadata,
        strict=args.strict,
        swift=SwiftAgentExportConfig(include_reasoning=not args.no_reasoning),
    )
    summary = export_web_search_run_to_sft(run_dir=Path(args.run_dir), output_path=args.output, config=config)
    print(
        json.dumps(
            {
                "output_path": summary.output_path,
                "manifest_path": summary.manifest_path,
                "total_trace_files": summary.total_trace_files,
                "exported": summary.exported,
                "skipped": summary.skipped,
                "failed": summary.failed,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
