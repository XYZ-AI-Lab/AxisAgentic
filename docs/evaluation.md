<p align="right">
  <a href="../README.md">Home</a> ·
  <strong>English</strong> ·
  <a href="evaluation.zh-CN.md">简体中文</a>
</p>

# Evaluation and reproducibility

AxisAgentic records the effective run configuration and task-level evidence needed to inspect, resume, rejudge, and compare agent experiments. The current public evaluation recipes focus on search-agent benchmarks.

## Supported benchmark recipes

| Benchmark | Entry config | Primary evaluation |
| --- | --- | --- |
| BrowseComp | [`configs/browsecomp.yaml`](../configs/browsecomp.yaml) | exact/LLM verification |
| BrowseComp-ZH | [`configs/browsecompzh.yaml`](../configs/browsecompzh.yaml) | exact/LLM verification |
| DeepSearchQA | [`configs/deepsearchqa.yaml`](../configs/deepsearchqa.yaml) | LLM verification and macro F1 pass |
| GAIA | [`configs/gaia.yaml`](../configs/gaia.yaml) | benchmark answer verification |
| Humanity's Last Exam | [`configs/hle.yaml`](../configs/hle.yaml) | exact/LLM verification |
| LiveBrowseComp | [`configs/livebrowsecomp.yaml`](../configs/livebrowsecomp.yaml) | repeated-run judging and aggregation |
| WideSearch | [`configs/widesearch.yaml`](../configs/widesearch.yaml) | row- and item-level precision/recall/F1 |

The [web-search recipe](../recipe/web_search/README.md) runs the first six benchmark families. [WideSearch](../recipe/wide_search/README.md) has a separate tabular answer and judge pipeline.

## Reproducible run records

Recipe runs write an input config and an effective config after environment and path resolution. Depending on the recipe, a run also contains:

- append-only task traces and per-attempt metadata;
- benchmark inputs/predictions and evaluation sidecars;
- token, timing, tool-call, and assistant-message summaries;
- incremental and final aggregate metrics;
- compact artifacts consumed by the dashboard.

Model request payloads and judge request payloads are opt-in. They are not necessary for standard trace inspection and can contain sensitive or very large content.

Use `--resume` to reuse completed tasks after an interrupted run. The web-search runner protects finalized output directories by default; `--force-resume-finalized-run` is available only for deliberate rewrites.

## Dashboard

Install the dashboard extra and launch Streamlit against one or more log roots:

```bash
python -m pip install -e '.[dashboard]'
streamlit run recipe/dashboard/app.py --server.fileWatcherType none -- \
  --log-dir "${AXIS_LOG_DIR}"
```

The dashboard compares experiments, accuracy, WideSearch metrics, timing, trace distributions, assistant messages, tool calls, task details, effective configuration, and prompts. See the [dashboard README](../recipe/dashboard/README.md).

## Interpreting the reported benchmark figure

The XYZ-Aquila technical report evaluates the system on public agentic search benchmarks while withholding external benchmarks from routine optimization decisions. See the technical report for the complete comparison figures and evaluation protocol.

Some baselines are taken from heterogeneous public reports. Their harnesses, web access, tools, judges, and evaluation dates can differ. The figure therefore supports benchmark-level comparison, not a fully controlled universal ranking.

Additional limitations from the report should be kept in mind:

- the current study does not provide a full causal decomposition of every intervention;
- adaptive overfitting can remain possible despite evaluator isolation;
- live-web benchmarks vary over time;
- end-to-end compute, cost, and latency are not yet reported uniformly;
- the screened answer-conditioned RL proposal was not trained through the final acceptance gate;
- empirical system validation currently focuses on Deep Search.

For the full protocol and analysis, read [AI4AI at Scale: A Full-Pipeline System for Enhancing LLM Agentic Capabilities](https://xyz-lab.ai/blogs/ai4ai-at-scale/).
