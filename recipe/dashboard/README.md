# Benchmark dashboard

The Streamlit dashboard compares AxisAgentic web-search and WideSearch runs. It can read completed aggregate artifacts and inspect live task traces while a run is active.

## Install and launch

```bash
python -m pip install -e '.[dashboard]'
streamlit run recipe/dashboard/app.py --server.fileWatcherType none -- \
  --log-dir "${AXIS_LOG_DIR}"
```

`--log-dir` (also accepted as `--log-root`) can be repeated to expose multiple roots. If no root is provided, the app uses `AXIS_LOG_DIR`, falling back to `logs`.

You can preselect concrete run directories for each comparison side:

```bash
streamlit run recipe/dashboard/app.py --server.fileWatcherType none -- \
  --log-dir /logs/axis \
  --left-log-dir /logs/axis/web_search_infer/experiment_a \
  --right-log-dir /logs/axis/web_search_infer/experiment_b
```

On a remote host, forward Streamlit's default port `8501` or configure a different Streamlit port.

## Views

- experiment overview and accuracy;
- WideSearch metrics;
- timing and trace-length distributions;
- assistant-message and tool-call statistics;
- per-task traces;
- effective config, system-prompt, and task-description comparisons.

The recipe runners update compact dashboard artifacts incrementally, so many views remain useful before a long run finishes.

See [Evaluation and reproducibility](../../docs/evaluation.md) for run layouts and interpretation guidance.
