# Shared recipe utilities

`recipe.common` contains implementation shared by the public recipes. It is a support package rather than a standalone CLI.

The package provides:

- artifact naming and atomic result updates;
- boxed-answer and evaluation-result helpers;
- retry and timing configuration shared by search/scrape/model calls;
- live and finalized evaluation summaries;
- dashboard artifact generation;
- trace distributions, assistant-message statistics, and trace references;
- resume guards for finalized runs.

User-facing commands are documented in the [web-search](../web_search/README.md), [WideSearch](../wide_search/README.md), and [dashboard](../dashboard/README.md) READMEs.
