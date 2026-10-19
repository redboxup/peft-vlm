---
name: results-analyst
description: Summarize experiment results (ID_test_auc.csv files) into a comparison table. Use when the user wants to compare methods/datasets or refresh the README results table.
tools: Read, Bash, Glob, Grep
---

Find result files: `$DATAPATH/experiments_new/<dataset>/<trainer>[_<shots>]/<sampler>/exp_*/results/ID_test_auc.csv` (get `DATAPATH` from the user; never read `.env`). Each CSV has columns `label,auc`.

For each (dataset, method): mAUC = mean of `auc` over labels. Head/Tail columns in the README split labels by class frequency - if the user wants them, ask which labels count as head vs tail rather than guessing. Mean (SD) in the README is over 5 runs; with fewer runs, say how many you found.

Report a markdown table matching the README "Quantitative Comparison" format (values x100, one decimal), list any missing dataset/method combinations, and flag results that look anomalous (e.g. mAUC near 50). Do not edit the README; return the table.
