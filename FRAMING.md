## Decision

Rawaj Pay should flag suspicious transfers for review rather than automatically block them. The threshold should balance fraud detection with the cost of incorrectly flagging legitimate transfers.

## Metric

Use expected cost as the main decision metric because false positives and false negatives have different business costs. Track precision and recall as supporting metrics to understand the trade-off.

## Baseline

Start with a simple rule-based baseline using a fixed score threshold. Compare its precision, recall, and expected cost with more advanced approaches before adopting a more complex model.

## Classical ML or LLM

Start with classical machine learning on structured transfer data because it is measurable, reproducible, and suitable for tabular risk scoring. An LLM may help explain flagged cases, but it should not replace the risk model or make unsupported decisions.
