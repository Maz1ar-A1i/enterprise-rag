---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:1162
- loss:BinaryCrossEntropyLoss
base_model: BAAI/bge-reranker-base
pipeline_tag: text-ranking
library_name: sentence-transformers
metrics:
- accuracy
- accuracy_threshold
- f1
- f1_threshold
- precision
- recall
- average_precision
model-index:
- name: CrossEncoder based on BAAI/bge-reranker-base
  results:
  - task:
      type: cross-encoder-binary-classification
      name: Cross Encoder Binary Classification
    dataset:
      name: eval reranker
      type: eval_reranker
    metrics:
    - type: accuracy
      value: 0.95
      name: Accuracy
    - type: accuracy_threshold
      value: 0.17563921213150024
      name: Accuracy Threshold
    - type: f1
      value: 0.9490445859872613
      name: F1
    - type: f1_threshold
      value: 0.17563921213150024
      name: F1 Threshold
    - type: precision
      value: 0.9675324675324676
      name: Precision
    - type: recall
      value: 0.93125
      name: Recall
    - type: average_precision
      value: 0.9866607461615036
      name: Average Precision
---

# CrossEncoder based on BAAI/bge-reranker-base

This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model finetuned from [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base) using the [sentence-transformers](https://www.SBERT.net) library. It computes scores for pairs of texts, which can be used for text reranking and semantic search.

## Model Details

### Model Description
- **Model Type:** Cross Encoder
- **Base model:** [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base) <!-- at revision 2cfc18c9415c912f9d8155881c133215df768a70 -->
- **Maximum Sequence Length:** 512 tokens
- **Number of Output Labels:** 1 label
- **Supported Modality:** Text
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Documentation:** [Cross Encoder Documentation](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Cross Encoders on Hugging Face](https://huggingface.co/models?library=sentence-transformers&other=cross-encoder)

### Full Model Architecture

```
CrossEncoder(
  (0): Transformer({'transformer_task': 'sequence-classification', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'logits'}}, 'module_output_name': 'scores', 'architecture': 'XLMRobertaForSequenceClassification'})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import CrossEncoder

# Download from the 🤗 Hub
model = CrossEncoder("cross_encoder_model_id")
# Get scores for pairs of inputs
pairs = [
    ['Why did Northpeak Search see their Dedicated Burst Usage billed higher than expected, and how do we prevent retry attempts from double-counting burst GPU-seconds now?', 'Fix billing double-counting on Dedicated burst retries (idempotency + reconciliation)\n\nProblem: we are double-counting burst usage when the same customer request is retried (client retries, gateway retries after 5xx/timeout, and some 429 backoff paths). In beta, this showed up as duplicate burst usage line items for the same request window and was flagged in SUP-28488 and during reconciliation sampling. Root cause: the burst metering emitter currently treats each attempt as billable because we do not have a stable request-scoped idempotency key carried end-to-end into the metering aggregation stage; request_id/trace_id are not consistently preserved across retry code paths. Fix: introduce a request-scoped idempotency key that is generated at the gateway if missing, propagated to the data plane and metering emitter, and used to dedupe burst billable events before exporting to billing. Also add a reconciliation guard that detects duplicates that slip through (e.g., telemetry lag / partial outages) and blocks export with an alert. Details: 1) API/gateway: accept and forward X-Redwood-Idempotency-Key; if absent, generate a ULID per incoming request and reuse it across internal retry attempts. 2) Metering: include idempotency_key on burst usage events; dedupe in the burst_usage_aggregator on (tenant_id, region, hw_class, pool_id, idempotency_key) with a bounded TTL and hashed storage to avoid high-cardinality raw keys. 3) Billing export: export a deduped_usage_count and dropped_duplicate_attempts metric; add a hard reconciliation check that fails the export batch if duplicate keys exceed a small threshold (defaults to 0.01%) and pages on-call for billing pipeline'],
    ['What percent reduction in monthly model or token spend was the founder of a small developer tools micro-SaaS aiming for while keeping acceptable latency?', 'Pilot pricing guidance (templates and assumptions)\n------------------------------------------------\n- Objective: remove pricing ambiguity for pilots while providing pathways to predictable committed pricing.\n- Pilot terms (recommended default): 30-day pilot, up to 5M tokens, hosted environment with limited IP allowlist, audit logging enabled. Fixed pilot fee: $4,500 (covers 5M tokens and entitlement to a single Dedicated sizing recommendation). Overage: $0.0008 per additional token.\n- Pilot-to-commit conversion discounts (illustrative): commit to Dedicated (6–12 month) and receive a 30–45% discount on per-token hosted-equivalent cost plus waived onboarding credits.\n\nPricing scenario table (plain-text)\n----------------------------------\n| Scenario | Pilot (30d) | Equivalent Monthly at 1M tokens/day | Notes |\n| Pilot fixed fee | $4,500 | — | includes Console dashboards, basic evals, single custom prompt set |\n| Host per-token (baseline) | $0.0009 | $27,000 | 1M tokens/day ~ 30M/month |\n| Dedicated estimate (committed) | N/A | $16,500 | ~40% cost reduction vs hosted baseline with SLA |\n\nRepresentative anonymized case studies\n--------------------------------------\n1) FinTech fraud-review pilot (Mid-market, US).\n - Ask: low-latency scoring for transaction triage, data residency controls, audit logs for model inputs.\n - Setup: 14-day Hosted pilot, IP-restricted endpoints, prompt-level evals.\n - Outcomes: Achieved p99 latency 110ms, reduced triage manual review by 48%, and converted to Dedicated after pilot; estimated savings 36% per inference vs prior hosted provider.'],
    ['In the cost-aware model routing proposal, what telemetry fields were suggested to emit so the team can analyze routing decisions and tradeoffs between spend and quality?', "eng-platform\n\nanna: Quick sync — thinking about introducing a cost-aware routing layer that prefers cheaper model variants for lower customer tiers and by region. It should also have automatic fallback to higher-quality variants if we detect quality regressions. High-level idea, not details yet.\n\nmike: +1 on the idea. Two engineering concerns: 1) model compatibility (tokenization, output format) 2) how we estimate token cost before routing. We need cheap per-request cost estimate in the routing path. :thinking:\n\njess: For telemetry we should emit: route_decision, estimated_token_cost, chosen_variant, quality_score, fallback_trigger, and token_counts. That lets us correlate cost savings vs quality regressions.\n\noliver: UX note — console needs a simple toggle per-customer and a rollout slider. Folks should be able to opt-out / pin to expensive variant. Also show expected savings and recent fallback rate in the UI.\n\nsam: Quotas/limits angle: if we route free-tier customers to cheaper variant we should also enforce a per-customer token_budget so they can't blow past cost forecasts. If budget exceeded, either degrade to a lower-rate limit or surface upsell.\n\nmike: Implementation sketch: routing policy lives in the control plane. At request time we compute a cheap estimate: tokens_est = ceil(prompt_len / avg_chunk) + ceil(max_output_len / avg_chunk) then cost_est = tokens_est * variant_unit_cost. If cost_est 0.5% over last 1k requests OR average quality_score canary (1%) scheduled at 2024-07-15T18:00Z (reacts :rocket:)"],
    ['In a case where server-sent event streaming responses sometimes lose the very last space or period only when traffic goes through an HTTP/2 edge proxy with stream preemption, what mitigation was proposed to stop the final bytes from being dropped?', "SSE stream cuts final whitespace and punctuation after HTTP/2 priority preemption behind customer's edge proxy\n\nIssue summary:\nCustomer reports that SSE streams delivered to their web clients occasionally arrive with the final characters (trailing whitespace and punctuation) missing. The content is syntactically complete but the last sentence sometimes loses a trailing space or final period, which breaks downstream client-side heuristics that expect a newline/terminator.\n\nImpact:\n- Affects BrightDocs' production readers in us-east (enterprise account).\n- Visible as malformed final paragraphs in ~0.2% of long streaming responses for a subset of sessions; higher during bursts.\n- Customer considers this a data-integrity issue for document export pipelines and flagged it as high priority.\n\nEnvironment:\n- Customer located behind an enterprise edge proxy that uses HTTP/2 with priority and stream preemption enabled.\n- Requests from browser -> edge proxy (h2) -> Redwood API Gateway (h1/h2) -> serving runtime.\n- Observed on models: gpt-large-v2, redwood-compat-1.0.\n\nSteps to reproduce (customer-provided):\n1) Client opens SSE connection to /v1/stream with long completion (>512 tokens) over HTTP/2 through their edge proxy.\n2) Under simulated burst (20 concurrent streams from same client IP) the client sees the SSE session finish normally but the last token is missing trailing whitespace/punctuation in ~1/500 sessions.\n3) Replaying the same request bypassing the edge proxy (direct to Redwood) does not reproduce."],
    ['What was the high-percentile latency concern reported after a smoke test of about 50 concurrent streaming chat sessions for Satellite Grove?', 'Satellite Grove\n\nLead source: self-serve signup (product-led). Team: CTO (Aisha Kaur), Eng Lead (Mateo Lopez), PM (Rina Shah). Early discovery call 2026-03-02 via Fireflies ff-20260302-0934. Summary: building an in-app conversational assistant for their B2B workflow SaaS; must feel instant (streaming responses) and run on public cloud.\n\nKey requirements:\n- Streaming websocket support + chunked tokens (UX requires UI tokens arrive in <50ms increments).\n- p50 target ~40-80ms, p95 <120ms, p99 <200ms for short responses (avg 60 tokens).\n- Concurrent active users at launch ~100-200, expected steady-state 20-50.\n- Cost-sensitive: want guidance on batching vs streaming tradeoffs and token/unit price impact.\n- Model preferences: prefer smaller fast open models for latency (we mentioned quantized Llama-family / Mistral-lite), but open to Redwood verified models for throughput.\n- Security: SOC2 on roadmap, need SSO (SAML) + audit logging for partner beta.\n\nPOC/workflow discussed:\n- They used free credits to wire up a quick websocket chat to Redwood hosted API. Observed good tail latencies but worried about unit cost when scaling. Did a smoke test with ~50 concurrent short chats (avg reply length 60 tokens) — saw 120ms median, 250ms p99 in one run.\n- Interested in Redwood Optimize suggestions (batching hints, prefix caching) to reduce cost without hurting streaming UX.\n\nQuotes / call notes (shorthand):\n- "Streaming must feel instant — if agents lag, users drop off." — Aisha (CTO).\n- "We need a clear cost model for streaming usage; tickets are tight for first 6 months." — Mateo (Eng Lead).'],
]
scores = model.predict(pairs)
print(scores)
# [1.     0.0025 1.     1.     1.    ]

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'Why did Northpeak Search see their Dedicated Burst Usage billed higher than expected, and how do we prevent retry attempts from double-counting burst GPU-seconds now?',
    [
        'Fix billing double-counting on Dedicated burst retries (idempotency + reconciliation)\n\nProblem: we are double-counting burst usage when the same customer request is retried (client retries, gateway retries after 5xx/timeout, and some 429 backoff paths). In beta, this showed up as duplicate burst usage line items for the same request window and was flagged in SUP-28488 and during reconciliation sampling. Root cause: the burst metering emitter currently treats each attempt as billable because we do not have a stable request-scoped idempotency key carried end-to-end into the metering aggregation stage; request_id/trace_id are not consistently preserved across retry code paths. Fix: introduce a request-scoped idempotency key that is generated at the gateway if missing, propagated to the data plane and metering emitter, and used to dedupe burst billable events before exporting to billing. Also add a reconciliation guard that detects duplicates that slip through (e.g., telemetry lag / partial outages) and blocks export with an alert. Details: 1) API/gateway: accept and forward X-Redwood-Idempotency-Key; if absent, generate a ULID per incoming request and reuse it across internal retry attempts. 2) Metering: include idempotency_key on burst usage events; dedupe in the burst_usage_aggregator on (tenant_id, region, hw_class, pool_id, idempotency_key) with a bounded TTL and hashed storage to avoid high-cardinality raw keys. 3) Billing export: export a deduped_usage_count and dropped_duplicate_attempts metric; add a hard reconciliation check that fails the export batch if duplicate keys exceed a small threshold (defaults to 0.01%) and pages on-call for billing pipeline',
        'Pilot pricing guidance (templates and assumptions)\n------------------------------------------------\n- Objective: remove pricing ambiguity for pilots while providing pathways to predictable committed pricing.\n- Pilot terms (recommended default): 30-day pilot, up to 5M tokens, hosted environment with limited IP allowlist, audit logging enabled. Fixed pilot fee: $4,500 (covers 5M tokens and entitlement to a single Dedicated sizing recommendation). Overage: $0.0008 per additional token.\n- Pilot-to-commit conversion discounts (illustrative): commit to Dedicated (6–12 month) and receive a 30–45% discount on per-token hosted-equivalent cost plus waived onboarding credits.\n\nPricing scenario table (plain-text)\n----------------------------------\n| Scenario | Pilot (30d) | Equivalent Monthly at 1M tokens/day | Notes |\n| Pilot fixed fee | $4,500 | — | includes Console dashboards, basic evals, single custom prompt set |\n| Host per-token (baseline) | $0.0009 | $27,000 | 1M tokens/day ~ 30M/month |\n| Dedicated estimate (committed) | N/A | $16,500 | ~40% cost reduction vs hosted baseline with SLA |\n\nRepresentative anonymized case studies\n--------------------------------------\n1) FinTech fraud-review pilot (Mid-market, US).\n - Ask: low-latency scoring for transaction triage, data residency controls, audit logs for model inputs.\n - Setup: 14-day Hosted pilot, IP-restricted endpoints, prompt-level evals.\n - Outcomes: Achieved p99 latency 110ms, reduced triage manual review by 48%, and converted to Dedicated after pilot; estimated savings 36% per inference vs prior hosted provider.',
        "eng-platform\n\nanna: Quick sync — thinking about introducing a cost-aware routing layer that prefers cheaper model variants for lower customer tiers and by region. It should also have automatic fallback to higher-quality variants if we detect quality regressions. High-level idea, not details yet.\n\nmike: +1 on the idea. Two engineering concerns: 1) model compatibility (tokenization, output format) 2) how we estimate token cost before routing. We need cheap per-request cost estimate in the routing path. :thinking:\n\njess: For telemetry we should emit: route_decision, estimated_token_cost, chosen_variant, quality_score, fallback_trigger, and token_counts. That lets us correlate cost savings vs quality regressions.\n\noliver: UX note — console needs a simple toggle per-customer and a rollout slider. Folks should be able to opt-out / pin to expensive variant. Also show expected savings and recent fallback rate in the UI.\n\nsam: Quotas/limits angle: if we route free-tier customers to cheaper variant we should also enforce a per-customer token_budget so they can't blow past cost forecasts. If budget exceeded, either degrade to a lower-rate limit or surface upsell.\n\nmike: Implementation sketch: routing policy lives in the control plane. At request time we compute a cheap estimate: tokens_est = ceil(prompt_len / avg_chunk) + ceil(max_output_len / avg_chunk) then cost_est = tokens_est * variant_unit_cost. If cost_est 0.5% over last 1k requests OR average quality_score canary (1%) scheduled at 2024-07-15T18:00Z (reacts :rocket:)",
        "SSE stream cuts final whitespace and punctuation after HTTP/2 priority preemption behind customer's edge proxy\n\nIssue summary:\nCustomer reports that SSE streams delivered to their web clients occasionally arrive with the final characters (trailing whitespace and punctuation) missing. The content is syntactically complete but the last sentence sometimes loses a trailing space or final period, which breaks downstream client-side heuristics that expect a newline/terminator.\n\nImpact:\n- Affects BrightDocs' production readers in us-east (enterprise account).\n- Visible as malformed final paragraphs in ~0.2% of long streaming responses for a subset of sessions; higher during bursts.\n- Customer considers this a data-integrity issue for document export pipelines and flagged it as high priority.\n\nEnvironment:\n- Customer located behind an enterprise edge proxy that uses HTTP/2 with priority and stream preemption enabled.\n- Requests from browser -> edge proxy (h2) -> Redwood API Gateway (h1/h2) -> serving runtime.\n- Observed on models: gpt-large-v2, redwood-compat-1.0.\n\nSteps to reproduce (customer-provided):\n1) Client opens SSE connection to /v1/stream with long completion (>512 tokens) over HTTP/2 through their edge proxy.\n2) Under simulated burst (20 concurrent streams from same client IP) the client sees the SSE session finish normally but the last token is missing trailing whitespace/punctuation in ~1/500 sessions.\n3) Replaying the same request bypassing the edge proxy (direct to Redwood) does not reproduce.",
        'Satellite Grove\n\nLead source: self-serve signup (product-led). Team: CTO (Aisha Kaur), Eng Lead (Mateo Lopez), PM (Rina Shah). Early discovery call 2026-03-02 via Fireflies ff-20260302-0934. Summary: building an in-app conversational assistant for their B2B workflow SaaS; must feel instant (streaming responses) and run on public cloud.\n\nKey requirements:\n- Streaming websocket support + chunked tokens (UX requires UI tokens arrive in <50ms increments).\n- p50 target ~40-80ms, p95 <120ms, p99 <200ms for short responses (avg 60 tokens).\n- Concurrent active users at launch ~100-200, expected steady-state 20-50.\n- Cost-sensitive: want guidance on batching vs streaming tradeoffs and token/unit price impact.\n- Model preferences: prefer smaller fast open models for latency (we mentioned quantized Llama-family / Mistral-lite), but open to Redwood verified models for throughput.\n- Security: SOC2 on roadmap, need SSO (SAML) + audit logging for partner beta.\n\nPOC/workflow discussed:\n- They used free credits to wire up a quick websocket chat to Redwood hosted API. Observed good tail latencies but worried about unit cost when scaling. Did a smoke test with ~50 concurrent short chats (avg reply length 60 tokens) — saw 120ms median, 250ms p99 in one run.\n- Interested in Redwood Optimize suggestions (batching hints, prefix caching) to reduce cost without hurting streaming UX.\n\nQuotes / call notes (shorthand):\n- "Streaming must feel instant — if agents lag, users drop off." — Aisha (CTO).\n- "We need a clear cost model for streaming usage; tickets are tight for first 6 months." — Mateo (Eng Lead).',
    ]
)
# [{'corpus_id': ..., 'score': ...}, {'corpus_id': ..., 'score': ...}, ...]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Cross Encoder Binary Classification

* Dataset: `eval_reranker`
* Evaluated with [<code>CEBinaryClassificationEvaluator</code>](https://sbert.net/docs/package_reference/cross_encoder/evaluation.html#sentence_transformers.cross_encoder.evaluation.CEBinaryClassificationEvaluator)

| Metric                | Value      |
|:----------------------|:-----------|
| accuracy              | 0.95       |
| accuracy_threshold    | 0.1756     |
| f1                    | 0.949      |
| f1_threshold          | 0.1756     |
| precision             | 0.9675     |
| recall                | 0.9313     |
| **average_precision** | **0.9867** |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 1,162 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 100 samples:
  |          | sentence_0                                                                         | sentence_1                                                                           | label                                                          |
  |:---------|:-----------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type     | string                                                                             | string                                                                               | float                                                          |
  | modality | text                                                                               | text                                                                                 |                                                                |
  | details  | <ul><li>min: 20 tokens</li><li>mean: 47.87 tokens</li><li>max: 96 tokens</li></ul> | <ul><li>min: 71 tokens</li><li>mean: 404.09 tokens</li><li>max: 512 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.55</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                             | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | label            |
  |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>Why did Northpeak Search see their Dedicated Burst Usage billed higher than expected, and how do we prevent retry attempts from double-counting burst GPU-seconds now?</code>    | <code>Fix billing double-counting on Dedicated burst retries (idempotency + reconciliation)<br><br>Problem: we are double-counting burst usage when the same customer request is retried (client retries, gateway retries after 5xx/timeout, and some 429 backoff paths). In beta, this showed up as duplicate burst usage line items for the same request window and was flagged in SUP-28488 and during reconciliation sampling. Root cause: the burst metering emitter currently treats each attempt as billable because we do not have a stable request-scoped idempotency key carried end-to-end into the metering aggregation stage; request_id/trace_id are not consistently preserved across retry code paths. Fix: introduce a request-scoped idempotency key that is generated at the gateway if missing, propagated to the data plane and metering emitter, and used to dedupe burst billable events before exporting to billing. Also add a reconciliation guard that detects duplicates that slip through (e.g., telemetry lag / partia...</code>                                            | <code>1.0</code> |
  | <code>What percent reduction in monthly model or token spend was the founder of a small developer tools micro-SaaS aiming for while keeping acceptable latency?</code>                 | <code>Pilot pricing guidance (templates and assumptions)<br>------------------------------------------------<br>- Objective: remove pricing ambiguity for pilots while providing pathways to predictable committed pricing.<br>- Pilot terms (recommended default): 30-day pilot, up to 5M tokens, hosted environment with limited IP allowlist, audit logging enabled. Fixed pilot fee: $4,500 (covers 5M tokens and entitlement to a single Dedicated sizing recommendation). Overage: $0.0008 per additional token.<br>- Pilot-to-commit conversion discounts (illustrative): commit to Dedicated (6–12 month) and receive a 30–45% discount on per-token hosted-equivalent cost plus waived onboarding credits.<br><br>Pricing scenario table (plain-text)<br>----------------------------------<br>\| Scenario \| Pilot (30d) \| Equivalent Monthly at 1M tokens/day \| Notes \|<br>\| Pilot fixed fee \| $4,500 \| — \| includes Console dashboards, basic evals, single custom prompt set \|<br>\| Host per-token (baseline) \| $0.0009 \| $27,000 \| 1M tokens/day ~ 30M/month \|<br>\|...</code> | <code>0.0</code> |
  | <code>In the cost-aware model routing proposal, what telemetry fields were suggested to emit so the team can analyze routing decisions and tradeoffs between spend and quality?</code> | <code>eng-platform<br><br>anna: Quick sync — thinking about introducing a cost-aware routing layer that prefers cheaper model variants for lower customer tiers and by region. It should also have automatic fallback to higher-quality variants if we detect quality regressions. High-level idea, not details yet.<br><br>mike: +1 on the idea. Two engineering concerns: 1) model compatibility (tokenization, output format) 2) how we estimate token cost before routing. We need cheap per-request cost estimate in the routing path. :thinking:<br><br>jess: For telemetry we should emit: route_decision, estimated_token_cost, chosen_variant, quality_score, fallback_trigger, and token_counts. That lets us correlate cost savings vs quality regressions.<br><br>oliver: UX note — console needs a simple toggle per-customer and a rollout slider. Folks should be able to opt-out / pin to expensive variant. Also show expected savings and recent fallback rate in the UI.<br><br>sam: Quotas/limits angle: if we route free-tier customers to cheaper vari...</code>                    | <code>1.0</code> |
* Loss: [<code>BinaryCrossEntropyLoss</code>](https://sbert.net/docs/package_reference/cross_encoder/losses.html#binarycrossentropyloss) with these parameters:
  ```json
  {
      "activation_fn": "torch.nn.modules.linear.Identity",
      "pos_weight": null
  }
  ```

### Training Hyperparameters

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 8
- `num_train_epochs`: 3
- `max_steps`: -1
- `learning_rate`: 5e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: False
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: None
- `trackio_bucket_id`: None
- `trackio_static_space_id`: None
- `per_device_eval_batch_size`: 8
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `dataloader_multiprocessing_context`: None
- `dataloader_in_order`: True
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_static_graph`: None
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: None
- `fsdp_config`: None
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}
- `warmup_ratio`: None

</details>

### Training Logs
| Epoch  | Step | eval_reranker_average_precision |
|:------:|:----:|:-------------------------------:|
| 0.6849 | 100  | 0.9818                          |
| 1.0    | 146  | 0.9850                          |
| 1.3699 | 200  | 0.9848                          |
| 2.0    | 292  | 0.9861                          |
| 2.0548 | 300  | 0.9862                          |
| 2.7397 | 400  | 0.9862                          |
| 3.0    | 438  | 0.9867                          |


### Training Time
- **Training**: 8.1 minutes

### Framework Versions
- Python: 3.13.15
- Sentence Transformers: 5.7.0
- Transformers: 5.16.1
- PyTorch: 2.11.0+cu128
- Accelerate: 1.14.0
- Datasets: 4.8.5
- Tokenizers: 0.23.1

## Additional Resources

- [Training and Finetuning Reranker Models with Sentence Transformers](https://huggingface.co/blog/train-reranker): the end-to-end guide for training or finetuning Cross Encoder (reranker) models.
- [Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/multimodal-sentence-transformers): use text, image, audio, and video reranker models through the same API.
- [Training and Finetuning Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/train-multimodal-sentence-transformers): training multimodal Cross Encoders.

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->