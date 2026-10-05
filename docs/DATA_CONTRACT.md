# Data contract and boundaries

## Transactions

`TransactionColumns(bond, time, target, ...)` maps actual source names. The engine retains the original columns for base features and custom slices, then adds a canonical schema.

| Mapping | Meaning |
|---|---|
| `bond` | Stable security identity; not issuer. CSV identifiers are read as strings to retain leading zeros. |
| `time` | The time the prediction would have been made. Features only use quotes known by this boundary. |
| `target` | Numeric regression label. Use a level by default; set `TrainingConfig(target_mode="delta")` for an anchor-relative label. |
| `actual` | Optional explicit realized outcome for scoring. It takes precedence over the inferred target or target + anchor. Useful when the fitted delta contains an upstream adjustment. Missing actuals remain unscored; they are not filled from the target. |
| `id` | Optional unique transaction identity. Without it, input row position supplies a stable ID for that input order. |
| `quantity`, `prev_quantity` | Current/previous trade notional. `quantity_scale` converts both to the units used by large-trade thresholds; use actual currency notionals for 1MM slices. |
| `issuer`, `sector` | Optional transaction-time metadata. No later issuer label is backfilled into earlier issuer features. |
| `maturity_years` or `maturity_date` | Optional remaining maturity; a date is converted at prediction time. Map one representation. |
| `anchor`, `cpp` | Optional values observable before the prediction. Their discrepancy supports pre-trade anchor-quality slices. |
| `split` | Optional explicit Train/Validation/Test/Embargo labels; chronological order is validated. |

Missing or invalid target values are retained for coverage diagnosis and excluded from fitting/scoring. Invalid transaction identities/times fail early rather than silently dropping prediction requests. Additional metadata such as rating, trading venue, trade direction or a user-defined liquidity category stays available for arbitrary slices.

Base feature lists are explicit and ordered. Categorical vocabularies are fitted on Train only; unseen evaluation categories become missing. Numeric infinities become missing model inputs. The engine rejects direct inclusion of the mapped target and actual outcome, but cannot prove that arbitrary supplied base features are causal. Establish this upstream.

`TrainingConfig(category_order="appearance")` preserves first-observed training category order; the generic default is `"sorted"`. `apply_model_defaults=False` passes only supplied model parameters to LightGBM, allowing an existing parameter dictionary to retain its library defaults. The generic default is `True`, which adds the engine's training defaults before applying supplied overrides. Both switches are saved in the run manifest.

## Raw quotes

Both forms require bond identity, dealer identity and **known time**:

```python
# CONFIGURATION LOGIC: A single size applies to both wide quote sides when separate sizes are unavailable.
wide = QuoteColumns("bond", "known", "dealer", bid="bid", ask="ask", quantity="size")
# CONFIGURATION LOGIC: Separate side quantities and one-sided quotes are also supported.
wide_sizes = QuoteColumns("bond", "known", "dealer", bid="bid", ask="ask", bid_size="bid_size", ask_size="ask_size")
# CONFIGURATION LOGIC: Long input matches the original BondCliQ-style layout.
long = QuoteColumns("cusip", "quote_timestamp_ET", "firm", side="side", value="spread", quantity="quantity")
```

`original_timestamp` can retain an exchange/source timestamp, but never replaces known time for joins. If the only timestamp is a vendor event time, the user must establish its availability meaning; the engine cannot reconstruct missing publication latency.

For wide input, a genuinely missing side is **no side observation**. A present malformed value is an **incomplete observation**, which blocks fallback to the older valid side. Nulls are not inferred cancellations. Feeds with deletion, withdrawal, executable order-book depth or snapshot-replacement protocols need a source adapter before use; the engine does not claim to implement those protocols.

Exact events are `(bond, dealer, side, known time)`, without timestamp rounding. Each event retains distinct value and quantity candidates. Complete source-row duplicates are counted before generated IDs are added. Repeated identical candidates receive no extra price weight. A candidate median may not equal an actual posted quote; ranges and support make that visible.

Unknown, zero and malformed size are distinct diagnoses. Positive raw size equality supports within-event/pair comparisons; it does not establish that a dealer would execute a transaction of that size. `quote_quantity_scale` is independent of transaction `quantity_scale`; no multiplier is inferred from observed values.

## Time and units

Naive timestamps are interpreted in the configured timezone. Aware timestamps are converted to it. Ambiguous/nonexistent naive DST timestamps raise, and mixed naive/aware input must be resolved upstream. State does not carry overnight. `allow_exact=True` includes a quote with known time equal to prediction time; use `False` when sequencing is uncertain. Duplicate transaction timestamps remain separate records.

`target_scale` multiplies target, actual, anchor and CPP. `quote_scale` multiplies quote values. These must produce compatible units before quote-minus-anchor features are formed. `error_scale` converts prediction errors to the displayed `unit` only; it does not rescale features or repair a quote/target mismatch. Example: decimal spreads → bps uses `target_scale=10000, quote_scale=10000, error_scale=1, unit="bps"`.

For spreads, signed width is bid minus ask; for prices, it is ask minus bid. Negative width denotes crossing in both modes. An increase in price is not called spread widening. Price-to-spread/yield conversions, currencies, benchmark definitions and accrued-interest conventions are outside this engine.

The internal `spread`, `bcq_`, `_bps` and `_30m` names are retained from audited numerical kernels to make their lineage inspectable. `_bps` values use configured quote units; path `_30m` windows follow `lookback_min`. Defaults: 30-minute lookback/freshness, 1-minute synchronization, 60-minute history break. `clip_floor` is in normalized quote-value units, not automatically bps. Optional peer-rule diagnostics need a floor appropriate for price mode; they are not in the fixed default model families.

## Scope

Only bonds occurring in supplied transactions enter the quote universe; original and retained quote counts are reported. No three-month limit is imposed. Local-day event state, per-bond calculations, bounded state/movement query matrices and stage caches reduce repeated work. The implementation uses in-memory pandas, not a distributed/out-of-core service. Filter source files to a deliberate research period and provide enough RAM for the normalized input, feature frame and models.

An optional `run_research(..., quote_universe=frame)` accepts an explicit DataFrame with a canonical `cusip` column. This keeps the original research universe when eligible model targets are a narrower subset; normalization reports `universe_bonds`. Without it, supplied transactions determine the universe as before.

Historical trade counts use supplied transactions in `[t-30 days,t)`. Equal-time trades do not count each other. A separate history-coverage flag identifies truncated windows. Even a full 30-day window is not proof of a complete market feed. Short datasets can still be diagnosed, but need feasible training/evaluation dates or explicitly declared splits.

Set `PipelineConfig(priority_history_column="your_prior_count")` to use an existing upstream history count for priority cohorts. This explicit source overrides the trailing count even when missing, in which case affected cohorts are reported unsupported. With `None`, the generic trailing-count preference is unchanged. The configured field's historical window and feed completeness remain the caller's responsibility.

Set `PipelineConfig(priority_cpp_gap_column="your_gap_bps")` to supply the precomputed CPP–anchor discrepancy in **bps** for the strict `>5` and `>10` priority cohorts. The engine uses its absolute value without applying `target_scale` or `error_scale` again. Missing values stay unknown; a missing configured column makes the CPP cohorts unsupported, even when `cpp` and `anchor` exist. This avoids changing exact threshold boundaries by reconstructing and then subtracting the anchor. A spread/bps contract is required. The default `None` retains `abs(cpp-anchor) * error_scale`.
