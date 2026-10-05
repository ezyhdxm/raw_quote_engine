# Data contract and boundaries

## Transactions

`TransactionColumns(bond, time, target, ...)` maps actual source names. The engine retains the original columns for base features and custom slices, then adds a canonical schema.

| Mapping | Meaning |
|---|---|
| `bond` | Stable security identity; not issuer. CSV identifiers are read as strings to retain leading zeros. |
| `time` | The time the prediction would have been made. Features only use quotes known by this boundary. |
| `target` | Numeric regression label. Use a level by default; set `TrainingConfig(target_mode="delta")` for an anchor-relative label. |
| `actual` | Optional explicit realized outcome for scoring. It takes precedence over the inferred target or target + anchor - rollover adjustment. Useful when the fitted delta contains an upstream adjustment. Missing actuals remain unscored; they are not filled from the target. |
| `id` | Optional unique transaction identity. Without it, input row position supplies a stable ID for that input order. |
| `quantity`, `prev_quantity` | Current/previous trade notional. `quantity_scale` converts both to the units used by large-trade thresholds; use actual currency notionals for 1MM slices. |
| `issuer`, `sector` | Optional transaction-time metadata. No later issuer label is backfilled into earlier issuer features. |
| `maturity_years` or `maturity_date` | Optional remaining maturity; a date is converted at prediction time. Map one representation. |
| `anchor`, `cpp` | Optional user-prepared values observable before prediction. Proxy discrepancy uses `abs(cpp - (anchor - adjustment))`; the engine never constructs a proxy. |
| `rollover_adjustment` | Optional prediction-time scalar for a target defined as `actual - anchor + adjustment`; delta predictions reconstruct as `prediction + anchor - adjustment`. No mapping means zero adjustment; a mapped missing/nonfinite value rejects the run. |
| `split` | Optional explicit Train/Validation/Test/Embargo labels; chronological order is validated. |

Missing or invalid target values are retained for coverage diagnosis and excluded from fitting/scoring. Invalid transaction identities/times fail early rather than silently dropping prediction requests. Additional metadata such as rating, trading venue, trade direction or a user-defined liquidity category stays available for arbitrary slices.

Base feature lists are explicit and ordered; `base_cat_features` is their explicit categorical subset. Other baseline fields are numeric. Categorical vocabularies are fitted on Train only; unseen evaluation categories become missing. Numeric infinities become missing model inputs. The engine rejects direct inclusion of the mapped target and actual outcome, but cannot prove that arbitrary supplied base features are causal. Establish this upstream.

`TrainingConfig(category_order="appearance")` preserves first-observed training category order; the generic default is `"sorted"`. `apply_model_defaults=False` passes only supplied model parameters to LightGBM, allowing an existing parameter dictionary to retain its library defaults. The generic default is also `False`; `True` explicitly adds engine training defaults before applying supplied overrides. Both switches are saved in the run manifest.

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

Naive transaction timestamps use `timezone`. Naive quote timestamps use `quote_timezone` when set, otherwise `timezone`. Aware timestamps are converted to the state timezone without changing their instants. Ambiguous/nonexistent naive DST timestamps raise, and mixed naive/aware input must be resolved upstream. State does not carry overnight. `allow_exact=True` includes a quote with known time equal to prediction time; use `False` when sequencing is uncertain. Duplicate transaction timestamps remain separate records.

`target_scale` multiplies target, actual, anchor, the optional proxy (`cpp`) and the optional rollover adjustment. `quote_scale` multiplies quote values. These must produce compatible units before quote-minus-anchor features are formed. `error_scale` converts prediction errors to the displayed `unit` only; it does not rescale features or repair a quote/target mismatch. Example: decimal spreads → bps uses `target_scale=10000, quote_scale=10000, error_scale=1, unit="bps"`.

Only spread-valued input is supported: `value_kind` must be `"spread"`, and price input is rejected. Signed width is bid minus ask; negative width denotes crossing. Price-to-spread/yield conversions, currencies, benchmark definitions and accrued-interest conventions belong upstream. The optional rollover scalar reconstructs the transaction baseline; it does not align historical quote benchmarks across time.

The internal `spread`, `bcq_`, `_bps` and `_30m` names are retained from audited numerical kernels to make their lineage inspectable. `_bps` values use configured quote units; path `_30m` windows follow `lookback_min`. Defaults: 30-minute lookback/freshness, 1-minute synchronization, 60-minute history break. `clip_floor` is in normalized quote-value units, not automatically bps. Set this floor in the normalized spread units; peer-rule diagnostics are not in the fixed default model families.

## Scope

Only bonds occurring in supplied transactions enter the quote universe; original and retained quote counts are reported. No three-month limit is imposed. Local-day event state, per-bond calculations, bounded state/movement query matrices and stage caches reduce repeated work. The implementation uses in-memory pandas, not a distributed/out-of-core service. Filter source files to a deliberate research period and provide enough RAM for the normalized input, feature frame and models.

An optional `run_research(..., quote_universe=frame)` accepts an explicit DataFrame with a canonical `cusip` column. This keeps the original research universe when eligible model targets are a narrower subset; normalization reports `universe_bonds`. Without it, supplied transactions determine the universe as before.

Historical trade counts use supplied transactions in `[t-30 days,t)`. Equal-time trades do not count each other. A separate history-coverage flag identifies truncated windows. Even a full 30-day window is not proof of a complete market feed. Short datasets can still be diagnosed, but need feasible training/evaluation dates or explicitly declared splits.

Set `PipelineConfig(priority_history_column="your_prior_count")` to use an existing upstream history count for priority cohorts. This explicit source overrides the trailing count even when missing, in which case affected cohorts are reported unsupported. With `None`, the generic trailing-count preference is unchanged. The configured field's historical window and feed completeness remain the caller's responsibility.

Set `PipelineConfig(priority_cpp_gap_column="your_gap_bps")` to supply the precomputed CPP–anchor discrepancy in **bps** for the strict `>5` and `>10` priority cohorts. The engine uses its absolute value without applying `target_scale` or `error_scale` again. Missing values stay unknown; a missing configured column makes the CPP cohorts unsupported, even when `cpp` and `anchor` exist. This avoids changing exact threshold boundaries by reconstructing and then subtracting the anchor. A spread/bps contract is required. The default `None` uses `abs(cpp - (anchor - adjustment)) * error_scale`, with zero adjustment only when no adjustment column is mapped. A mapped missing/nonfinite rollover value rejects the run; missing proxy values remain unknown.

## Ownership and interactive configuration

`research_form` takes two prepared DataFrames, `BASE_FEATURES`, `BASE_CAT_FEATURES` and `LGB_PARAMS`. Configuration objects can prefill source mappings and research settings; the form remains editable. Validate performs structural preflight and optional fold-date/label-release scans; Run validation starts feature/model computation. Displaying the form does not train. Dataset-specific transformations, sector maps, proxy construction, baseline engineering and benchmark repairs stay in the user notebook or upstream pipeline; no dataset preset is installed.

## Optional walk-forward contract

`WalkForwardConfig` counts observed local dates. Minimum training and complete validation blocks are positive; inner embargo is nonnegative. `step_dates` must be at least the validation-block length, so no record receives two out-of-fold validation predictions. A rolling `max_train_dates` must be at least the minimum; omission gives expanding history. `n_splits` selects the latest available complete folds by dates only.

The raw engine reserves final Test through `TrainingConfig` or supplied outer split labels. Development ends at the last outer Validation timestamp and includes earlier embargo history. Walk-forward `holdout_dates` and `holdout_embargo_dates` must therefore both be zero. Missing prediction times reject the fold preview rather than silently dropping records. Preview counts describe input rows; fitting/scoring still apply finite-label and prediction support checks.

Each fold fits fresh models and Train-only categorical vocabularies. The pooled Validation table contains `cv_fold`; selection scores pooled records, not unweighted averages of fold losses. After selection, Base and the chosen candidate refit on development history, capped by `max_train_dates` if supplied. User-prepared BASE features must still be causal; the engine does not recompute arbitrary upstream rolling statistics or fit preprocessing on the user's behalf.

Optional `label_available_column` is an upstream label-release timestamp, not the outcome itself. This eligibility field cannot be a BASE feature. Fold Train excludes unknown release times and times at/after validation-start local midnight. Development refits apply the same availability rule at the final-Test origin. Preview fold counts include the resulting purges. Without this field, timely labels are an explicit user assumption; an embargo alone cannot infer an unknown release delay.

One shared LightGBM dictionary drives different feature counts. Feature-specific constraints, penalties, forced split/bin files and categorical/name overrides are rejected instead of being misaligned across families. Declare categoricals through `BASE_CAT_FEATURES`; independently constrained models belong in separate factories in `model_comparison_engine`.

With label availability configured, candidate selection also excludes OOF outcomes unavailable before the final origin: first Test local midnight, or next local midnight after development when Test is absent. The saved `cv_label_available` flag, masked scoring level and `unavailable_validation_labels` manifest count disclose these exclusions while original target/source columns remain preserved.
