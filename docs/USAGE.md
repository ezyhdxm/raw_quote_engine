# From prepared DataFrames to a research report

The primary interface is an editable notebook form. It takes a transaction DataFrame, raw spread-quote DataFrame, ordered `BASE_FEATURES`, explicit `BASE_CAT_FEATURES` and `LGB_PARAMS`. The engine starts with these supplied definitions; production feature construction and source-specific preprocessing stay outside it.

For the existing BondCliQ/data_ig setup, open [bondcliq_research.ipynb](../bondcliq_research.ipynb) and follow [BONDCLIQ_EXAMPLE.md](BONDCLIQ_EXAMPLE.md). For other sources, open [raw_quote_research.ipynb](../raw_quote_research.ipynb).

## 1. Install and prepare

Download the repository ZIP or clone it. From its directory, run `python -m pip install -e ".[notebook]"`. Python 3.10+ and a working LightGBM installation are required. Use the same environment for the notebook kernel.

Read your inputs into DataFrames using your existing loader. Supply baseline features that are available at prediction time. Prepare identifiers, sector/issuer metadata, proxy levels, units, quantity multipliers and benchmark conventions upstream. The engine does not fetch a sector map, create a CPP proxy or convert prices to spreads. Only spread-valued quote input is supported.

The optional CLI accepts CSV/Parquet paths. CSV identities mapped by the engine are read as strings to preserve leading zeros. If you read CSVs yourself, explicitly preserve your identity columns' dtype.

## 2. Open the form

```python
# SETUP LOGIC: No data loading or model fitting occurs on import.
from raw_quote_engine import research_form
# UI LOGIC: Map required columns and configure optional fields in the displayed form.
controller = research_form(
    transactions_df, quotes_df, BASE_FEATURES, BASE_CAT_FEATURES, LGB_PARAMS,
    output='runs/my_research',
)
```

Map the transaction bond ID, prediction time and target; map the quote bond ID, known time, dealer, side/value or bid/ask values. Optional mappings include actual scoring spread, prediction anchor, rollover adjustment, proxy, quantity, issuer, sector, maturity and an explicit split. A configured optional field must exist. Leave its mapping unset when unavailable.

Review the ordered baseline list, its categorical subset, LightGBM parameters, source timezones, scales, date policy and output location. **Validate inputs** performs preflight without training; optional walk-forward mode also scans transaction times/split labels to preview folds. **Run validation** starts Steps 1–5 after valid settings are accepted. Progress identifies stages and completed work units. `controller.run` is `None` before completion and holds the resulting `ResearchRun` afterward. Notebook Run All only displays the form.

Supply `config=PipelineConfig(...)` and `training=TrainingConfig(...)` to prefill the form. Optionally supply `quote_universe=pd.DataFrame({'cusip': ...})` when the full research universe is wider than eligible modeling transactions. `cache_dir` can place reusable caches in another local folder.

## 3. Target and time contracts

With `target_mode='level'`, models predict the supplied spread label. With `target_mode='delta'`, the supplied target is unchanged, and prediction reconstruction is:

```text
predicted spread = predicted target + anchor - rollover adjustment
```

Map a rollover adjustment only if the upstream target has the convention `actual - anchor + adjustment`. If no column is mapped, adjustment is zero. A mapped missing/nonfinite adjustment rejects the run. An explicit `actual` column is the scoring truth and takes precedence over reconstructed labels. For example, predicted target `.03`, anchor `1.00`, adjustment `.02` and actual `1.02` produce spread `1.01` and error `-1 bp` with `error_scale=100`.

This scalar correction does not repair historical benchmark changes across quote timestamps. Source spreads and optional proxy must already share a justified benchmark and scale. The proxy/anchor gap compares the supplied proxy with `anchor - adjustment`; no proxy or fallback source is constructed. See [DATA_CONTRACT.md](DATA_CONTRACT.md).

Naive transaction timestamps use `timezone`; naive quote timestamps use `quote_timezone` when supplied, otherwise `timezone`. Aware timestamps preserve their instants and convert to the state timezone. Known time controls availability; an original vendor timestamp never replaces it. Same-day state never carries overnight.

## 4. Stages and evaluation policy

| Stage | Calculation | Decision supported |
|---|---|---|
| 1 | Schema, time, units, universe overlap and population coverage | Whether the feed covers the requested prediction population |
| 2 | Repeats, exact-event candidates, incomplete states and size status | Whether a message is a refresh, a changed state or an ambiguous event |
| 3 | Same-day quote levels, reliability, matched-dealer direction, issuer other-bond changes and path history | Which information is available beyond the supplied baseline |
| 4 | Age, gap, crossing, size and representative-case diagnostics | Which mechanisms need inspection and which comparisons have common support |
| 5 | Fixed LightGBM families, paired validation losses, selection and saved comparisons | Whether quote features improve the declared cohort and how tails behave |

Default fractions reserve 20% of observed dates for Validation and 20% for Test, with one observed-date embargo before each and at least three training dates. Fixed `validation_dates` and `test_dates` may instead specify counts. `embargo_dates` is before Validation; `test_embargo_dates` controls the Test boundary separately. Explicit input split labels override date allocation, subject to chronological validation. Single dates never straddle stages.

For the short BondCliQ example: `validation_dates=5`, `test_dates=5`, `embargo_dates=2`, `test_embargo_dates=0`, `min_train_dates=10`. On 22 observed weekdays from March 2–31, 2026, Train is March 2–13, embargo March 16–17, Validation March 18–24 and Test March 25–31.

Four families share training rows and parameters: Base, Quote, Quote+Path and Quote+CrossBond. The last adds same-bond matched-dealer movements and, when an issuer mapping is supplied, other-bond issuer movements. Without that mapping, the compatibility label `Quote+CrossBond` contains same-bond movement only; the family name is not evidence of available cross-bond support. There is no parameter search or test-dependent early stopping. Transactions without quotes remain. Category vocabularies are learned on Train only; the categorical subset is explicitly supplied.

Selection uses the lowest validation MAE among quote candidates on the declared cohort, with a support-based fallback recorded in `selection.json`. The five-input form defaults to All records. The BondCliQ example selects quantity >=1MM and maturity >1 year; choose this cohort explicitly in the generic form when appropriate. Short maturity and unknown metadata remain separate diagnoses. The 5% practical threshold is not a significance test; overlapping exploratory slices are not independent evidence.

## Optional walk-forward validation

Keep **Single split** for the existing workflow, or select **Walk forward** in the form. Configure inner fold sizes separately from the outer Test reservation. Click **Validate inputs** before Run: the form displays every fold's train/validation date boundaries, observed date counts and supplied row counts, plus the total fit budget. This date scan reads prediction times, outer split labels and optional label-release timestamps; it reads no target values, builds no quote features and fits no model. Changed fold dates/settings require reviewing a new preview before work starts.

```python
# SETUP LOGIC: The same policy object supports the form and programmatic pipeline.
from raw_quote_engine import WalkForwardConfig, research_form
# CONFIGURATION LOGIC: Expand from 20 observed dates, validate 5 dates, and embargo 2 before each fold.
walk_forward = WalkForwardConfig(
    min_train_dates=20, validation_dates=5, embargo_dates=2,
    step_dates=5, max_train_dates=None, n_splits=4,
)
# UI LOGIC: The form previews the latest available complete folds; training remains an explicit click.
controller = research_form(
    transactions_df, quotes_df, BASE_FEATURES, BASE_CAT_FEATURES, LGB_PARAMS,
    config=config, training=training, walk_forward=walk_forward, output='runs/walk_forward',
)
```

`n_splits` limits the plan to the latest complete folds using dates alone. `step_dates=None` defaults to the validation-block size; a larger step leaves intentional unscored gaps. Steps smaller than the validation block reject because validation rows may not overlap. `max_train_dates=None` expands training from the earliest eligible date; a positive cap creates a rolling training window and must be at least `min_train_dates`. Incomplete final blocks are not scored. Counts use observed local dates, not calendar subtraction.

The engine first assigns the existing outer Train/Validation/Test split. Development includes every row up to the last outer Validation timestamp, including earlier outer embargo rows. The final Test embargo and Test rows remain excluded from all folds and development refits. Do not set inner `holdout_dates` or `holdout_embargo_dates`: the raw engine requires both to be zero because the outer split owns that reservation.

Optional `label_available_column` names an upstream timestamp for when each training label became known. Labels with missing release time or release at/after the fold's validation-start local midnight are excluded from that fold's training records; the preview reports those purged counts. The same contract applies to development refits before final Test. The release-time column is an eligibility field and cannot be a BASE feature. Before candidate selection, out-of-fold outcomes still unavailable at the final origin are masked from scoring: the first Test date's local midnight, or the next local midnight after development when no Test exists. Saved `cv_label_available` flags and the manifest's `unavailable_validation_labels` count expose this exclusion; original supplied target/source fields remain available for provenance. Without this column, label availability at the origin is a caller assumption, and an embargo may be needed for the label horizon.

Every fold independently fits the four declared model families, with categories learned only on that fold's training rows. Predictions are pooled into a common-record out-of-fold Validation table with `cv_fold`. Candidate selection uses those pooled records, weighted by record count rather than averaging fold means. After freezing the winner, only Base and that candidate are refitted on development history; a configured rolling cap also limits those refits. Final Test remains an explicit later action. Total cost is **4 × folds + 2 fits**; event aggregation and time-causal quote features run once.

Use `run.review(stage='Validation')` to compare pooled out-of-fold predictions and choose `cv_fold` as a custom slice. Computed tables `cv_folds`, `cv_metrics` and `cv_stability` document date support, per-fold results and variation. The generated report carries this evidence. No confidence interval or significance claim follows automatically from a handful of folds. Overlapping training windows and correlated securities mean folds are not independent experiments.

The engine controls model-fitting boundaries, but cannot repair a user feature engineered with future data. Prepare baseline rolling features, imputations and benchmark inputs causally; any learned transformation must respect fold training boundaries. Quote feature reuse is valid only because quote calculations themselves obey known-time boundaries and do not use future labels.

Feature-specific LightGBM constraints cannot be shared blindly across augmented feature families. The raw engine rejects per-feature monotonicity/interaction vectors, feature penalties and forced split/bin files; use `BASE_CAT_FEATURES` for categories. To train models with independently defined feature constraints or preprocessing, use separate model factories in the standalone `model_comparison_engine` walk-forward interface.

For batch runs, pass `walk_forward=walk_forward` to `run_research`. In CLI JSON, set a top-level `walk_forward` object with the same fields; `null` or omission selects Single split. The example config keeps this option off.

## 5. Programmatic and CLI use

```python
# SETUP LOGIC: Explicit calls are suitable for batch runs after source preparation and configuration.
from raw_quote_engine import run_research
# ORCHESTRATION LOGIC: Start validation with the same input contract as the notebook form.
run = run_research(
    transactions_df, quotes_df, BASE_FEATURES, LGB_PARAMS, config,
    base_cat_features=BASE_CAT_FEATURES, training=training, output='runs/my_research',
)
```

For a CLI run, edit [examples/config.json](../examples/config.json), including `base_features` and `base_cat_features`:

```bash
python -m raw_quote_engine.cli run --config examples/config.json
```

Relative input/output paths resolve beside the JSON file. Commands intentionally start work; the notebook form requires its explicit Run action.

## 6. Compare and export saved results

Open the generated `report.html`. Every numerical result comes from the current run's tables or comparison objects. To review later without source files or refitting:

```python
# FILE IO LOGIC: Restore a trusted completed local run.
from raw_quote_engine import load_run, Slice
run = load_run('runs/my_research')
# UI LOGIC: Select saved models, population filters, single slices and two-column heatmaps.
panel = run.review(stage='Validation')
# CONFIGURATION LOGIC: Any pair of saved models can be selected; Base is not mandatory.
comparison = run.compare('Quote', 'Quote+CrossBond', stage='Validation')
# REPORTING LOGIC: Aggregate paired records on a retained user metadata column.
table = comparison.slice('rating', min_count=50)
# FILE IO LOGIC: Export complete PNG/CSV/HTML evidence without printing large result tables.
comparison.export('reviews', interactions=[('SECTOR', Slice('QUANTITY', [0, 1e6, 5e6, float('inf')], right=False))])
```

The comparison UI supports custom bins, two intersecting filters, interactions with count panels, support thresholds, gain/error metrics and exports. Additional source metadata stays available in saved predictions. `comparison.filter(...)` supports further programmatic intersections. Generic `compare_predictions` and `compare_models` also compare external saved results or loaded estimators without training them.

## 7. Final test and resume

After accepting the frozen validation choice, explicitly call `run.finalize_test()` or:

```bash
python -m raw_quote_engine.cli final-test examples/runs/my_research
```

Only Base and the selected candidate are evaluated. Single-split runs reuse the Train-fitted models. Walk-forward runs use the two development refits completed after the pooled-validation choice; final test starts no additional fit. Repeat calls reuse saved predictions. Prior external exposure of those dates remains a limitation; the engine cannot restore unseen status.

Outputs include `report.html`, full tables/figures, `features.parquet`, validation predictions, model files and schemas, `selection.json` and a manifest of configuration, versions, fingerprints and evaluation state. Explicit finalization adds test predictions and test evidence. Source inputs and generated artifacts stay local.

An unchanged call resumes completed stages. Changed inputs or settings require a new output folder; a shared content-addressed cache may reuse unaffected stages. Comments alone do not invalidate numerical signatures. Regenerate HTML with `python -m raw_quote_engine.cli report runs/my_research`. Use only trusted local cache/model files. Reports retain supplied labels and representative records, so review them before sharing.
