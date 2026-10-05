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

Review the ordered baseline list, its categorical subset, LightGBM parameters, source timezones, scales, date policy and output location. **Validate inputs** performs preflight without training. **Run validation** starts Steps 1–5 after valid settings are accepted. Progress identifies stages and completed work units. `controller.run` is `None` before completion and holds the resulting `ResearchRun` afterward. Notebook Run All only displays the form.

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

Four families share training rows and parameters: Base, Quote, Quote+Path and Quote+CrossBond. The last adds same-bond matched-dealer and other-bond issuer movements. There is no parameter search or test-dependent early stopping. Transactions without quotes remain. Category vocabularies are learned on Train only; the categorical subset is explicitly supplied.

Selection uses the lowest validation MAE among quote candidates on the declared cohort, with a support-based fallback recorded in `selection.json`. The five-input form defaults to All records. The BondCliQ example selects quantity >=1MM and maturity >1 year; choose this cohort explicitly in the generic form when appropriate. Short maturity and unknown metadata remain separate diagnoses. The 5% practical threshold is not a significance test; overlapping exploratory slices are not independent evidence.

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

Only Base and the selected candidate are evaluated, using the same Train-fitted models. No final refit is performed. Repeat calls reuse saved predictions. Prior external exposure of those dates remains a limitation; the engine cannot restore unseen status.

Outputs include `report.html`, full tables/figures, `features.parquet`, validation predictions, model files and schemas, `selection.json` and a manifest of configuration, versions, fingerprints and evaluation state. Explicit finalization adds test predictions and test evidence. Source inputs and generated artifacts stay local.

An unchanged call resumes completed stages. Changed inputs or settings require a new output folder; a shared content-addressed cache may reuse unaffected stages. Comments alone do not invalidate numerical signatures. Regenerate HTML with `python -m raw_quote_engine.cli report runs/my_research`. Use only trusted local cache/model files. Reports retain supplied labels and representative records, so review them before sharing.
