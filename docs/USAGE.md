# Reproducing a report from raw data

## 1. Install

Download this repository as ZIP or clone it. Open a terminal in the extracted directory and run `python -m pip install -e ".[notebook]"`. Python 3.10+ is required. A separate virtual environment is recommended. On systems where LightGBM requires an OpenMP runtime, use a working LightGBM installation before starting a long run.

The notebook and CLI call the same public pipeline. The `demo` command needs no user data; its report is labelled synthetic throughout.

## 2. Map inputs once

Copy `examples/config.json`. Set transaction and quote file paths, base feature names, model parameters, and source column mappings. Relative paths are resolved beside the JSON file, including output paths. Use CSV or Parquet, or pass pandas DataFrames directly in the notebook.

Essential choices are target column, bond ID, prediction time, known quote time, dealer, quote sides/values, and spread-versus-price units. Optional metadata enables more analyses; it is never guessed from an unrelated column. Remove unavailable optional mappings. See [DATA_CONTRACT.md](DATA_CONTRACT.md) for exact meanings.

For a `data_ig`-style target that is a spread change, map its delta label as `target`, map `PREV_BM_SPREAD` as `anchor`, and set `training.target_mode` to `delta`. Models fit the delta; comparisons reconstruct levels with the same known anchor. Missing delta anchors are an input error. Do not map a delta label as a level merely because their error differences appear similar.

## 3. Run Steps 1–5

```bash
python -m raw_quote_engine.cli run --config examples/config.json
```

Progress identifies the current stage, elapsed time and completed work units. Event aggregation may initially be indeterminate while pandas groups messages. State progress counts bonds; path progress counts bond/day groups. These counters reflect the work, not independent observations or an estimated completion time.

| Stage | Calculation | Decision supported |
|---|---|---|
| 1 | Schema/time/units, universe overlap, global/sector/issuer/dealer/bond coverage | Whether the raw feed covers the prediction population; which metadata or mappings need repair |
| 2 | Full-row repeats, exact-event candidate sets, incomplete events, size status and guarded history | Whether an apparent move is a real state change, a refresh, or an ambiguous multi-price/size event |
| 3 | Same-day state, fresh/decayed quote levels, reliability, matched-dealer direction, issuer other-bond changes, path history | Which quote information is causally available beyond Base; support and missingness |
| 4 | Pair/crossing/age/size diagnostics, fixed random/typical/high-impact cases, feature availability | Which cases need source inspection and which candidate controls can be compared on identical support |
| 5 | Fixed LightGBM ablations, same-row validation losses, frozen candidate, saved predictions and slice comparisons | Whether quote features reduce error on the declared business cohort and whether tails deteriorate |

Default date fractions are 20% Validation, 20% Test, with one observed-date embargo before each and at least three training dates. A ten-date example becomes Train1–4 / Embargo5 / Validation6–7 / Embargo8 / Test9–10. The two stages never split a single local date. Shorter inputs require smaller fractions/embargo, validation-only `test_fraction=0`, or an explicit chronological split column.

Four model families share the same finite-label training rows and parameters: Base; Quote levels and reliability; Quote+Path; Quote+CrossBond, which adds same-bond dealer direction and other-bond issuer movement. There is no hyperparameter sweep or test early stopping. Transactions without quotes remain. The selected candidate has the lowest validation MAE on the predeclared selection population; ties prefer the earlier simpler family. A minimum-support fallback to all common validation rows is explicit in `selection.json`.

Default selection prioritizes quantity >=1MM and maturity >1 year when both mappings exist. Missing metadata or a small cohort produces a disclosed fallback, not fabricated support. Short maturity remains separately reported. A 5% practical improvement threshold is a decision aid, not a statistical significance claim. Multiple slice results are exploratory and overlap.

## 4. Open and review

Open the output `report.html`. Every numerical result is computed from this run and exported in `tables/` or the comparison assets. The report does not read the old BondCliQ aggregate bundle or contain hand-entered benchmark losses.

In a notebook:

```python
# FILE IO LOGIC: Restore a completed run, without reading raw quote files or fitting models again.
from raw_quote_engine import load_run, Slice
run = load_run("runs/my_research")
# UI LOGIC: Switch saved reference/candidate models, filters, one-dimensional slices and interactions.
run.review()
# CONFIGURATION LOGIC: Compare any two saved models, not necessarily Base.
comparison = run.compare("Quote", "Quote+CrossBond", stage="Validation")
# REPORTING LOGIC: Recompute an exact paired table for an arbitrary existing metadata column.
table = comparison.slice("rating", min_count=50)
# REPORTING LOGIC: Cross slices are computed from individual paired records, not averaged group metrics.
cross = comparison.cross_slice("SECTOR", Slice("QUANTITY", [0, 1e6, 5e6, float("inf")], right=False))
# FILE IO LOGIC: Export new complete PNG/CSV/HTML evidence without printing large tables.
comparison.export("reviews", interactions=[("SECTOR", Slice("MATURITY_YEARS", [0,1,3,10,float("inf")]))])
```

The UI supports custom numerical bins, category slices, two intersecting filters, two-column heatmaps with count panels, minimum support, gain/error metrics and saved exports. Add any new slice column to the original transaction DataFrame; it survives into saved predictions. `comparison.filter(...)` can create more complex intersections programmatically. Saved models from other projects can be wrapped in `Model` and compared through `compare_models`; saved prediction columns can use `compare_predictions`. These generic APIs do not train models.

## 5. Final evaluation

After fixing the validation choice:

```bash
python -m raw_quote_engine.cli final-test examples/runs/my_research
```

This uses the same trained Base and selected candidate, without a refit or additional candidate search. It verifies the frozen selection and evaluates only those two models. Repeat calls reuse saved test predictions. Previously inspected or externally exposed test data is not magically locked again; disclose that history in interpreting the report.

For an unattended run, use `run --config ... --final-test`. Its ordering still freezes the validation choice before generating test predictions. The engine never automatically changes cleaning rules after seeing test losses.

## Outputs and resume

`report.html` links actual tables/plots and documents exclusions, missing metadata, date support and limitations. `features.parquet` preserves one row per requested transaction. `predictions_validation.parquet` contains every fixed validation model; `predictions_test.parquet` contains only Base and the frozen candidate. `models/` contains native LightGBM text models and their exact feature/category schemas. `selection.json` is the validation decision. `manifest.json` records config, fingerprints, versions and evaluation status.

Repeat the same call to reuse completed stages. Changed inputs/settings require a new output directory and can reuse the shared content-addressed cache. Comments alone do not invalidate numerical cache signatures. To regenerate HTML only, run `python -m raw_quote_engine.cli report runs/my_research`. An interrupted incomplete run can be resumed with the original raw inputs/config; no kernel restart is needed to refresh a report.

Data, model binaries, caches, predictions and run reports remain local and are ignored by Git. Only code, the empty-output starter notebook and documentation are shipped. Cache/model pickle files must come from your own trusted run directory. Sharing a report may expose issuer/dealer labels and representative transaction records; use the resulting local files according to your data access policy.
