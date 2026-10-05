# %% [markdown]
# # Raw quote research: inputs to report
#
# Run the cells in order. The default is a labelled **synthetic smoke run**. For real data, edit `examples/config.json`, set MODE to `"real"`, and keep the same pipeline. No prior BondCliQ artifacts are required.
#
# The final test is opt-in and uses the frozen validation choice. Do not use test slices to repeatedly revise the model.

# %%
# CONFIGURATION LOGIC: Pick synthetic smoke data or the user-edited file/column mappings.
MODE = "demo"
CONFIG_FILE = "examples/config.json"
OUTPUT = "runs/notebook_demo"
RUN_FINAL_TEST = False

# SETUP LOGIC: The installed package supplies the same workflow used by the command line.
from pathlib import Path
from IPython.display import HTML, display
from raw_quote_engine import load_run, run_research, Slice
from raw_quote_engine.cli import run_file
from raw_quote_engine.demo import demo_inputs

# %% [markdown]
# ## Run Steps 1–5
#
# This cell normalizes once, caches event/feature/model stages, and emits compact progress. Repeating it with unchanged inputs restores completed work. To compare a changed experiment, choose a new output directory; shared caches can still be reused.

# %%
# ORCHESTRATION LOGIC: Either branch calls the public raw-input pipeline; no numerical results are supplied by hand.
if MODE == "demo":
    transactions, quotes, base_features, config = demo_inputs()
    run = run_research(transactions, quotes, base_features, {"n_estimators": 25, "n_jobs": 2},
                       config, output=OUTPUT, evaluate_test=RUN_FINAL_TEST)
elif MODE == "real":
    run = run_file(CONFIG_FILE, evaluate_test=RUN_FINAL_TEST)
else:
    raise ValueError('MODE must be "demo" or "real".')
# UI LOGIC: Keep the full report as a local artifact; avoid printing large tables into the notebook.
display(HTML(f'<p>Report written to <code>{run.report}</code>. Open this file in your browser.</p>'))

# %% [markdown]
# ## Review saved model predictions
#
# Use model selectors, one-dimensional slices and two-column heatmaps. Metadata missing from the inputs remains unsupported. These controls reuse saved predictions, never fit another model. Large-trade results and short-maturity results should be interpreted separately.

# %%
# UI LOGIC: One compact workbench with explicit Apply and export controls.
panel = run.review(stage="Validation")

# %% [markdown]
# ## Optional custom export
#
# Any retained transaction column can be sliced. This example uses canonical sector and quantity only when both mappings are available. Exports include full PNGs and exact CSVs; no QR transfer or table printing.

# %%
# CONFIGURATION LOGIC: Select any two saved models and declare fixed slice boundaries before reading their losses.
comparison = run.compare("Base", run.metadata["selection"]["selected"], stage="Validation")
size = Slice("QUANTITY", [0, 1e6, 5e6, float("inf")], right=False)
# FILE IO LOGIC: Export the requested interaction only when the data contract supplied both columns.
if {"SECTOR", "QUANTITY"}.issubset(comparison.data):
    review_folder = comparison.export(run.output / "custom_reviews", interactions=[("SECTOR", size)])

# %% [markdown]
# ## Final test, when ready
#
# The initial `RUN_FINAL_TEST` option can execute this automatically after validation selection. Otherwise call `run.finalize_test()` explicitly after accepting the frozen choice. Repeating that call reuses saved test predictions. The engine cannot certify that someone has not previously inspected these test dates.
#
# To reopen later without raw inputs: `run = load_run("runs/notebook_demo")`. To regenerate HTML only: `python -m raw_quote_engine.cli report runs/notebook_demo`.
