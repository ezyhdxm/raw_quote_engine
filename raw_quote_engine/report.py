"""Generate a local report directly from one run's diagnostics and computed comparisons."""
# SETUP LOGIC: Report creation is explicit; importing never reads data or trains models.
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from html import escape
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
from .diagnostics import diagnostic_view, priority_groups, settings

# CONFIGURATION LOGIC: Keep generated evidence and interpretation visibly separate.
STYLE = '''body{max-width:1200px;margin:36px auto;padding:0 22px;font:16px/1.6 system-ui;color:#193844}
h1,h2,h3{line-height:1.25}table{border-collapse:collapse;font-size:13px;width:100%}
th,td{padding:7px;border:1px solid #d8e2e6;text-align:left}th{background:#edf4f5}
.scroll{overflow:auto;max-height:550px}img{width:100%;height:auto}.note{padding:14px;background:#edf6f7;border-left:5px solid #1a8285}
.warning{background:#fff2dd;border-color:#b77d24}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f7;padding:14px}
section{margin:30px 0}a{color:#087482}small{color:#556c75}code{overflow-wrap:anywhere}'''
DESCRIPTIONS = {
    'global_summary': 'Separate target, bond, canonical quote-side row and exact-event denominators. Outside-universe and invalid values are audited upstream as well.',
    'bond': 'Every supplied traded bond, including bonds without any quote record in the supplied file. This is file coverage, not prediction-time availability.',
    'sector': 'Bond metadata is assigned only when its observed label is unique. Unknown and conflicting groups are retained; this is descriptive mapping, not causal feature construction.',
    'issuer': 'All observed issuer groups, including no-quote bonds. Counts are not equally weighted model errors or independent issuer samples.',
    'dealer': 'Raw rows and distinct events have separate denominators. Repeated messages and multiple candidate prices do not establish independent dealer votes.',
    'quantity': 'Quote-size classes describe the source field. Missing differs from invalid when the ingestion quantity_kind flag is supplied. Quote quantity is not automatically comparable with transaction par amount.',
    'event_quality': 'Exact dealer/bond/side/known-time events retain distinct price candidates, positive-size counts and incomplete records. Repeat counts are not unique price counts.',
    'target_coverage': 'Availability at actual transaction times. Two usable sides do not imply a fresh same-dealer pair or matched positive size. Unknown features are unassessed.',
    'pair_sources': 'Distributions of saved target-level pair and age aggregates. These identify conditions worth examining, not causal attribution to individual dealer pairs. Missing support is not a measured zero gap.',
    'feature_availability': 'Each feature has its own observed denominator. Legacy names ending in _bps do not override the declared canonical value scale; ages are minutes, shares are fractions, and support fields are counts.',
    'priority_support': 'Fixed large-trade, maturity, sparse-history and optional CPP groups. Unsupported groups remain explicit. Observed low counts can reflect truncated history rather than illiquidity.',
    'maturity_support': 'All four declared maturity bins are retained for all targets and large trades, including empty groups. Negative and nonfinite remaining terms are unknown; short instruments are not deleted.',
    'history_coverage': 'Trailing 30-day observed counts and supplied-history coverage. A complete 30-day time span cannot certify feed completeness; shorter history cannot establish true illiquidity.',
    'cases': 'Four deterministic explanatory roles: one seeded random event, one typical candidate range and up to two widest ranges. A repeated event can serve multiple roles. High range is not demonstrated model harm. These local rows contain private identifiers.',
    'case_pair_queries': 'At most eighteen frozen transaction queries chosen by seeded random, typical absolute gap and high absolute gap roles; overlapping identities are deduplicated. Selection never uses target labels or model errors.',
    'case_pair_policy': 'Case-only paired calculator: A uses fresh original slots; B selects match-eligible slots but retains original candidates; C matches positive quantities on exactly those B slots. B and C have identical counts. Gaps are bid spread minus ask spread in canonical spread units.',
    'case_pair_effect': 'Case-only differences: A-to-B is a support-selection change; B-to-C is same-slot quantity matching. This isolates the arithmetic policy effect but does not establish causal size effects or population benefit. No matching support produces an unknown difference.',
}


def _portable(value):
    # SERIALIZATION LOGIC: Preserve configuration/provenance with standard JSON, without arbitrary object repr.
    if is_dataclass(value):
        return _portable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _portable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [_portable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f'Report metadata must be JSON-compatible, not {type(value).__name__}.')


def _table(table, limit=100):
    # FORMATTING LOGIC: Escape input labels and show a bounded preview; the CSV always contains every row.
    shown = table.head(limit)
    note = f'<p><small>Showing {len(shown):,} of {len(table):,} rows. Download the CSV for full precision and all groups.</small></p>'
    return note + '<div class="scroll">' + shown.to_html(index=False, escape=True, float_format=lambda value: f'{value:,.6g}') + '</div>'


def _filename(index, label):
    # FILE NAMING LOGIC: An integer prefix prevents sanitized labels from overwriting one another.
    stem = re.sub(r'[^A-Za-z0-9_-]+', '_', str(label)).strip('_')[:70] or 'table'
    return f'{index:02d}_{stem}.csv'


def comparison_priorities(comparison, config=None):
    """Recompute paired metrics on fixed groups; unsupported optional groups remain visible."""
    # SETUP LOGIC: Reuse the independent comparison engine; this path calls no estimators.
    from bond_pricer import compare_predictions
    options = settings(config)
    declared = diagnostic_view(comparison.data, options)
    groups, history, cpp_available = priority_groups(declared, options)
    output = []
    for name, mask, required in groups:
        # CORE LOGIC: STEP 1 — Apply the same group mask to both models before recomputing losses.
        # Input: quantities=[2e6,5e5], reference abs errors=[4,2], candidate abs errors=[3,1].
        # Output: >=1MM has n=1,reference_mae=4,candidate_mae=3,mae_improvement_pct=25.
        # Explanation: The small trade contributes to neither side of the large-trade comparison.
        # Trick: Filtering the input population recomputes coverage; percentiles are never subtracted across overlapping cohorts.
        supported = all(c is not None and c in declared for c in required) and ('CPP' not in name or cpp_available)
        subset = comparison.data.loc[mask] if supported else comparison.data.iloc[:0]
        result = compare_predictions(subset, **comparison.config)
        record = result.summary(min_count=options.get('selection_min_count', 30)).iloc[0].to_dict()
        record.update(cohort=name, supported=supported, input_n=len(subset), history_field=history)
        output.append(record)
    return pd.DataFrame(output)


def _comparison_section(output, label, comparison, index, config):
    # CONFIGURATION LOGIC: Canonical maturity and rolling-history names are explicit additions to generic defaults.
    from bond_pricer import Slice, default_slices
    declared = diagnostic_view(comparison.data, config, preserve_priority_sources=False)
    slices = default_slices(declared.columns)
    if 'cv_fold' in comparison.data:
        slices.append(Slice('cv_fold', top_n=100, name='Walk-forward fold'))
    maturity = next((name for name in ['MATURITY_YEARS', 'YRS_TO_MATURITY'] if name in declared), None)
    if maturity and not any(item.column == maturity for item in slices):
        slices.append(Slice(maturity, [-np.inf, 0, .25, 1, 3, 5, 10, np.inf], name='Remaining maturity'))
    if 'prior_trade_count_30d' in comparison.data:
        slices.append(Slice('prior_trade_count_30d', [-np.inf, 0, 15, 30, np.inf], right=False, name='Observed prior 30-day trade count'))
    # CONFIGURATION LOGIC: Fixed interactions are offered only when both actual metadata fields exist.
    size = next((item for item in slices if item.column == 'QUANTITY'), None)
    interactions = []
    if size is not None and 'SECTOR' in declared:
        interactions.append((Slice('SECTOR', top_n=12), size))
    if size is not None and maturity:
        interactions.append((next(item for item in slices if item.column == maturity), size))
    # FILE IO LOGIC: Generic reports export aggregate tables and figures without row-level predictions.
    detail = comparison.export(output / 'comparisons', slices=slices, interactions=interactions,
                               min_count=config.get('selection_min_count', 30))
    priority = comparison_priorities(comparison, config)
    filename = _filename(index, f'{label}_priorities')
    priority.to_csv(output / 'tables' / filename, index=False)
    # REPORTING LOGIC: Label the supplied stage/run; neither label nor report order selects a candidate.
    summary = comparison.summary(config.get('selection_min_count', 30))
    location = detail.relative_to(output).as_posix()
    section = f'<section><h3>{escape(str(label))}</h3><p>{escape(comparison.candidate_name)} versus {escape(comparison.reference_name)} · error unit: {escape(comparison.unit)}.</p>'
    section += _table(summary) + f'<img src="{location}/overview.png" alt="Computed paired model comparison">'
    section += '<h4>Fixed priority groups</h4><p>These overlapping slices are diagnostic; missing prerequisites remain unsupported. A 5% gain, if used, is a descriptive threshold rather than significance.</p>'
    section += _table(priority) + f'<p><a href="tables/{filename}">Exact priority metrics CSV</a> · <a href="{location}/report.html">Complete comparison, slices, date sensitivity and configuration</a></p></section>'
    return section, priority


def _decision(metadata):
    # REPORTING LOGIC: Surface the frozen validation decision and explicit evaluation status before diagnostics.
    selection = metadata.get('selection')
    if not selection:
        return '<section><h2>Recorded decision</h2><p>No selection record was supplied. This report does not choose a model.</p></section>'
    # CORE LOGIC: STEP 1 — Evaluate the declared practical threshold from the recorded validation gain.
    # Input: improvement_pct=1.2,threshold_pct=5.
    # Output: practical_threshold_met=False.
    # Explanation: The business gate compares the recorded gain with its predeclared threshold.
    # Trick: Missing/nonfinite values leave the gate unassessed; this is never a significance test.
    values = pd.to_numeric(pd.Series([selection.get('improvement_pct'), selection.get('threshold_pct')]), errors='coerce')
    known = np.isfinite(values).all()
    threshold_met = bool(values.iloc[0] >= values.iloc[1]) if known else None
    # REPORTING LOGIC: Every displayed number comes from the selection artifact, not a newly searched slice.
    rows = [('Frozen candidate', selection.get('selected', 'Not recorded')),
            ('Requested validation population', selection.get('requested_slice', 'Not recorded')),
            ('Effective validation population', selection.get('effective_slice', 'Not recorded')),
            ('Support fallback used', selection.get('support_fallback', 'Not recorded')),
            ('Validation rows used', selection.get('rows', 'Not recorded')),
            ('OOF labels unavailable before selection', selection.get('unavailable_validation_labels', 'Not configured')),
            ('Validation MAE improvement (%)', selection.get('improvement_pct', 'Not recorded')),
            ('Declared practical threshold (%)', selection.get('threshold_pct', 'Not recorded')),
            ('Practical threshold met', threshold_met if known else 'Unassessed'),
            ('Test used for selection', selection.get('test_used_for_selection', 'Not recorded')),
            ('Fit/evaluation policy', selection.get('fit_policy', 'Not recorded')),
            ('Test status', metadata.get('test_status', 'Not recorded')),
            ('Earlier Test exposure', metadata.get('test_exposure_history', 'Not recorded'))]
    table = pd.DataFrame(rows, columns=['Recorded item', 'Value'])
    return '<section><h2>Frozen validation decision and Test status</h2>' + _table(table) + '<p>Threshold attainment is descriptive. Test comparisons below review the already chosen candidate; they do not select a replacement.</p></section>'


def _test_summary(comparisons, selected):
    # REPORTING LOGIC: Show a selected-candidate Test score only when the caller actually supplied those results.
    pieces = []
    for label, comparison in comparisons.items():
        stages = comparison.data.get('split', comparison.data.get('stage', pd.Series(dtype='string'))).dropna().unique()
        if len(stages) == 1 and stages[0] == 'Test' and comparison.candidate_name == selected:
            pieces.append(comparison.summary().assign(comparison=str(label)))
    if not pieces:
        return '<section><h2>Selected-candidate Test results</h2><p>No identifiable selected-candidate Test comparison was supplied. No Test number is inferred.</p></section>'
    return '<section><h2>Selected-candidate Test results</h2><p>Computed directly from the supplied saved Test comparison, on paired rows.</p>' + _table(pd.concat(pieces, ignore_index=True)) + '</section>'


def write_report(out, tables, comparisons=None, metadata=None):
    """Write report.html, complete diagnostic CSVs and comparison reports; return the HTML path.

    tables is diagnostic_tables(...) plus optional computed tables. comparisons maps
    display labels to bond_pricer.Comparison objects. metadata['config'] holds the
    pipeline configuration; metadata['synthetic'] overrides its synthetic flag.
    """
    # VALIDATION LOGIC: Refuse accidental raw objects where computed tables or model comparisons are expected.
    if any(not isinstance(value, pd.DataFrame) for value in tables.values()):
        raise TypeError('Report tables must all be pandas DataFrames.')
    metadata, comparisons = dict(metadata or {}), dict(comparisons or {})
    config = settings(metadata.get('config'))
    synthetic = metadata.get('synthetic', config.get('synthetic'))
    # FILE IO LOGIC: The pipeline supplies a run-specific directory; only report artifacts are written here.
    output = Path(out)
    (output / 'tables').mkdir(parents=True, exist_ok=True)
    provenance = _portable(metadata)
    (output / 'report_metadata.json').write_text(json.dumps(provenance, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
    # REPORTING LOGIC: Never describe an unmarked input as verified real data or a synthetic run as market evidence.
    source = 'SYNTHETIC DEMONSTRATION — not evidence of market performance' if synthetic else 'Supplied input run — inspect provenance and sample coverage'
    if synthetic is None:
        source = 'Input provenance unspecified — interpret only after verifying the supplied data'
    intro = f'<h1>Raw quote research run</h1><p class="note warning">{escape(source)}</p>'
    intro += '<p>Every numerical table and figure in this report is generated from this run\'s supplied data, computed diagnostics and comparison objects.</p>'
    intro += '<p>Read coverage first, then event ambiguity, pair/age conditions, fixed cases and model comparisons. All-target coverage and paired-model accuracy answer different questions.</p>'
    intro += _decision(metadata) + _test_summary(comparisons, metadata.get('selection', {}).get('selected'))
    if metadata.get('walk_forward') is not None:
        intro += '<section><h2>Walk-forward validation</h2><p>Validation comparisons pool nonoverlapping out-of-fold predictions. '
        intro += 'Each fold fits fresh estimators and category vocabularies using earlier training records. Prior validation records may enter a later training window. '
        intro += 'The final Test and its outer buffer are excluded from every fold. After selection, only Base and the frozen candidate are refitted on eligible development history.</p>'
        intro += '<p>Selection uses record-weighted pooled loss; fold win rates and worst-fold gains are descriptive stability checks, not confidence intervals. '
        intro += 'Configured label-availability times purge unavailable labels; without them, the caller must establish that training labels are already known.</p>'
        intro += '<pre>'+escape(json.dumps(metadata['walk_forward'], indent=2))+'</pre></section>'
    # REPORTING LOGIC: Unit and timing contracts are explicit; interpretation cannot be inferred from column names.
    contract = {name: config.get(name, 'not supplied') for name in ['value_kind', 'unit', 'quote_scale', 'target_scale',
                'error_scale', 'quantity_scale', 'quote_quantity_scale', 'timezone', 'quote_timezone', 'allow_exact', 'age_min', 'sync_min', 'lookback_min', 'case_seed']}
    intro += '<details><summary>Declared unit, timing and case-selection contract</summary><pre>' + escape(json.dumps(contract, indent=2)) + '</pre></details>'
    intro += '<p>Only spread input is supported. Quotes and target/anchor levels must share canonical units after input scaling. Error scale converts prediction errors to the displayed unit. Raw quality ranges are in canonical spread units; age fields are minutes.</p>'
    # REPORTING LOGIC: State the configured reconstruction rule and distinguish prepared labels from predictions.
    roles = config.get('transactions', {})
    delta = metadata.get('training', {}).get('target_mode') == 'delta'
    formula = 'prediction = model output + anchor - rollover adjustment' if delta else 'prediction = model output'
    adjustment = roles.get('rollover_adjustment') or 'not supplied (zero)'
    truth = roles.get('actual') or ('target + anchor - rollover adjustment' if delta else 'target')
    intro += '<p>' + escape(formula) + '; adjustment source: <code>' + escape(adjustment) + '</code>; scoring truth: <code>' + escape(truth) + '</code>. The supplied training target is never rewritten.</p>'
    # FILE IO LOGIC: Export every supplied diagnostic table, preserving empty and unsupported groups.
    sections, manifest = [], []
    for index, (name, table) in enumerate(tables.items(), 1):
        filename = _filename(index, name)
        table.to_csv(output / 'tables' / filename, index=False)
        description = DESCRIPTIONS.get(name, 'Additional computed run table supplied by the pipeline.')
        sections.append(f'<section><h2>{escape(str(name).replace("_", " ").title())}</h2><p>{escape(description)}</p>' + _table(table) +
                        f'<p><a href="tables/{filename}">Download complete CSV</a></p></section>')
        manifest.append(dict(table=str(name), rows=len(table), columns=list(table.columns), file=f'tables/{filename}'))
    # REPORTING LOGIC: Comparison objects already contain predictions; reports do not fit, infer or choose a winner.
    model_sections = ['<h2>Computed model comparisons</h2>']
    for index, (label, comparison) in enumerate(comparisons.items(), len(tables) + 1):
        section, priority = _comparison_section(output, label, comparison, index, config)
        model_sections.append(section)
    if not comparisons:
        model_sections.append('<p>No model comparisons were supplied. These diagnostics do not establish predictive gain.</p>')
    # FILE IO LOGIC: Save only an aggregate table manifest; local explanatory case CSVs are explicitly identified.
    (output / 'table_manifest.json').write_text(json.dumps(_portable(manifest), allow_nan=False, indent=2), encoding='utf-8')
    limits = '''<section><h2>Interpretation limits</h2><ul>
<li>Known time controls availability. Original event timestamps do not justify using a record before receipt; exact-time inclusion follows the declared setting. Same-day resets prevent overnight carry-forward.</li>
<li>Duplicates, multi-price states, zeros, negative values and crossing are observations to diagnose. They are not automatic deletion rules. A median between candidates is a statistic, not necessarily an executable quote.</li>
<li>Spread width is bid spread minus ask spread; negative width denotes crossing. A positive/negative target-level mean gap does not identify every dealer pair. Synchronization and size matching change the eligible population; compare the same slots before attributing an effect.</li>
<li>Quote quantity units and transaction par units need independent confirmation. Size matching is equality of supplied positive labels, not an estimated size curve. Current quantity, prior trade quantity and quote quantity are distinct fields.</li>
<li>Message age and real-change age summarize different dealer histories and may have different assessable samples. A zero single-dealer dispersion is not consensus. No peer-supported side or fresh pair does not mean no quotes exist.</li>
<li>Low historical trade counts can reflect missing or truncated data. Unknown maturity remains unknown; short-maturity trades remain present. Reference-proxy discrepancy uses proxy minus (anchor minus configured rollover adjustment), then the declared error scale, unless an explicit discrepancy column is supplied. No source selection or missing-row proxy fallback is inferred. Discrepancy does not certify a bad anchor.</li>
<li>Rollover adjustment changes delta reconstruction and the anchor basis; it does not rebuild targets, rewrite base features or infer benchmark changes across the raw quote history. Users must prepare comparable spread definitions and point-in-time inputs.</li>
<li>Common-dealer direction and other-bond issuer movement require valid support; changing dealers, quantities or donors can alter aggregate levels. They do not fit or identify a latent dealer/bond factor model.</li>
<li>The supplied baseline feature schema, categorical list, target and anchor define this run. No trade-context category or base feature is generated implicitly. Baseline context may already contain issuer, quantity or previous-trade information; compare incremental families on identical rows.</li>
<li>Validation selection must precede final Test evaluation. This report does not certify an unexposed Test or choose a candidate from Test. Inspect split/selection/exposure provenance. LODO is descriptive removal of saved dates, not retraining, a confidence interval or significance.</li>
<li>Local fixed-case CSVs may contain private dealer/bond identifiers. The generic comparison export is aggregate, but diagnostic bond/dealer tables and cases require review before sharing. No data from this run belongs in the shipped source repository by default.</li>
</ul></section>'''
    provenance_html = '<details><summary>Complete configuration and provenance</summary><pre>' + escape(json.dumps(provenance, ensure_ascii=False, indent=2)) + '</pre></details>'
    document = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Raw quote research run</title><style>' + STYLE + '</style></head><body>'
    target = output / 'report.html'
    target.write_text(document + intro + ''.join(sections) + ''.join(model_sections) + limits + provenance_html + '</body></html>', encoding='utf-8')
    return target
