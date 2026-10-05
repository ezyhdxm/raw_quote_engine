"""Raw inputs to reproducible diagnostics, fixed model experiments, and an evidence-backed report."""
# SETUP LOGIC: Importing does not load user data, train models or open a notebook display.
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
import json
import pickle
import hashlib
import os
import numpy as np
import pandas as pd
from bond_pricer import compare_predictions, show_comparison, Slice
from bond_pricer.slices import default_slices
from .config import PipelineConfig
from .ingest import normalize_transactions, normalize_quotes
from .state import prepare_quote_events
from .features import assemble_features, feature_dictionary
from .training import TrainingConfig, assign_splits, variants, fit_models, predict_stage, comparisons_for, select_candidate, validate_model_parameters
from .cache import Progress, StageCache, code_signature, fingerprint, digest, write_json
from .diagnostics import diagnostic_tables, diagnostic_view
from .report import write_report
from .validation import WalkForwardConfig, walk_forward_plan, run_walk_forward, final_training_frame


@dataclass
class ResearchRun:
    # SETUP LOGIC: A result is inspectable without fitting again or reloading raw quote messages.
    output: Path
    features: pd.DataFrame
    predictions: dict
    tables: dict
    comparisons: dict
    metadata: dict

    @property
    def report(self):
        # FILE IO LOGIC: Reports are generated inside the run directory from its saved tables.
        return self.output/'report.html'

    def compare(self, reference='Base', candidate=None, stage='Validation'):
        # CONFIGURATION LOGIC: Any two saved models may be compared; neither is required to be Base.
        candidate = candidate or self.metadata['selection']['selected']
        config = PipelineConfig.from_dict(self.metadata['config'])
        return compare_predictions(self.predictions[stage], 'actual_level', 'prediction_'+reference,
            'prediction_'+candidate, reference_name=reference, candidate_name=candidate,
            id_column='row_id', time_column='time', bond_column='cusip',
            error_scale=config.error_scale, unit=config.unit, timezone=config.timezone)

    def review(self, stage='Validation'):
        # UI LOGIC: The existing reusable comparison panel exposes model selectors, custom slices and heatmaps.
        frame = self.predictions[stage]
        options = {c.removeprefix('prediction_'): c for c in frame if c.startswith('prediction_')}
        config = PipelineConfig.from_dict(self.metadata['config'])
        slices = default_slices(diagnostic_view(frame, config.to_dict(), preserve_priority_sources=False).columns)
        if 'cv_fold' in frame:
            slices.append(Slice('cv_fold', top_n=100, name='Walk-forward fold'))
        return show_comparison(frame, actual='actual_level', predictions=options,
            id_column='row_id', time_column='time', bond_column='cusip',
            error_scale=config.error_scale, unit=config.unit, timezone=config.timezone,
            default_slices=slices)

    def finalize_test(self):
        # EVALUATION LOGIC: This explicit call reuses the frozen validation choice and trained estimators.
        return finalize_test(self.output)


def run_research(transactions, quotes, base_features, model_params, config, *, output='runs/research',
                 base_cat_features=None, training=None, evaluate_test=False, cache_dir=None, progress=None,
                 quote_universe=None, walk_forward=None):
    """Run Steps 1–5. Existing completed stages resume only when data and settings match."""
    # CONFIGURATION LOGIC: Keep source mappings, feature definitions and model settings explicit.
    config = config if isinstance(config, PipelineConfig) else PipelineConfig.from_dict(config)
    training = training if isinstance(training, TrainingConfig) else TrainingConfig(**(training or {}))
    walk_forward = walk_forward if isinstance(walk_forward, WalkForwardConfig) or walk_forward is None else WalkForwardConfig(**walk_forward)
    if walk_forward is not None and walk_forward.label_available_column in base_features:
        raise ValueError('Label availability timestamps are evaluation metadata, not baseline features.')
    validate_model_parameters(model_params)
    if config.transactions.rollover_adjustment is not None and training.target_mode != 'delta':
        raise ValueError('A rollover_adjustment mapping requires target_mode="delta"; supply target=actual-anchor+adjustment.')
    notify = Progress(progress)
    output = Path(output).expanduser().resolve()
    # FILE IO LOGIC: An unrelated existing folder is never treated as an engine-owned result directory.
    if output.exists() and any(output.iterdir()) and not (output/'manifest.json').exists():
        raise ValueError('Choose an empty output directory or an existing engine run; unrelated files will not be overwritten.')
    output.mkdir(parents=True, exist_ok=True)
    cache = StageCache(cache_dir or output.parent/'.raw_quote_cache', notify)
    # INGESTION LOGIC: Source files are read once; users retain their original DataFrames unchanged.
    notify('step1', None, None, 'Normalizing supplied transactions and known-time raw quotes')
    tx, tx_audit = normalize_transactions(transactions, config)
    validate_features(tx, base_features, config, training, base_cat_features)
    raw, quote_audit = normalize_quotes(quotes, config, tx if quote_universe is None else quote_universe)
    tx = assign_splits(tx, training, use_supplied_split=config.transactions.split is not None)
    if walk_forward is not None:
        walk_forward_plan(tx, walk_forward, config.timezone)
    # CACHEING LOGIC: Numerical code, full-precision input contents and explicit settings identify this run.
    numerical = code_signature(['ingest.py', 'state.py', 'movement.py', 'path.py', 'features.py'])
    event_key = digest(fingerprint(raw), code_signature(['state.py']), version('pandas'), version('numpy'))
    feature_key = digest(fingerprint(tx), event_key, config.to_dict(), numerical)
    library_versions = {name: version(name) for name in ('pandas', 'numpy', 'lightgbm')}
    model_inputs = [feature_key, base_features, base_cat_features, model_params, training.to_dict(), code_signature(['training.py']), library_versions]
    if walk_forward is not None:
        model_inputs += [walk_forward.to_dict(), code_signature(['validation.py', '../bond_pricer/walk_forward.py'])]
    model_key = digest(*model_inputs)
    receipt = output/'manifest.json'
    if receipt.exists() and json.loads(receipt.read_text()).get('run_key') != model_key:
        raise ValueError('This output directory belongs to different inputs/settings. Choose a new output directory; shared caches remain reusable.')
    # CACHEING LOGIC: Completed runs keep their original frozen decision and any already exposed test.
    if receipt.exists() and json.loads(receipt.read_text()).get('status') != 'running':
        run = load_run(output)
        # PROVENANCE LOGIC: Excluded rows can change audit counts without changing numerical cache inputs.
        run.metadata['normalization'] = {'transactions': tx_audit, 'quotes': quote_audit}
        write_json(receipt, run.metadata)
        render(run, config)
        notify('report', 1, 1, 'Reusing complete run: '+str(run.report))
        return finalize_test(output, progress=progress) if evaluate_test else run
    metadata = manifest(config, training, model_key, feature_key, model_params, base_features, tx_audit, quote_audit, base_cat_features)
    metadata['walk_forward'] = None if walk_forward is None else walk_forward.to_dict()
    write_json(receipt, metadata)
    # ORCHESTRATION LOGIC: One event aggregation feeds state, movement, path and diagnostic calculations.
    events = cache.get('step2_events', event_key, lambda: prepare_quote_events(raw, notify))
    frame, feature_meta = cache.get('step3_features', feature_key, lambda: assemble_features(tx, raw, events, config, notify))
    notify('step4', None, None, 'Summarizing quality, same-sample support and representative cases')
    tables = diagnostic_tables(tx, raw, events, frame, config)
    tables['feature_dictionary'] = feature_dictionary(frame, feature_meta['groups'], config)
    tables['splits'] = split_table(frame)
    definitions = variants(base_features, feature_meta['groups'])
    # MODELING LOGIC: The optional fold path reuses the same feature frame and persists completed folds independently.
    actual_column = 'actual' if config.transactions.actual is not None else None
    adjustment_column = 'rollover_adjustment' if config.transactions.rollover_adjustment is not None else None
    if walk_forward is None:
        models = cache.get('step5_models', model_key, lambda: fit_models(frame, definitions, model_params, training, notify, base_cat_features))
        validation = cache.get('step5_validation', model_key, lambda: predict_stage(frame, models, 'Validation', training, actual_column, adjustment_column))
    else:
        validation, fold_tables = run_walk_forward(frame, definitions, model_params, training, config,
            walk_forward, cache, model_key, notify, base_cat_features)
        tables.update(fold_tables)
    selected = select_candidate(validation, config, training)
    if walk_forward is not None:
        selected.update(validation_policy='Pooled nonoverlapping out-of-fold predictions; record-weighted selection',
            fit_policy='Fresh model per fold; after selection, refit Base and frozen candidate on development history only')
        selected['unavailable_validation_labels'] = int((~validation.cv_label_available).sum()) if 'cv_label_available' in validation else None
    # FILE IO LOGIC: Commit the validation decision before any test predictions or test loss calculations.
    write_json(output/'selection.json', selected)
    if walk_forward is not None:
        final_frame = final_training_frame(frame, walk_forward, config.timezone)
        final_definitions = {name: definitions[name] for name in ('Base', selected['selected'])}
        models = cache.get('step5_final_models', model_key, lambda: fit_models(
            final_frame, final_definitions, model_params, training, notify, base_cat_features))
        tables['cv_final_fit'] = split_table(final_frame)
        metadata['cv_fit_count'] = 4*len(tables['cv_folds'])+2
    metadata.update(selection=selected, feature_groups=feature_meta['groups'], status='validation_complete',
                    effective_model_params=models['Base']['estimator'].get_params())
    frame.to_parquet(output/'features.parquet', index=False)
    validation.to_parquet(output/'predictions_validation.parquet', index=False)
    save_models(output, models, model_key)
    comparisons = {'Validation / '+name: item for name, item in comparisons_for(validation, config).items()}
    predictions = {'Validation': validation}
    # REPORTING LOGIC: Persist computed tables before rendering; no historical or manually entered losses are used.
    metadata['artifacts'] = {'model_key': model_key, 'models': list(models), 'tables': list(tables)}
    save_tables(output, tables)
    write_json(receipt, metadata)
    run = ResearchRun(output, frame, predictions, tables, comparisons, metadata)
    render(run, config)
    notify('report', 1, 1, str(run.report))
    return finalize_test(output, progress=progress) if evaluate_test else run


def validate_features(tx, base_features, config, training, base_cat_features=None):
    # VALIDATION LOGIC: Guard direct leakage/collisions without claiming to audit user-supplied feature provenance.
    if tx.empty:
        raise ValueError('At least one valid transaction is required.')
    if not base_features or len(set(base_features)) != len(base_features) or set(base_features)-set(tx):
        raise ValueError('Provide a nonempty, unique list of existing base feature columns.')
    forbidden = {config.transactions.target, config.transactions.actual, config.transactions.id,
                 config.transactions.time, config.transactions.split, 'target', 'actual', 'row_id', 'time', 'split'}
    if set(base_features) & forbidden:
        raise ValueError('Target, actual outcome, identity, timestamp and split columns cannot be base features.')
    if base_cat_features is not None and (len(set(base_cat_features)) != len(base_cat_features) or set(base_cat_features)-set(base_features)):
        raise ValueError('base_cat_features must be a unique subset of base_features; [] explicitly means all numeric.')
    reserved = [c for c in tx if c.startswith(('bcq_', 'prediction_', 'cv_', '__')) or c in {'actual_level', 'prior_trade_count_30d', 'history_days_available', 'history_30d_complete'}]
    if reserved:
        raise ValueError(f'Rename source columns reserved for generated research outputs: {reserved}')
    if training.target_mode == 'delta' and (config.transactions.anchor is None or 'anchor' not in tx or tx.anchor.isna().any()):
        raise ValueError('Delta targets require a finite prediction-time anchor for every transaction.')


def manifest(config, training, key, feature_key, params, base, tx_audit, quote_audit, base_cat_features=None):
    # PROVENANCE LOGIC: Record the input audit and exact libraries, not a claim of generalization.
    versions = {name: version(name) for name in ('pandas', 'numpy', 'lightgbm', 'matplotlib')}
    restoration = 'predicted_delta + anchor - rollover_adjustment' if config.transactions.rollover_adjustment is not None else 'predicted_delta + anchor'
    reconstruction = dict(target_transform=f'supplied target * {config.target_scale:g}; no target recomputation',
                          target_convention='actual spread' if training.target_mode == 'level' else restoration.replace('predicted_delta + ', 'actual - ').replace(' - rollover_adjustment', ' + rollover_adjustment'),
                          prediction='predicted_level' if training.target_mode == 'level' else restoration,
                          actual='mapped actual column' if config.transactions.actual is not None else ('target' if training.target_mode == 'level' else restoration.replace('predicted_delta', 'target')),
                          omitted_rollover_adjustment=0, adjustment_available_at_prediction_time='caller responsibility')
    return dict(engine_version='0.3.0', run_key=key, feature_key=feature_key, synthetic=config.synthetic,
                config=config.to_dict(), training=training.to_dict(), base_features=list(base), model_params=params,
                base_cat_features=None if base_cat_features is None else list(base_cat_features),
                reconstruction=reconstruction,
                dependencies=versions, normalization={'transactions': tx_audit, 'quotes': quote_audit},
                test_status='not evaluated', test_exposure_history='unknown; caller must disclose prior use',
                status='running', limitations=['Base feature causality must be established by the caller.',
                'No price-to-spread conversion, currency conversion, or executable-size inference.',
                'No exchange cancellation/withdrawal protocol inferred from null bid/ask.',
                'In-memory pandas engine; use filtered Parquet for data exceeding available memory.',
                'History counts describe supplied data, not certified market-wide liquidity.'])


def split_table(frame):
    # CORE LOGIC: STEP 1 — Count rows and finite labels by the actual persisted split.
    # Input: split=['Train','Train','Validation'], target=[1,NaN,2].
    # Output: Train rows=2, finite_labels=1; Validation rows=1, finite_labels=1.
    # Explanation: Missing target records stay visible in diagnostics even though they cannot fit or score a model.
    # Trick: observed=True avoids inventing unused category stages.
    work = frame.assign(finite_label=np.isfinite(frame.target), date=frame.time.dt.date)
    return work.groupby('split', observed=True).agg(rows=('row_id', 'size'), finite_labels=('finite_label', 'sum'),
        bonds=('cusip', 'nunique'), dates=('date', 'nunique'), start=('time', 'min'), end=('time', 'max')).reset_index()


def save_models(output, models, key):
    # FILE IO LOGIC: Native boosters plus category schemas support reuse outside the research session.
    folder = output/'models'
    folder.mkdir(exist_ok=True)
    for index, (name, model) in enumerate(models.items()):
        stem = f'model_{index:02d}'
        model['estimator'].booster_.save_model(str(folder/(stem+'.txt')))
        write_json(folder/(stem+'.json'), {'name': name, 'features': model['features'], 'schema': model['schema']})
    # CACHEING LOGIC: This private local bundle also preserves sklearn estimator behavior for final evaluation.
    path = output/'models.pkl'
    payload = pickle.dumps({'run_key': key, 'models': models}, protocol=5)
    temporary = path.with_suffix('.tmp')
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    write_json(output/'model_bundle.json', {'sha256': hashlib.sha256(payload).hexdigest(), 'run_key': key})


def save_tables(output, tables):
    # FILE IO LOGIC: Exact calculated tables accompany the English HTML and PNG report.
    folder = output/'tables'
    folder.mkdir(exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(folder/(name+'.csv'), index=False)


def render(run, config):
    # REPORTING LOGIC: The renderer receives current computed results and explicit provenance only.
    write_report(run.output, run.tables, run.comparisons, run.metadata)


def load_run(output):
    # FILE IO LOGIC: Restore saved results for slicing/reporting, without raw data or fitting.
    output = Path(output).expanduser().resolve()
    metadata = json.loads((output/'manifest.json').read_text())
    if metadata['status'] == 'running':
        raise ValueError('Run is incomplete. Resume run_research with the same inputs/settings.')
    config = PipelineConfig.from_dict(metadata['config'])
    frame = pd.read_parquet(output/'features.parquet')
    tables = {}
    for name in metadata['artifacts']['tables']:
        try:
            tables[name] = pd.read_csv(output/'tables'/(name+'.csv'))
        except pd.errors.EmptyDataError:
            tables[name] = pd.DataFrame()
    predictions, comparisons = {}, {}
    for stage in ('Validation', 'Test'):
        path = output/('predictions_'+stage.lower()+'.parquet')
        if path.exists():
            predictions[stage] = pd.read_parquet(path)
            comparisons.update({stage+' / '+name: item for name, item in comparisons_for(predictions[stage], config).items()})
    # PROVENANCE LOGIC: A saved test is already exposed even if interruption prevented its final manifest commit.
    if 'Test' in predictions and not metadata['test_status'].startswith('evaluated once'):
        metadata.update(test_status='saved Test predictions already exposed; finalization receipt incomplete',
                        status='test_predictions_saved')
    return ResearchRun(output, frame, predictions, tables, comparisons, metadata)


def finalize_test(output, progress=None):
    # EVALUATION LOGIC: Test requires an existing validation decision and never ranks alternative candidates.
    notify = Progress(progress)
    run = load_run(output)
    settings = TrainingConfig(**run.metadata['training'])
    config = PipelineConfig.from_dict(run.metadata['config'])
    frozen = json.loads((run.output/'selection.json').read_text())
    if frozen != run.metadata['selection']:
        raise ValueError('Frozen selection and manifest disagree; do not modify a completed run.')
    # RECOVERY LOGIC: Validate the frozen selection before reusing test, then repair any interrupted final receipt.
    if 'Test' in run.predictions:
        run.metadata.update(test_status='evaluated once; selection fixed on validation', status='complete')
        write_json(run.output/'manifest.json', run.metadata)
        render(run, config)
        notify('test', 1, 1, 'Reusing the existing final test; no repeated fitting or new candidate search')
        return run
    if not run.features.split.eq('Test').any():
        raise ValueError('This run reserved no Test rows. Its report is validation-only.')
    # CACHEING LOGIC: Restore only this run's trusted model bundle; the run key must match.
    payload = (run.output/'models.pkl').read_bytes()
    receipt = json.loads((run.output/'model_bundle.json').read_text())
    if receipt['sha256'] != hashlib.sha256(payload).hexdigest() or receipt['run_key'] != run.metadata['run_key']:
        raise ValueError('Saved model bundle integrity check failed; do not evaluate an incomplete or changed artifact.')
    saved = pickle.loads(payload)
    if saved['run_key'] != run.metadata['run_key']:
        raise ValueError('Model artifacts belong to a different run.')
    models = {name: saved['models'][name] for name in ('Base', frozen['selected'])}
    notify('test', None, None, f'Evaluating frozen {frozen["selected"]} versus Base')
    actual_column = 'actual' if config.transactions.actual is not None else None
    adjustment_column = 'rollover_adjustment' if config.transactions.rollover_adjustment is not None else None
    predictions = predict_stage(run.features, models, 'Test', settings, actual_column, adjustment_column)
    # FILE IO LOGIC: Save one fixed test comparison and clearly mark that test is now exposed.
    temporary = run.output/'predictions_test.parquet.tmp'
    predictions.to_parquet(temporary, index=False)
    os.replace(temporary, run.output/'predictions_test.parquet')
    run.predictions['Test'] = predictions
    run.comparisons.update({'Test / '+name: item for name, item in comparisons_for(predictions, config).items()})
    run.metadata.update(test_status='evaluated once; selection fixed on validation', status='complete')
    write_json(run.output/'manifest.json', run.metadata)
    render(run, config)
    notify('report', 1, 1, str(run.report))
    return run
