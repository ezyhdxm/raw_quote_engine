"""Chronological out-of-fold quote experiments with one shared causal feature frame."""
# SETUP LOGIC: Importing validation utilities never fits an estimator or opens final Test.
import pandas as pd
from bond_pricer.walk_forward import WalkForwardConfig, walk_forward_splits
from bond_pricer.data import local_time
from .cache import digest
from .training import fit_models, predict_stage, comparisons_for


def development_frame(frame):
    # VALIDATION LOGIC: The outer chronological partition owns final Test and its preceding buffer.
    if 'split' not in frame or not frame.split.eq('Validation').any():
        raise ValueError('Walk-forward requires an outer Validation boundary before the reserved Test.')
    # CORE LOGIC: STEP 1 — Retain development history only through the last outer validation instant.
    # Input: dates=2026-01-01 through 2026-01-06, split=[Train,Embargo,Validation,Validation,Embargo,Test].
    # Output: dates=[2026-01-01,2026-01-02,2026-01-03,2026-01-04]; Jan05 buffer and Jan06 Test are excluded.
    # Explanation: Inner folds replace the original development split; earlier embargo rows become eligible history.
    # Trick: The outer split is chronological; the cutoff reads time and split, never a Test outcome.
    cutoff = frame.loc[frame.split.eq('Validation'), 'time'].max()
    return frame.loc[frame.time.le(cutoff)].copy()


def walk_forward_plan(frame, settings, timezone):
    # CONFIGURATION LOGIC: Avoid two independently configured final holdouts in the same experiment.
    settings = settings if isinstance(settings, WalkForwardConfig) else WalkForwardConfig(**settings)
    if settings.holdout_dates or settings.holdout_embargo_dates:
        raise ValueError('Raw engine reserves Test through TrainingConfig; walk-forward holdout fields must be zero.')
    development = development_frame(frame)
    folds = walk_forward_splits(development, 'time', settings, timezone=timezone)
    return development, folds


def _fold_frame(development, fold):
    # CORE LOGIC: STEP 1 — Assign one fold's own training and evaluation roles to positional subsets.
    # Input: source row_id=[9,3,7,5], train_positions=[1,3], validation_positions=[0,2].
    # Output: row_id=[3,5,9,7], split=[Train,Train,Validation,Validation].
    # Explanation: Other folds and their future labels cannot reach this fold's estimator.
    # Trick: Row identity survives concatenation; reset_index removes incidental duplicate pandas labels.
    train = development.iloc[fold.train_positions].assign(split='Train')
    validation = development.iloc[fold.validation_positions].assign(split='Validation')
    return pd.concat([train, validation], ignore_index=True)


def _fit_fold(development, fold, definitions, params, training, config, notify, categorical):
    # ORCHESTRATION LOGIC: Fresh models and categorical vocabularies are fitted exclusively within this fold.
    frame = _fold_frame(development, fold)
    def progress(stage, done, total, message):
        # UI LOGIC: Keep the current fold visible during each model's fitting progress.
        notify(f'cv{fold.fold_id}_{stage}', done, total, message)
    models = fit_models(frame, definitions, params, training, progress, categorical)
    actual = 'actual' if config.transactions.actual is not None else None
    adjustment = 'rollover_adjustment' if config.transactions.rollover_adjustment is not None else None
    predictions = predict_stage(frame, models, 'Validation', training, actual, adjustment)
    # CORE LOGIC: STEP 1 — Tag held-out predictions with their unique forecast origin.
    # Input: validation row_id=[9,7], fold.fold_id=2.
    # Output: row_id=[9,7], cv_fold=[2,2]; prediction values and source metadata are unchanged.
    # Explanation: The fold tag supports temporal diagnosis without fitting again.
    # Trick: cv_fold is a reserved output column so source metadata cannot be silently overwritten.
    return predictions.assign(cv_fold=fold.fold_id)


def final_origin(frame):
    # CORE LOGIC: STEP 1 — Locate when the frozen choice must exist before the final forecast begins.
    # Input: development ends Nov1 2026 12:00 New York, no Test; this local day contains a DST fallback.
    # Output: origin=Nov2 2026 00:00-05:00, not Nov1 23:00-05:00.
    # Explanation: With Test, its first local midnight is the origin; otherwise use midnight after development.
    # Trick: A calendar DateOffset preserves local midnight across both 23-hour and 25-hour days.
    test = frame.loc[frame.split.eq('Test'), 'time']
    if len(test):
        return test.min().normalize()
    return development_frame(frame).time.max().normalize()+pd.DateOffset(days=1)


def selection_visible_predictions(predictions, frame, settings, timezone):
    # CONFIGURATION LOGIC: With no release-time mapping, label availability remains an explicit caller assumption.
    if settings.label_available_column is None:
        return predictions
    # CORE LOGIC: STEP 1 — Hide outcomes unavailable when the validation choice must be frozen.
    # Input: origin=2026-01-11 00:00 UTC; releases=[2026-01-10 12:00 UTC,2026-01-11 00:00 UTC,NaT], actual_level=[100,200,300].
    # Output: cv_label_available=[True,False,False], actual_level=[100,NaN,NaN]; predictions are unchanged.
    # Explanation: A delayed validation label cannot influence selection merely because the retrospective file contains it.
    # Trick: Keep original target/source columns for provenance, but every comparison scores the masked actual_level.
    available = local_time(predictions[settings.label_available_column], timezone)
    eligible = available.notna() & available.lt(final_origin(frame))
    output = predictions.assign(cv_label_available=eligible)
    output['actual_level'] = output.actual_level.where(eligible)
    return output


def run_walk_forward(frame, definitions, params, training, config, settings, cache, key, notify, categorical):
    # CONFIGURATION LOGIC: Feature construction is already complete and time-causal; only estimators vary by fold.
    development, folds = walk_forward_plan(frame, settings, config.timezone)
    predictions, records = [], []
    # CACHEING LOGIC: Save each completed fold independently; an interruption repeats at most an incomplete fold.
    for fold in folds:
        notify('walk_forward', fold.fold_id-1, len(folds), f'Fold {fold.fold_id}/{len(folds)}')
        fold_key = digest(key, fold.to_dict())
        predicted = cache.get('step5_cv_fold', fold_key, lambda: _fit_fold(
            development, fold, definitions, params, training, config, notify, categorical))
        predictions.append(predicted)
        records.append(fold.to_dict())
    # CORE LOGIC: STEP 1 — Pool nonoverlapping out-of-fold records in chronological order.
    # Input: fold1 row_id=[3,4], fold2 row_id=[5,6], with increasing prediction times.
    # Output: row_id=[3,4,5,6], cv_fold=[1,1,2,2]; each transaction contributes once.
    # Explanation: Candidate selection uses individual held-out errors, not an average of fold gain percentages.
    # Trick: A duplicate row_id rejects the experiment rather than giving repeated predictions extra weight.
    pooled = pd.concat(predictions, ignore_index=True).sort_values('time', kind='stable').reset_index(drop=True)
    if pooled.row_id.duplicated().any():
        raise ValueError('Walk-forward validation windows must not score a transaction more than once.')
    # EVALUATION LOGIC: Fold training and pooled selection both respect supplied label release times.
    pooled = selection_visible_predictions(pooled, frame, settings, config.timezone)
    # REPORTING LOGIC: Every per-fold metric is calculated from the same saved prediction records.
    metrics = []
    for _, predicted in pooled.groupby('cv_fold', sort=True):
        for name, comparison in comparisons_for(predicted, config).items():
            record = comparison.summary(config.selection_min_count).iloc[0].to_dict()
            metrics.append(dict(fold_id=int(predicted.cv_fold.iloc[0]), model=name, **record))
    return pooled, dict(cv_folds=pd.DataFrame(records), cv_metrics=pd.DataFrame(metrics),
                        cv_stability=fold_stability(pd.DataFrame(metrics)))


def fold_stability(metrics):
    # CORE LOGIC: STEP 1 — Measure how consistently a candidate improves across the declared folds.
    # Input: model=['Q','Q'], mae_delta=[-1,2], mae_improvement_pct=[10,-20].
    # Output: Q folds=2, assessable_folds=2, improved_folds=1, improved_fold_pct=50, worst_fold_gain_pct=-20, mean_fold_mae_delta=.5.
    # Explanation: These equal-fold summaries accompany the separately reported record-weighted pooled losses.
    # Trick: Nonfinite gain percentages remain missing; zero BASE loss does not imply infinite improvement.
    frame = metrics.assign(improved=metrics.mae_delta.lt(0))
    grouped = frame.groupby('model', sort=False, observed=True)
    result = grouped.agg(folds=('fold_id', 'size'), assessable_folds=('mae_delta', 'count'), improved_folds=('improved', 'sum'),
        worst_fold_gain_pct=('mae_improvement_pct', 'min'), mean_fold_mae_delta=('mae_delta', 'mean'))
    result['improved_fold_pct'] = result.improved_folds/result.assessable_folds.where(result.assessable_folds.gt(0))*100
    return result.reset_index()


def final_training_frame(frame, settings, timezone='UTC'):
    # CORE LOGIC: STEP 1 — Refit the frozen model pair on eligible development history, with the same rolling cap.
    # Input: development=2026-01-01 through 2026-01-08, max_train_dates=3; Test starts Jan10 after buffer Jan09.
    # Output: dates=[2026-01-06,2026-01-07,2026-01-08], split=[Train,Train,Train]; Jan09/Jan10 are excluded.
    # Explanation: This is a new fit after candidate selection, not a reuse of the last fold's estimator.
    # Trick: None expands through all development dates; observed-date membership retains every trade on kept dates.
    development = development_frame(frame)
    days = development.time.dt.normalize()
    dates = sorted(days.unique())
    if settings.max_train_dates is not None:
        development = development.loc[days.isin(dates[-settings.max_train_dates:])]
    # CORE LOGIC: STEP 2 — Purge labels unavailable at the final forecast origin when a release-time column is supplied.
    # Input: Test starts 2026-01-10 UTC; releases=[2026-01-09 12:00 UTC,2026-01-10 00:00 UTC,NaT].
    # Output: only the first training record remains; all retained rows have split=Train.
    # Explanation: A label released exactly at the Test date boundary is not treated as known before fitting.
    # Trick: Without Test, origin is next local midnight after development; DateOffset respects daylight-saving days.
    if settings.label_available_column is not None:
        available = local_time(development[settings.label_available_column], timezone)
        development = development.loc[available.notna() & available.lt(final_origin(frame))]
    return development.assign(split='Train')
