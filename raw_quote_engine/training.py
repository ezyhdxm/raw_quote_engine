"""Fixed chronological experiments, validation selection and an explicitly enabled final test."""
# SETUP LOGIC: Model fitting is invoked only by the pipeline, never on import.
from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from bond_pricer import compare_predictions


@dataclass(frozen=True)
class TrainingConfig:
    # CONFIGURATION LOGIC: Fractions count observed local dates; embargo dates are discarded between stages.
    validation_fraction: float = .2
    test_fraction: float = .2
    validation_dates: int = None
    test_dates: int = None
    embargo_dates: int = 1
    test_embargo_dates: int = None
    min_train_dates: int = 3
    target_mode: str = 'level'
    selection_slice: str = 'large_long'
    min_selection_rows: int = 30
    useful_improvement_pct: float = 5.0
    seed: int = 2026
    category_order: str = 'sorted'
    apply_model_defaults: bool = False

    def __post_init__(self):
        # VALIDATION LOGIC: Fail before expensive feature calculation on impossible experiment settings.
        for name in ('embargo_dates', 'min_train_dates', 'min_selection_rows', 'seed'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f'{name} must be a nonnegative integer.')
        for name in ('validation_dates', 'test_dates', 'test_embargo_dates'):
            value = getattr(self, name)
            minimum = 0 if name == 'test_embargo_dates' else 1
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < minimum):
                raise ValueError(f'{name} must be an integer >= {minimum}, or None.')
        if not np.isfinite(self.useful_improvement_pct) or self.useful_improvement_pct < 0:
            raise ValueError('useful_improvement_pct must be finite and nonnegative.')
        if not 0 < self.validation_fraction < 1 or not 0 <= self.test_fraction < 1:
            raise ValueError('Validation fraction must be in (0,1); test fraction in [0,1).')
        if self.validation_dates is None and self.test_dates is None and self.validation_fraction+self.test_fraction >= 1:
            raise ValueError('Leave a positive training fraction.')
        if self.embargo_dates < 0 or self.min_train_dates < 1 or self.min_selection_rows < 1:
            raise ValueError('Use nonnegative embargo dates and positive support thresholds.')
        if self.target_mode not in {'level', 'delta'}:
            raise ValueError('target_mode must be level or delta (requires prediction-time anchor).')
        if self.selection_slice not in {'large_long', 'large', 'all'}:
            raise ValueError('selection_slice must be large_long, large or all.')
        if self.category_order not in {'sorted', 'appearance'}:
            raise ValueError('category_order must be sorted or appearance.')
        if not isinstance(self.apply_model_defaults, bool):
            raise ValueError('apply_model_defaults must be a boolean.')

    def to_dict(self):
        # SERIALIZATION LOGIC: Freeze the entire experiment definition in the report manifest.
        return asdict(self)


def assign_splits(tx, settings, *, use_supplied_split=True):
    # VALIDATION LOGIC: Engine callers enable source labels only through an explicit split-column mapping.
    if not isinstance(use_supplied_split, bool):
        raise ValueError('use_supplied_split must be a boolean.')
    # CORE LOGIC: STEP 1 — Use declared labels, or allocate dates despite an unrelated source column named split.
    # Input: 10 ordered dates; split=['Train']*8+['Validation']*2; fractions=.2/.2; embargo=0; use_supplied_split=False.
    # Output: split=['Train']*6+['Validation']*2+['Test']*2; the caller's original split column is unchanged.
    # Explanation: Explicitly disabled source labels cannot override the requested date policy.
    # Trick: With use_supplied_split=True and a split column, labels are retained; copied frames isolate mutations.
    frame = tx.copy()
    if not use_supplied_split or 'split' not in frame:
        frame = automatic_splits(frame, settings)
    # VALIDATION LOGIC: Disallow interleaved stages and equal-time leakage across stage boundaries.
    allowed = {'Train', 'Validation', 'Test', 'Embargo'}
    if frame.split.isna().any() or not set(frame.split).issubset(allowed):
        raise ValueError(f'split labels must be nonmissing and in {sorted(allowed)}.')
    for stage in ('Train', 'Validation'):
        if not frame.split.eq(stage).any():
            raise ValueError(f'No {stage} rows; provide more history or an explicit chronological split.')
    windows = [frame.loc[frame.split.eq(stage), 'time'] for stage in ('Train', 'Validation', 'Test')]
    windows = [values for values in windows if len(values)]
    if any(left.max() >= right.min() for left, right in zip(windows, windows[1:])):
        raise ValueError('Train, Validation and Test must have strictly nonoverlapping chronological times.')
    return frame


def automatic_splits(frame, settings):
    # CORE LOGIC: STEP 1 — Allocate held-out date blocks and intervening embargo blocks.
    # Input: ten dates, validation_dates=2, test_dates=2, embargo_dates=1, test_embargo_dates=0.
    # Output: Train dates1-5, Embargo6, Validation7-8, Test9-10.
    # Explanation: Explicit date counts override fractions; test_embargo_dates overrides only the gap before test.
    # Trick: Counts/fractions use observed dates, not rows; omitted test embargo inherits embargo_dates.
    days = frame.time.dt.normalize()
    dates = sorted(days.unique())
    n_val = settings.validation_dates or max(1, int(np.ceil(len(dates)*settings.validation_fraction)))
    n_test = settings.test_dates or (max(1, int(np.ceil(len(dates)*settings.test_fraction))) if settings.test_fraction else 0)
    test_gap = settings.embargo_dates if settings.test_embargo_dates is None else settings.test_embargo_dates
    test_start = len(dates)-n_test
    val_end = test_start-test_gap if n_test else len(dates)
    val_start = val_end-n_val
    train_end = val_start-settings.embargo_dates
    # VALIDATION LOGIC: Short datasets need an explicit feasible policy, never a randomized fallback.
    if train_end < settings.min_train_dates:
        raise ValueError('Too few dates for train/embargo/validation/test. Supply more history, reduce fractions/embargo, '
                         'reduce date counts, set test_fraction=0 without test_dates, or provide an explicit chronological split column.')
    # CORE LOGIC: STEP 2 — Assign each transaction by its local date while preserving input order.
    # Input: dates1-10 allocation above; input transaction dates=[9,2,6,5].
    # Output: split=['Test','Train','Embargo','Train'] in the same input order.
    # Explanation: Date membership labels every row; a busy date never straddles model stages.
    # Trick: Features may use causal historical observations from embargo dates, but their labels never train a model.
    labels = pd.Series('Embargo', index=frame.index)
    labels.loc[days.isin(dates[:train_end])] = 'Train'
    labels.loc[days.isin(dates[val_start:val_end])] = 'Validation'
    if n_test:
        labels.loc[days.isin(dates[test_start:])] = 'Test'
    return frame.assign(split=labels)


def category_values(values, order):
    # CORE LOGIC: STEP 1 — Freeze a category vocabulary in the declared training-only order.
    # Input: values=['Z','A',None,'Z'], order='appearance'.
    # Output: ['Z','A']; order='sorted' produces ['A','Z'] for the same input.
    # Explanation: Drop missing values, stringify observed labels and remove repeated labels before ordering.
    # Trick: unique preserves first occurrence; string conversion matches the later encoding contract.
    categories = values.dropna().astype(str).unique().tolist()
    return sorted(categories) if order == 'sorted' else categories


def prepare_schema(train, columns, category_order='sorted', categorical_features=None):
    # VALIDATION LOGIC: Explicit features cannot directly contain the target or generated predictions/stages.
    if not columns or len(set(columns)) != len(columns) or any(c not in train for c in columns):
        raise ValueError('Base/model features must be a nonempty, unique list of existing columns.')
    if set(columns) & {'target', 'actual', 'split', 'row_id', 'time', 'actual_level'}:
        raise ValueError('Target, identity, time and split columns cannot be model features.')
    if category_order not in {'sorted', 'appearance'}:
        raise ValueError('category_order must be sorted or appearance.')
    if categorical_features is not None and (len(set(categorical_features)) != len(categorical_features) or set(categorical_features)-set(columns)):
        raise ValueError('Categorical features must be a unique subset of the supplied feature columns.')
    if any(pd.api.types.is_datetime64_any_dtype(train[column]) for column in columns):
        raise ValueError('Convert timestamp features to explicitly causal numeric features before running the engine.')
    # CORE LOGIC: STEP 1 — Learn categorical vocabularies using training rows only.
    # Input: train side=[2,1,None], amount=['5','bad','7']; categorical_features=['side'], category_order='sorted'.
    # Output: side={'kind':'category','categories':['1.0','2.0']}, amount={'kind':'numeric'}.
    # Explanation: Numeric side codes are categorical by request; amount is numeric and 'bad' becomes NaN when encoded.
    # Trick: [] forces every feature numeric; None retains dtype inference. Unseen/missing categories map to missing at prediction.
    schema = {}
    for column in columns:
        values = train[column]
        categorical = column in categorical_features if categorical_features is not None else not pd.api.types.is_numeric_dtype(values)
        if categorical:
            schema[column] = {'kind': 'category', 'categories': category_values(values, category_order)}
        else:
            schema[column] = {'kind': 'numeric'}
    return schema


# SETUP LOGIC: Encoding applies an already learned schema without reading target labels.
def encode(frame, schema):
    # CORE LOGIC: STEP 1 — Apply the fixed training schema without imputation learned from evaluation data.
    # Input: x=[1,inf], sector=['Bank','TMT']; train categories=['Bank','Energy'].
    # Output: x=[1,NaN], sector categorical codes=[0,-1].
    # Explanation: Infinite numeric values and unseen categories become LightGBM missing inputs.
    # Trick: Build columns together to avoid fragmented wide frames; input indices and category codes stay fixed.
    values_by_column = {}
    for column, specification in schema.items():
        values = frame[column]
        if specification['kind'] == 'numeric':
            values_by_column[column] = pd.to_numeric(values, errors='coerce').replace([np.inf, -np.inf], np.nan).astype(float)
        else:
            values_by_column[column] = pd.Categorical(values.astype('string'), categories=specification['categories'])
    result = pd.DataFrame(values_by_column, index=frame.index)
    # CORE LOGIC: STEP 2 — Use stable internal names so arbitrary user feature labels are accepted by LightGBM.
    # Input: ordered feature columns=['quote:level','size[USD]'].
    # Output: estimator input columns=['feature_0000','feature_0001']; values/order are unchanged.
    # Explanation: Original labels stay in the saved schema while JSON-special punctuation never reaches LightGBM.
    # Trick: Prediction applies the same ordered schema, so encoding never changes a feature's position.
    result.columns = [f'feature_{index:04d}' for index in range(len(result.columns))]
    return result


def variants(base_features, groups):
    # CORE LOGIC: STEP 1 — Nest fixed quote ablations while keeping the original base feature order.
    # Input: base=['x']; groups={'Quote':['q1'],'Quote+Path':['q1','q2']}.
    # Output: {'Base':['x'],'Quote':['x','q1'],'Quote+Path':['x','q1','q2']}.
    # Explanation: Each candidate adds only its declared family; duplicate names are removed stably.
    # Trick: dict.fromkeys preserves first occurrence; variants are not chosen from test results.
    result = {'Base': list(base_features)}
    for name, extra in groups.items():
        result[name] = list(dict.fromkeys(list(base_features)+extra))
    return result


def fit_models(frame, definitions, params, settings, progress, base_cat_features=None):
    # CORE LOGIC: STEP 1 — Fit every candidate on exactly the same finite-label training rows.
    # Input: stages=['Train','Train','Validation'], target=[1,NaN,5].
    # Output: training row positions=[0]; validation target5 is never passed to fit.
    # Explanation: All candidates share this one label mask, including transactions without quotes.
    # Trick: Base missing values stay in the training population; LightGBM handles them natively.
    mask = frame.split.eq('Train') & np.isfinite(frame.target)
    train = frame.loc[mask]
    if len(train) < 2:
        raise ValueError('At least two finite-label training records are required.')
    # MODELING LOGIC: Fixed user parameters; no early stopping, tuning or test-dependent model count.
    defaults = dict(n_estimators=200, learning_rate=.05, num_leaves=31, random_state=settings.seed,
                    n_jobs=-1, verbosity=-1, deterministic=True)
    if not settings.apply_model_defaults:
        defaults = {}
    defaults.update(params or {})
    models = {}
    for index, (name, columns) in enumerate(definitions.items()):
        progress('models', index, len(definitions), f'Fitting {name} on {len(train):,} training records')
        schema = prepare_schema(train, columns, settings.category_order, base_cat_features)
        estimator = LGBMRegressor(**defaults)
        estimator.fit(encode(train, schema), train.target)
        models[name] = {'estimator': estimator, 'schema': schema, 'features': columns}
        progress('models', index+1, len(definitions), f'Finished {name}')
    return models


def predict_stage(frame, models, stage, settings, actual_column=None, adjustment_column=None):
    # VALIDATION LOGIC: A mapped adjustment has one explicit algebraic convention, supported only for delta targets.
    if adjustment_column is not None and settings.target_mode != 'delta':
        raise ValueError('rollover_adjustment requires target_mode="delta"; level targets need no anchor restoration.')
    if adjustment_column is not None and not np.isfinite(frame[adjustment_column]).all():
        raise ValueError('Configured rollover_adjustment must be finite for every transaction.')
    # CORE LOGIC: STEP 1 — Select one declared evaluation stage, retaining invalid-label rows for coverage.
    # Input: Validation row target=3, anchor=100, rollover_adjustment=2, actual=101; delta mode and both mappings supplied.
    # Output: one evaluation row with actual_level=101; without actual mapping 3+100-2 also gives 101.
    # Explanation: User-supplied target means actual-anchor+adjustment; explicit actual remains authoritative.
    # Trick: An omitted adjustment mapping means zero even if an unrelated source column is named rollover_adjustment.
    output = frame.loc[frame.split.eq(stage)].copy()
    adjustment = output[adjustment_column] if adjustment_column is not None else 0.
    output['actual_level'] = output.target
    if settings.target_mode == 'delta':
        output['actual_level'] = output.target+output.anchor-adjustment
    if actual_column is not None:
        output['actual_level'] = output[actual_column]
    # CORE LOGIC: STEP 2 — Restore a predicted delta to spread units using the declared benchmark adjustment.
    # Input: estimator predicts delta=4, anchor=100, rollover_adjustment=2, target_mode='delta'.
    # Output: prediction=102; with no adjustment mapping the same delta and anchor yield 104.
    # Explanation: Add the prior anchor and subtract the rollover offset used when the user constructed the target.
    # Trick: No current realized outcome enters inference; estimators see only their fixed training schema.
    for name, model in models.items():
        output['prediction_'+name] = model['estimator'].predict(encode(output, model['schema']))
        if settings.target_mode == 'delta':
            output['prediction_'+name] += output.anchor-adjustment
    return output


def comparisons_for(predictions, config):
    # REPORTING LOGIC: Common-sample comparisons use the same scale, identity and local-time convention.
    result = {}
    for column in [c for c in predictions if c.startswith('prediction_') and c != 'prediction_Base']:
        name = column.removeprefix('prediction_')
        result[name] = compare_predictions(predictions, 'actual_level', 'prediction_Base', column,
            reference_name='Base', candidate_name=name, id_column='row_id', time_column='time',
            bond_column='cusip', error_scale=config.error_scale, unit=config.unit, timezone=config.timezone)
    return result


def selection_population(predictions, settings, config=None):
    # CONFIGURATION LOGIC: In engine runs, optional metadata contributes only when explicitly mapped.
    use_quantity = config is None or config.transactions.quantity is not None
    use_maturity = config is None or config.transactions.maturity_years is not None or config.transactions.maturity_date is not None
    # CORE LOGIC: STEP 1 — Apply the predeclared business cohort, falling back explicitly if metadata is absent.
    # Input: quantity=[2MM,500K,1MM], maturity=[2,3,.5], selection_slice='large_long'.
    # Output: mask=[True,False,False], description='quantity>=1MM and maturity>1y'.
    # Explanation: The boundary at exactly 1MM is included; <=1y is reported separately.
    # Trick: This condition is defined before evaluating losses; it is not the best-looking slice search.
    mask = pd.Series(True, index=predictions.index)
    description = 'all transactions'
    if settings.selection_slice in {'large', 'large_long'} and use_quantity and 'QUANTITY' in predictions:
        mask &= predictions.QUANTITY.ge(1e6)
        description = 'quantity>=1MM'
        if settings.selection_slice == 'large_long' and use_maturity and 'MATURITY_YEARS' in predictions:
            mask &= predictions.MATURITY_YEARS.gt(1)
            description += ' and maturity>1y'
    return mask, description


def select_candidate(predictions, config, settings):
    # CORE LOGIC: STEP 1 — Require common finite predictions across all validation candidates for selection.
    # Input: targets=[1,2,3], Base=[1,2,3], Quote=[1,NaN,2], cohort=[True,True,True].
    # Output: common=[True,False,True]; selection candidates are scored on rows1/3 only.
    # Explanation: A model cannot win selection by failing to predict difficult rows.
    # Trick: A sparse priority cohort falls back to all common validation rows, with the reason recorded.
    columns = [c for c in predictions if c.startswith('prediction_')]+['actual_level']
    common = np.isfinite(predictions[columns]).all(axis=1)
    # CORE LOGIC: STEP 2 — Exclude a row from every selection score when any scaled model error overflows.
    # Input: target=1e308, Base=-1e308, Quote=1e308; error_scale=1.
    # Output: common=False for this row, despite all three supplied numbers being finite.
    # Explanation: The Base error overflows; letting only Quote score this row would change the comparison population.
    # Trick: errstate suppresses expected overflow warnings; nonfinite error masks are applied jointly.
    with np.errstate(over='ignore', invalid='ignore'):
        for column in columns[:-1]:
            error = (predictions[column]-predictions.actual_level)*config.error_scale
            common &= np.isfinite(error)
    # CORE LOGIC: STEP 3 — Use the fixed priority population when its common support is sufficient.
    # Input: common=[True,True,True], cohort=[True,False,False], min_selection_rows=2.
    # Output: fallback=True, chosen=[True,True,True].
    # Explanation: One priority record is below minimum support, so all three common validation rows are used.
    # Trick: Fallback is support-based, never based on whether its error gain looks better.
    cohort, description = selection_population(predictions, settings, config)
    chosen = common & cohort
    fallback = int(chosen.sum()) < settings.min_selection_rows
    if fallback:
        chosen = common
    if not chosen.any():
        raise ValueError('No common finite validation predictions; cannot select a candidate.')
    # CORE LOGIC: STEP 4 — Select the lowest-MAE quote candidate and freeze its identity before opening test.
    # Input: Base errors=[2,2], Quote=[1,2], Quote+Path=[1,1], error_scale=1.
    # Output: selected='Quote+Path', Base MAE=2, candidate MAE=1, improvement_pct=50.
    # Explanation: Selection compares fixed candidates on one cohort; equal scores prefer the earlier simpler variant.
    # Trick: Base remains a benchmark even when the best quote candidate is worse; adoption is a separate decision.
    losses = {}
    actual = predictions.loc[chosen, 'actual_level']
    for column in columns[:-1]:
        losses[column.removeprefix('prediction_')] = stable_mae((predictions.loc[chosen, column]-actual)*config.error_scale)
    selected = min((name for name in losses if name != 'Base'), key=losses.get)
    gain = 100*(losses['Base']-losses[selected])/losses['Base'] if losses['Base'] else None
    # REPORTING LOGIC: Preserve the original requested cohort even when a support fallback was necessary.
    return dict(selected=selected, requested_slice=settings.selection_slice, effective_slice='all common rows' if fallback else description,
                cohort_before_fallback=description, support_fallback=fallback, rows=int(chosen.sum()), validation_mae=losses,
                improvement_pct=gain, threshold_pct=settings.useful_improvement_pct,
                threshold_met=gain is not None and gain >= settings.useful_improvement_pct,
                test_used_for_selection=False, fit_policy='Train only; same fitted estimators for validation and final test')


# SETUP LOGIC: A stable scalar loss helper supports the same finite-error population as the comparison engine.
def stable_mae(error):
    # CORE LOGIC: STEP 1 — Normalize before averaging to avoid overflowing a sum of large finite errors.
    # Input: errors=[1e308,-1e308].
    # Output: MAE=1e308, a finite value.
    # Explanation: Scale to [1,1], average to 1, then multiply by 1e308.
    # Trick: A zero maximum means every error is exactly zero and avoids division by zero.
    values = np.abs(np.asarray(error, dtype=float))
    scale = values.max()
    return float(np.mean(values/scale)*scale) if scale else 0.0
