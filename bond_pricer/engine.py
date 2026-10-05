"""Public paired-model API for bond spread, price or other regression targets."""
# SETUP LOGIC: Generic diagnostics do not import BondCliQ's training or state machinery.
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .data import paired_rows, read_data
from .metrics import metrics, grouped_metrics, date_sensitivity
from .slices import Slice, slice_labels, default_slices


class Comparison:
    # SETUP LOGIC: Retain source inputs and explicit configuration for reproducible filtered reviews.
    def __init__(self, data, rows, coverage, config):
        self.data, self.rows, self.coverage, self.config = data, rows, coverage, config
        self.reference_name, self.candidate_name = config['reference_name'], config['candidate_name']
        self.unit, self.tolerance = config['unit'], config['tolerance']
        self.default_slices = [spec for spec in default_slices(rows.columns)
                               if config['time_column'] is not None or spec.column not in {'__date', '__hour'}]
        self.filter_history = []

    def summary(self, min_count=30):
        # REPORTING LOGIC: Reuse the same metric definitions at every aggregation level.
        return pd.DataFrame([metrics(self.rows, tolerance=self.tolerance, min_count=min_count)])

    def slice(self, column, bins=None, labels=None, min_count=30, top_n=20, right=True):
        # CONFIGURATION LOGIC: Accept either a reusable Slice or simple keyword arguments.
        spec = column if isinstance(column, Slice) else Slice(column, bins, labels, right, top_n)
        # CORE LOGIC: STEP 1 — Attach groups to paired records, then recompute per-group losses.
        # Input: sector=['A','B'], reference abs=[2,4], candidate abs=[1,2].
        # Output: group A has n=1, MAE=(2,1); group B has n=1, MAE=(4,2).
        # Explanation: Slice membership is defined from data columns, independently of prediction gains.
        # Trick: Copying preserves the shared comparison rows when the user changes slice controls.
        rows = self.rows.assign(group=slice_labels(self.rows, spec))
        return grouped_metrics(rows, ['group'], tolerance=self.tolerance, min_count=min_count)

    def cross_slice(self, x, y, x_bins=None, y_bins=None, min_count=30, top_n=12,
                    x_right=True, y_right=True):
        # CONFIGURATION LOGIC: Both axes may use distinct fixed bins or categorical limits.
        sx = x if isinstance(x, Slice) else Slice(x, x_bins, right=x_right, top_n=top_n)
        sy = y if isinstance(y, Slice) else Slice(y, y_bins, right=y_right, top_n=top_n)
        if sx.column == sy.column:
            raise ValueError('Choose two different slice columns.')
        # CORE LOGIC: STEP 1 — Compute the intersection on records, not on pre-averaged one-way metrics.
        # Input: sector=['A','A','B'], size=['big','small','big'], ref_abs=[2,4,6], cand_abs=[1,3,2].
        # Output: (A,big): MAE=(2,1), n=1; (A,small): (4,3), n=1; (B,big): (6,2), n=1.
        # Explanation: Each combination gets its actual trades; (B,small) has no observations.
        # Trick: Empty heatmap cells remain unavailable, rather than appearing as perfect zero error.
        rows = self.rows.assign(x=slice_labels(self.rows, sx), y=slice_labels(self.rows, sy))
        return grouped_metrics(rows, ['x','y'], tolerance=self.tolerance, min_count=min_count)

    def daily(self, min_count=30):
        # CORE LOGIC: STEP 1 — Group ISO dates chronologically, retaining unknown dates separately.
        # Input: date=['2026-04-02','2026-04-01','2026-04-02'], ref_abs=[3,1,5], cand_abs=[2,0,4].
        # Output: Apr 1 has n=1, MAE=(1,0); Apr 2 has n=2, MAE=(4,3), in this date order.
        # Explanation: ISO date strings sort chronologically, regardless of each day's trade count.
        # Trick: Date plots never reuse the frequency-ranked top-category display or collapse dates into Other.
        rows = self.rows.assign(group=self.rows['__date'].fillna('Unknown date'))
        return grouped_metrics(rows,['group'],tolerance=self.tolerance,min_count=min_count)

    def stability(self):
        # REPORTING LOGIC: Return descriptive leave-one-date-out sensitivity without a training call.
        return date_sensitivity(self.rows)

    def filter(self, column, *, minimum=None, maximum=None, values=None, include_missing=False,
               minimum_inclusive=True, maximum_inclusive=True):
        # VALIDATION LOGIC: Filters are explicit conditions, never evaluated as Python expressions.
        if column not in self.data:
            raise ValueError(f'Unknown filter column: {column!r}')
        if values is not None and (minimum is not None or maximum is not None):
            raise ValueError('Use either category values or numerical boundaries in one filter.')
        # CORE LOGIC: STEP 1 — Filter the original population, then recalculate paired coverage and losses.
        # Input: QUANTITY=[500000,1000000,2000000,None], minimum=1000000, include_missing=False.
        # Output: retained QUANTITY=[1000000,2000000]; total and paired counts are recalculated on 2 rows.
        # Explanation: Inclusive lower bounds retain the exact 1MM threshold; missing quantity is excluded explicitly.
        # Trick: Chained filters form intersections; prediction values are reused and no models are fitted.
        source = self.data[column]
        number = pd.to_numeric(source, errors='coerce').replace([np.inf,-np.inf], np.nan)
        mask = source.notna() if values is None else source.astype('string').isin([str(v) for v in values])
        if minimum is not None:
            mask &= number.ge(minimum) if minimum_inclusive else number.gt(minimum)
        if maximum is not None:
            mask &= number.le(maximum) if maximum_inclusive else number.lt(maximum)
        mask |= source.isna() & include_missing
        result = compare_predictions(self.data.loc[mask], **self.config)
        # REPORTING LOGIC: Preserve population conditions in every exported review and chained filter.
        condition = dict(column=column,minimum=minimum,maximum=maximum,values=values,include_missing=include_missing,
                         minimum_inclusive=minimum_inclusive,maximum_inclusive=maximum_inclusive)
        result.filter_history = self.filter_history+[condition]
        return result

    def cases(self, limit=20, order='harm'):
        # VALIDATION LOGIC: Case ranking is explanatory, not a cleaning or model-selection rule.
        if limit < 1 or order not in {'harm','help','absolute'}:
            raise ValueError('Use a positive limit and order=harm, help or absolute.')
        # CORE LOGIC: STEP 1 — Rank actual paired trades by their loss change or candidate error.
        # Input: ids=[1,2,3], ref_abs=[2,4,1], cand_abs=[1,8,1], order='harm', limit=2.
        # Output: ids=[2,3], loss_change=[4,0].
        # Explanation: Trade 2 adds four units of error, trade 3 ties, and trade 1 improves.
        # Trick: A stable sort preserves source order among ties; these examples do not estimate prevalence.
        rows = self.rows.assign(loss_change=self.rows['__ae_candidate']-self.rows['__ae_reference'])
        key = '__ae_candidate' if order == 'absolute' else 'loss_change'
        return rows.sort_values(key, ascending=order=='help', kind='stable').head(limit)

    def missingness(self):
        # CORE LOGIC: STEP 1 — Count missing metadata/features in the complete supplied evaluation population.
        # Input: sector=['A',None], feature=[1,2].
        # Output: sector missing_n=1, missing_pct=50; feature missing_n=0, missing_pct=0.
        # Explanation: Coverage diagnosis includes rows excluded from the paired error comparison.
        # Trick: This table measures pandas missing values; invalid target/prediction strings are separately audited.
        counts = self.data.isna().sum()
        return pd.DataFrame({'column':counts.index,'missing_n':counts.values,'missing_pct':counts.values/max(len(self.data),1)*100})

    def export(self, folder, slices=None, interactions=None, min_count=30, metric='mae_delta'):
        # FILE IO LOGIC: Export a new immutable review directory; never overwrite model artifacts.
        from .report import export_comparison
        return export_comparison(self, folder, slices, interactions, min_count, metric)


def compare_predictions(data, actual, reference, candidate, *, reference_name=None, candidate_name=None,
                        id_column=None, time_column=None, bond_column=None, error_scale=1.0, unit='bps',
                        timezone='America/New_York', tolerance=1.0, reference_anchor=None, candidate_anchor=None):
    """Compare two prediction columns on a common sample; no model training or automatic split."""
    # VALIDATION LOGIC: Empty comparisons are represented explicitly; invalid configuration is an error.
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('tolerance must be a nonnegative finite value in the displayed error unit.')
    frame = read_data(data, string_columns=[id_column, bond_column])
    reference_name, candidate_name = reference_name or reference, candidate_name or candidate
    if reference_name == candidate_name:
        raise ValueError('Use two distinct model display names.')
    config = dict(actual=actual, reference=reference, candidate=candidate, reference_name=reference_name,
                  candidate_name=candidate_name, id_column=id_column, time_column=time_column,
                  bond_column=bond_column, error_scale=error_scale, unit=unit, timezone=timezone,
                  tolerance=tolerance, reference_anchor=reference_anchor, candidate_anchor=candidate_anchor)
    # DATA PREPARATION LOGIC: One shared adapter supplies all subsequent plots, slices and exports.
    options = {k:v for k,v in config.items() if k not in {'reference_name','candidate_name','unit','tolerance'}}
    rows, coverage = paired_rows(frame, **options)
    return Comparison(frame, rows, coverage, config)


@dataclass
class Model:
    # CONFIGURATION LOGIC: Each estimator receives its own explicit ordered feature list.
    name: str
    estimator: object
    features: list
    anchor: str = None


def _predict(frame, model):
    # VALIDATION LOGIC: Require explicit feature names and a regression-style one-value-per-row result.
    if not model.features or len(set(model.features)) != len(model.features):
        raise ValueError('Each model needs a nonempty, unique, ordered feature list.')
    missing = set(model.features)-set(frame)
    if missing:
        raise ValueError(f'{model.name} is missing features: {sorted(missing)}')
    # INFERENCE LOGIC: Invoke predict only; never fit, tune, reorder columns or impute model inputs.
    prediction = model.estimator.predict(frame.loc[:,model.features])
    if isinstance(prediction, (pd.Series,pd.DataFrame)) and not prediction.index.equals(frame.index):
        raise ValueError(f'{model.name} returned a pandas result with a different row index.')
    values = np.asarray(prediction)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:,0]
    if values.ndim != 1 or len(values) != len(frame):
        raise ValueError(f'{model.name}.predict must return one regression prediction per input row.')
    return values


def compare_models(data, actual, reference, candidate, **kwargs):
    """Run predict on two already-fitted Model objects with different feature sets if needed."""
    # VALIDATION LOGIC: Prediction metadata must not conflict with caller-supplied comparison arguments.
    frame = read_data(data, string_columns=[kwargs.get('id_column'), kwargs.get('bond_column')])
    columns = ['Reference prediction','Candidate prediction']
    if set(columns) & set(frame):
        raise ValueError('Rename existing Reference prediction/Candidate prediction columns first.')
    if not isinstance(reference, Model) or not isinstance(candidate, Model):
        raise TypeError('Wrap both fitted estimators in Model(name, estimator, features, anchor=None).')
    # INFERENCE LOGIC: Prediction only, using explicit independent feature schemas.
    frame[columns[0]], frame[columns[1]] = _predict(frame, reference), _predict(frame, candidate)
    return compare_predictions(frame, actual, *columns, reference_name=reference.name,
                               candidate_name=candidate.name, reference_anchor=reference.anchor,
                               candidate_anchor=candidate.anchor, **kwargs)
