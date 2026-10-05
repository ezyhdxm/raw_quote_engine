"""Record-weighted paired losses and date sensitivity; no statistical-significance claims."""
# SETUP LOGIC: All losses have already been converted into the caller's chosen unit.
import numpy as np
import pandas as pd


def metrics(rows, *, tolerance=1.0, min_count=30):
    # CORE LOGIC: STEP 1 — Measure both models on the identical set of records.
    # Input: reference errors=[-1,3], candidate errors=[0,2], tolerance=1.
    # Output: MAE=(2,1), RMSE≈(2.236068,1.414214), P95=(2.9,1.9), bias=(1,1).
    # Explanation: Absolute errors average to 2 and 1; RMSE takes the square root after averaging squares.
    # Trick: Normalize by maximum magnitude before squares/means to avoid overflow; P95 uses individual errors.
    result = dict(n=len(rows), bonds=rows['__bond'].nunique(), dates=rows['__date'].nunique())
    for side in ['reference','candidate']:
        err, ae = rows[f'__error_{side}'], rows[f'__ae_{side}']
        scale = ae.max()
        normalized = err/scale if scale > 0 else err
        result.update({f'{side}_mae': normalized.abs().mean()*scale, f'{side}_rmse': np.sqrt(np.square(normalized).mean())*scale})
        result.update({f'{side}_p95': ae.quantile(.95), f'{side}_bias': normalized.mean()*scale})
        result.update({f'{side}_median': ae.quantile(.5), f'{side}_within_tolerance_pct': ae.le(tolerance).mean()*100})
    # CORE LOGIC: STEP 2 — Report gain, paired wins and low support without filtering difficult groups.
    # Input: reference abs=[1,3], candidate abs=[0,2], n=2, min_count=30.
    # Output: mae_delta=-1, mae_improvement_pct=50, p95_delta=-1, win_rate_pct=100, tie_rate_pct=0, low_support=True.
    # Explanation: MAE falls from 2 to 1 and both trades improve; the small group is flagged, not removed.
    # Trick: A zero reference MAE has no defined relative percentage gain; return NaN instead of infinity.
    result['mae_delta'] = result['candidate_mae'] - result['reference_mae']
    result['mae_improvement_pct'] = -result['mae_delta']/result['reference_mae']*100 if result['reference_mae'] > 0 else np.nan
    result['p95_delta'] = result['candidate_p95'] - result['reference_p95']
    result['win_rate_pct'] = rows['__ae_candidate'].lt(rows['__ae_reference']).mean()*100
    result['tie_rate_pct'] = rows['__ae_candidate'].eq(rows['__ae_reference']).mean()*100
    result['low_support'] = len(rows) < min_count
    return result


def grouped_metrics(rows, keys, *, tolerance=1.0, min_count=30):
    # CORE LOGIC: STEP 1 — Recompute losses within each observed group using the same paired records.
    # Input: group=['A','A','B'], ref_abs=[1,3,4], cand_abs=[0,2,2].
    # Output: A has n=2, MAE=(2,1); B has n=1, MAE=(4,2).
    # Explanation: Each record contributes once to its group; there are no fabricated empty combinations.
    # Trick: observed=True omits unused categorical bins; explicit missing labels still form observed groups.
    records = []
    for labels, subset in rows.groupby(keys, observed=True, sort=True, dropna=False):
        labels = labels if isinstance(labels, tuple) else (labels,)
        records.append(dict(zip(keys, labels), **metrics(subset, tolerance=tolerance, min_count=min_count)))
    return pd.DataFrame(records, columns=[*keys,*metrics(rows.iloc[:0]).keys()])


def date_sensitivity(rows):
    # CORE LOGIC: STEP 1 — Aggregate paired error differences on records with known dates.
    # Input: dates=['D1','D1','D2'], candidate−reference absolute errors=[-1,-1,2].
    # Output: D1 has sum=-2,n=2,mean=-1; D2 has sum=2,n=1,mean=2.
    # Explanation: Differences are paired before aggregation; rows without dates cannot define a day.
    # Trick: These are error differences, not signed prediction errors or changes in target spreads.
    known = rows.loc[rows['__date'].notna()].copy()
    known['delta'] = known['__ae_candidate'] - known['__ae_reference']
    daily = known.groupby('__date', sort=True)['delta'].agg(['sum','count','mean'])
    # CORE LOGIC: STEP 2 — Remove each date's contribution without refitting any model.
    # Input: daily sums=[-2,2], counts=[2,1], means=[-1,2].
    # Output: date_equal_delta=.5, lodo=[2,-1], lodo_min=-1, lodo_max=2, dates=2.
    # Explanation: Removing D1 leaves D2's mean 2; removing D2 leaves D1's mean -1.
    # Trick: Date-equal and remaining-record weights differ; this range is not a confidence interval.
    remaining_n = daily['count'].sum() - daily['count']
    lodo = (daily['sum'].sum() - daily['sum']) / remaining_n.where(remaining_n > 0)
    result = dict(dates=len(daily), dated_rows=len(known), missing_date_rows=len(rows)-len(known))
    result.update(date_equal_delta=daily['mean'].mean(), lodo_min=lodo.min(), lodo_max=lodo.max())
    return pd.DataFrame([result])
