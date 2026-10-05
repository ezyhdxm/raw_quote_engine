"""Descriptive raw-quote diagnostics computed from one supplied run, without fitting."""
# SETUP LOGIC: Diagnostics accept canonical in-memory tables and never load private files.
from dataclasses import asdict, is_dataclass
from collections.abc import Mapping
import numpy as np
import pandas as pd

# CONFIGURATION LOGIC: Event keys describe simultaneous messages, not a persistent quote identity.
EVENT_KEYS = ['firm', 'cusip', 'side', 'quote_timestamp_ET']
QUANTITY_KINDS = ['Positive', 'Zero', 'Missing', 'Other']


def settings(config):
    # CONFIGURATION LOGIC: Accept the pipeline dataclass or an explicit dictionary.
    if config is None:
        return {}
    return asdict(config) if is_dataclass(config) else dict(config)


def numeric(frame, column):
    # DATA CONVERSION LOGIC: Missing columns produce aligned unknown values, never zeros.
    values = frame.get(column, pd.Series(np.nan, index=frame.index))
    return pd.to_numeric(values, errors='coerce').replace([np.inf, -np.inf], np.nan).astype(float)


def diagnostic_view(frame, config, preserve_priority_sources=True):
    # CONFIGURATION LOGIC: Only engine-standard roles are gated; arbitrary source columns remain in saved results.
    if 'transactions' not in config:
        return frame
    aliases = {'issuer': ['ISSUER', 'issuer'], 'sector': ['SECTOR', 'sector'],
               'quantity': ['QUANTITY', 'quantity'], 'prev_quantity': ['PREV_QUANTITY', 'prev_quantity'],
               'anchor': ['anchor'], 'cpp': ['cpp'], 'rollover_adjustment': ['rollover_adjustment']}
    roles = config['transactions']
    if not (roles.get('maturity_years') or roles.get('maturity_date')):
        aliases['maturity_years'] = ['MATURITY_YEARS', 'YRS_TO_MATURITY', 'maturity']
    # CORE LOGIC: STEP 1 — Exclude unconfigured metadata from automatic diagnostics without editing saved data.
    # Input: frame=[{cusip:'X',SECTOR:'Energy',x:1}], configured sector=None.
    # Output: diagnostic view=[{cusip:'X',x:1}]; the caller's frame still contains SECTOR='Energy'.
    # Explanation: A field name alone cannot opt a source into sector diagnosis after the user selected None.
    # Trick: Priority calculation preserves explicitly mapped history/gap fields; standard slices disable that exception.
    protected = {config.get('priority_history_column'), config.get('priority_cpp_gap_column')} if preserve_priority_sources else set()
    excluded = [column for role, columns in aliases.items() if roles.get(role) is None for column in columns if column not in protected]
    return frame.drop(columns=excluded, errors='ignore')


def quantity_classes(values):
    # CORE LOGIC: STEP 1 — Preserve the distinction between absent and invalid raw quantity.
    # Input: values=[2,0,None,-1,'bad',inf].
    # Output: ['Positive','Zero','Missing','Other','Other','Other'].
    # Explanation: Only finite positive values are Positive; an unparseable nonnull value is Other.
    # Trick: Test original nulls before coercion; float conversion makes nullable numeric masks safe for np.select.
    number = pd.to_numeric(values, errors='coerce').astype(float)
    finite = np.isfinite(number)
    kinds = np.select([values.isna(), finite & number.eq(0), finite & number.gt(0)],
                      ['Missing', 'Zero', 'Positive'], default='Other')
    return pd.Series(kinds, index=values.index)


def _universe(tx):
    # CORE LOGIC: STEP 1 — Define bond coverage from all supplied transaction bonds.
    # Input: tx.cusip=['X','X','Y']; no quote records are required.
    # Output: bond index=['X','Y'], target_rows=[2,1].
    # Explanation: Repeated trades count as targets while each bond occurs once in the coverage table.
    # Trick: Bonds without quotes remain in this denominator; invalid identifiers are audited separately.
    valid = tx.cusip.notna() & tx.cusip.astype('string').str.strip().ne('')
    source = tx.loc[valid]
    result = source.groupby('cusip', sort=True, observed=True).size().to_frame('target_rows')
    # CONFIGURATION LOGIC: Metadata is descriptive over the supplied period, not a prediction-time feature.
    for column in ['ISSUER', 'SECTOR']:
        # CORE LOGIC: STEP 2 — Retain conflicting labels instead of picking a convenient first value.
        # Input: X has SECTOR=['Energy','Utilities']; Y has SECTOR=[None].
        # Output: X='[Conflicting]', Y='[Unknown]'.
        # Explanation: Only a unique nonblank label is assigned to a bond.
        # Trick: Series align on the grouped bond index; this mapping is used for diagnostics only.
        labels = source.get(column, pd.Series(pd.NA, index=source.index)).astype('string').str.strip().replace('', pd.NA)
        groups = labels.groupby(source.cusip, observed=True)
        count, first = groups.nunique(), groups.first()
        result[column] = first.reindex(result.index).fillna('[Unknown]').mask(count.reindex(result.index).gt(1), '[Conflicting]')
    return result


def _aligned_features(tx, features):
    # VALIDATION LOGIC: Never attach quote states by row position or silently multiply targets.
    if tx.row_id.isna().any() or tx.row_id.duplicated().any():
        raise ValueError('Diagnostic transactions require unique nonmissing row_id.')
    if features is None:
        return tx.copy()
    if 'row_id' not in features or features.row_id.isna().any() or features.row_id.duplicated().any():
        raise ValueError('Diagnostic features require unique nonmissing row_id.')
    # CORE LOGIC: STEP 1 — Join only computed quote features while preserving every transaction.
    # Input: tx.row_id=[2,1,3]; features(row_id,bcq_has_quote)=[(1,1),(2,0)].
    # Output: row_id=[2,1,3], bcq_has_quote=[0,1,NaN].
    # Explanation: The missing feature row remains unknown; an explicit zero remains an observed absence.
    # Trick: Drop old same-name quote columns before joining, so stale features cannot shadow this run.
    columns = [c for c in features if c.startswith('bcq_') or c in
               ['prior_trade_count_30d', 'history_days_available', 'history_30d_complete']]
    left = tx.drop(columns=[c for c in columns if c in tx])
    return left.merge(features[['row_id', *columns]], on='row_id', how='left', sort=False, validate='one_to_one')


def _event_view(events):
    # DATA ADAPTER LOGIC: Accept the shared event cache or its already computed table.
    frame = events['events'] if isinstance(events, Mapping) else events
    if not isinstance(frame, pd.DataFrame):
        raise TypeError('events must be a DataFrame or a mapping containing events.')
    if not set(EVENT_KEYS + ['candidate_count', 'complete', 'quantity_set']).issubset(frame):
        raise ValueError('Event diagnostics require keys, candidate_count, complete and quantity_set.')
    result = frame.copy()
    # CORE LOGIC: STEP 1 — Derive event quality flags from retained distinct candidate sets.
    # Input: candidate_count=[2,1], complete=[True,False], quantity_set=[('q=2.0','q=3.0'),('Missing',)].
    # Output: multi_price=[True,False], incomplete=[False,True], positive_quantity_count=[2,0].
    # Explanation: Multiple raw rows do not establish multiple prices; distinct candidate counts do.
    # Trick: Missing event completeness is unknown, not evidence of a complete event.
    result['multi_price'] = numeric(result, 'candidate_count').gt(1)
    result['incomplete'] = ~result.get('complete', pd.Series(False, index=result.index)).fillna(False).astype(bool)
    tags = result.get('quantity_set', pd.Series([()] * len(result), index=result.index))
    result['positive_quantity_count'] = tags.map(lambda values: sum(str(v).startswith('q=') for v in values) if isinstance(values, (tuple, list, set, frozenset)) else 0)
    result['multiple_positive_quantities'] = result.positive_quantity_count.gt(1)
    result['gap'] = numeric(result, 'gap')
    return result


def _bond_table(universe, quotes, events, targets):
    # CORE LOGIC: STEP 1 — Attach quote/event support to the complete traded-bond denominator.
    # Input: universe X/Y with target_rows=2/1; quotes only X with three rows; events only X with two events.
    # Output: X raw_rows=3,event_count=2; Y raw_rows=0,event_count=0,no_quotes_in_file=True.
    # Explanation: Reindexing observed support to the universe restores bonds with no quote records.
    # Trick: No quotes anywhere in the file differs from no usable state at a target's prediction time.
    result = universe.copy()
    result['raw_rows'] = quotes.groupby('cusip', observed=True).size().reindex(result.index, fill_value=0)
    result['event_count'] = events.groupby('cusip', observed=True).size().reindex(result.index, fill_value=0)
    result['dealers'] = quotes.groupby('cusip', observed=True).firm.nunique().reindex(result.index, fill_value=0)
    result['no_quotes_in_file'] = result.raw_rows.eq(0)
    # CORE LOGIC: STEP 2 — Retain event ambiguity and target-state support with different denominators.
    # Input: X events multi_price=[True,False]; X targets bcq_has_quote=[1,0]; Y target bcq_has_quote=[NaN].
    # Output: X multi_events=1,targets_with_state=1,targets_state_unknown=0; Y targets_state_unknown=1.
    # Explanation: Event multiplicity counts events; target coverage counts evaluated transactions.
    # Trick: Unknown features cannot be labeled observed quote absence or filled with a numeric state.
    result['multi_events'] = events.groupby('cusip', observed=True).multi_price.sum().reindex(result.index, fill_value=0)
    result['incomplete_events'] = events.groupby('cusip', observed=True).incomplete.sum().reindex(result.index, fill_value=0)
    available = numeric(targets, 'bcq_has_quote')
    result['targets_with_state'] = available.gt(0).groupby(targets.cusip, observed=True).sum().reindex(result.index, fill_value=0)
    result['targets_state_unknown'] = available.isna().groupby(targets.cusip, observed=True).sum().reindex(result.index, fill_value=0)
    return result.reset_index()


def _metadata_summary(bonds, column):
    # CORE LOGIC: STEP 1 — Sum bond and target counts within explicit metadata groups.
    # Input: sector A has X(targets=2,raw=3) and Y(targets=1,raw=0).
    # Output: A bonds=2,target_rows=3,raw_rows=3,no_quote_bonds=1.
    # Explanation: No-quote bonds remain in the same sector denominator as quote-covered bonds.
    # Trick: Missing/conflicting metadata forms its own observed group; it is not dropped.
    result = bonds.groupby(column, observed=True, dropna=False).agg(
        bonds=('cusip', 'size'), target_rows=('target_rows', 'sum'), raw_rows=('raw_rows', 'sum'),
        event_count=('event_count', 'sum'), no_quote_bonds=('no_quotes_in_file', 'sum'),
        multi_events=('multi_events', 'sum'), incomplete_events=('incomplete_events', 'sum'),
        targets_with_state=('targets_with_state', 'sum'), targets_state_unknown=('targets_state_unknown', 'sum'))
    return result.reset_index()


def _dealer_summary(quotes, events):
    # CORE LOGIC: STEP 1 — Count dealer records and events without treating messages as independent votes.
    # Input: dealer D has three quote rows for X and one event with two distinct candidates.
    # Output: D raw_rows=3,bonds=1,event_count=1,multi_events=1.
    # Explanation: The table keeps raw and event counts separate.
    # Trick: A missing dealer stays visible among raw rows even though it cannot form a keyed event.
    result = quotes.groupby('firm', observed=True, dropna=False).agg(raw_rows=('cusip', 'size'), bonds=('cusip', 'nunique'))
    grouped = events.groupby('firm', observed=True, dropna=False)
    result['event_count'] = grouped.size().reindex(result.index, fill_value=0)
    result['multi_events'] = grouped.multi_price.sum().reindex(result.index, fill_value=0)
    result['incomplete_events'] = grouped.incomplete.sum().reindex(result.index, fill_value=0)
    return result.reset_index()


def _numeric_summary(frame, columns):
    # CONFIGURATION LOGIC: One fixed schema supports missing fields and empty populations.
    output = []
    for column in columns:
        # CORE LOGIC: STEP 1 — Describe finite values and report unavailable rows explicitly.
        # Input: values=[-2,0,4,NaN].
        # Output: n=4,available=3,missing=1,negative=1,zero=1,positive=1,median=0,min=-2,max=4.
        # Explanation: Quantiles use finite observations only; missing values are not zero measurements.
        # Trick: Each feature has its own availability denominator; these summaries are not paired model scores.
        values = numeric(frame, column)
        median = values.median() if values.notna().any() else np.nan
        output.append(dict(field=column, field_present=column in frame, n=len(frame), available=int(values.notna().sum()),
                           missing=int(values.isna().sum()), negative=int(values.lt(0).sum()), zero=int(values.eq(0).sum()),
                           positive=int(values.gt(0).sum()), median=median, p05=values.quantile(.05),
                           p95=values.quantile(.95), minimum=values.min(), maximum=values.max()))
    return pd.DataFrame(output, columns=['field', 'field_present', 'n', 'available', 'missing', 'negative',
                                         'zero', 'positive', 'median', 'p05', 'p95', 'minimum', 'maximum'])


def _coverage(targets):
    # CORE LOGIC: STEP 1 — Identify usable sides and fresh/matched pairs independently at target time.
    # Input: bid_n=[1,0,NaN],ask_n=[1,0,NaN],n_pair=[0,0,NaN],n_size_time_pair=[0,0,NaN].
    # Output: two_sided=1,no_usable_side=1,fresh_pair=0,size_time_pair=0; each condition unknown=1.
    # Explanation: Two complete sides need not form a fresh same-dealer pair.
    # Trick: Support masks require finite counts; an absent feature row stays unassessed.
    bid, ask = numeric(targets, 'bcq_bid_n_dealers'), numeric(targets, 'bcq_ask_n_dealers')
    known = bid.notna() & ask.notna()
    masks = {'two_sided': (bid.gt(0) & ask.gt(0), known), 'one_sided': (bid.gt(0) ^ ask.gt(0), known),
             'no_usable_side': (bid.eq(0) & ask.eq(0), known)}
    for label, field in [('fresh_pair', 'bcq_n_pair'), ('size_time_pair', 'bcq_n_size_time_pair')]:
        value = numeric(targets, field)
        masks[label] = (value.gt(0), value.notna())
    # CORE LOGIC: STEP 2 — Preserve all-target and assessable denominators for every condition.
    # Input: condition=[True,False,False],known=[True,True,False],n=3.
    # Output: matched=1,assessed=2,unknown=1,pct_assessed=50.
    # Explanation: The one unknown row remains in total n but not in the assessable rate.
    # Trick: Zero assessed rows produce NaN rather than an apparent zero-percent incidence.
    records = [dict(condition=name, n=len(targets), matched=int((mask & known).sum()), assessed=int(known.sum()),
                    unknown=int((~known).sum()), pct_assessed=float((mask & known).sum()/known.sum()*100) if known.any() else np.nan)
               for name, (mask, known) in masks.items()]
    return pd.DataFrame(records)


def proxy_discrepancy(targets, config):
    # CORE LOGIC: STEP 1 — Compare a supplied proxy with the reference anchor in the current benchmark basis.
    # Input: anchor=[1.00,1.00], rollover_adjustment=[.02,.00], proxy=[1.04,1.04], error_scale=100; adjustment mapped.
    # Output: current-basis anchors=[.98,1.00], discrepancies approximately [6,4] bps.
    # Explanation: Subtract the declared offset from the anchor, then convert absolute proxy differences to bps.
    # Trick: An unmapped source column named rollover_adjustment is ignored; user-precomputed gap columns bypass this function.
    anchor = numeric(targets, 'anchor')
    if config.get('transactions', {}).get('rollover_adjustment') is not None:
        anchor = anchor - numeric(targets, 'rollover_adjustment')
    return (numeric(targets, 'cpp') - anchor).abs() * config.get('error_scale', 1)


def priority_groups(targets, config):
    """Return declared masks and required columns, independent of predictions and gains."""
    # CONFIGURATION LOGIC: Fixed diagnostic slices are independent of model gains and include availability notes.
    targets = diagnostic_view(targets, config)
    roles = config.get('transactions')
    size_field = 'QUANTITY' if roles is None or roles.get('quantity') is not None else None
    maturity = next((c for c in ['MATURITY_YEARS', 'YRS_TO_MATURITY'] if c in targets), None)
    if roles is not None and not (roles.get('maturity_years') or roles.get('maturity_date')):
        maturity = None
    history = config.get('priority_history_column')
    if history is None:
        candidates = ['prior_trade_count_30d'] if 'transactions' in config else ['prior_trade_count_30d', 'TRADE_COUNTS_PREV_MONTH', 'prior_count']
        history = next((c for c in candidates if c in targets), None)
    size, term, count = numeric(targets, size_field), numeric(targets, maturity), numeric(targets, history)
    # CORE LOGIC: STEP 1 — Form large, long-maturity and sparse-history masks without imputing unknowns.
    # Input: quantity=[1e6,2e6,2e6],term=[2,.1,NaN],count=[0,14,NaN].
    # Output: large=[True,True,True],long=[True,False,False],sparse=[True,True,False].
    # Explanation: Exactly 1MM qualifies; maturity must strictly exceed one year; history must be an integer in 0..14.
    # Trick: Negative/noninteger counts and missing maturity never enter the normal sparse/long groups.
    large, long = size.ge(1e6), term.gt(1)
    valid_count = count.notna() & count.ge(0) & count.eq(np.floor(count))
    sparse = valid_count & count.le(14)
    groups = [('All', pd.Series(True, index=targets.index), []), ('>=1MM', large, [size_field]),
              ('>=1MM & maturity>1y', large & long, [size_field, maturity]),
              ('Prior count 0-14', sparse, [history]), ('>=1MM & prior count 0-14', large & sparse, [size_field, history]),
              ('>=1MM & prior count 0-14 & maturity>1y', large & sparse & long, [size_field, history, maturity])]
    # CONFIGURATION LOGIC: Choose declared discrepancy or canonical levels; both require a spread/bps contract.
    spread = config.get('value_kind', 'spread') == 'spread' and config.get('unit', 'bps') == 'bps'
    gap_column = config.get('priority_cpp_gap_column')
    cpp_fields = [gap_column] if gap_column is not None else ['cpp', 'anchor']
    if gap_column is None and config.get('transactions', {}).get('rollover_adjustment') is not None:
        cpp_fields.append('rollover_adjustment')
    cpp_available = spread and all(c in targets for c in cpp_fields)
    if roles is not None and gap_column is None:
        cpp_available &= roles.get('cpp') is not None and roles.get('anchor') is not None
    # CORE LOGIC: STEP 2 — Apply declared spread-scale discrepancy thresholds without calling the anchor bad.
    # Input: priority_cpp_gap_column='gap_bps', gap_bps=[5,10,10.01,NaN], quantities all 2MM, error_scale=100.
    # Output: >5 selects [False,True,True,False]; >10 selects [False,False,True,False].
    # Explanation: Explicit discrepancies are already bps; otherwise the proxy is compared with the adjusted anchor.
    # Trick: No re-scaling, anchor reconstruction or row fallback changes explicit gap values or exact boundaries.
    if gap_column is None:
        gap = proxy_discrepancy(targets, config)
    else:
        gap = numeric(targets, gap_column).abs()
    for cutoff in [5, 10]:
        selected = large & gap.gt(cutoff) if cpp_available else pd.Series(False, index=targets.index)
        groups.extend([(f'>=1MM & CPP gap>{cutoff}bps', selected, [size_field, *cpp_fields]),
                       (f'>=1MM & CPP gap>{cutoff}bps & maturity>1y', selected & long, [size_field, maturity, *cpp_fields])])
    return groups, history, cpp_available


def _priority_support(targets, config):
    # CONFIGURATION LOGIC: Share exact fixed masks with the model-comparison report.
    groups, history, cpp_available = priority_groups(targets, config)
    # CORE LOGIC: STEP 1 — Publish unsupported groups instead of silently omitting them.
    # Input: no MATURITY_YEARS or YRS_TO_MATURITY column; requested group='>=1MM & maturity>1y'.
    # Output: supported=False,n=0,note='Missing required columns or spread-unit contract'.
    # Explanation: An unavailable slice cannot establish either prevalence or model benefit.
    # Trick: Counts are descriptive support only; model comparisons must recompute paired losses on these rows.
    records = []
    for name, mask, required in groups:
        supported = all(c is not None and c in targets for c in required) and ('CPP' not in name or cpp_available)
        subset = targets.loc[mask] if supported else targets.iloc[:0]
        records.append(dict(cohort=name, supported=supported, n=len(subset), bonds=subset.cusip.nunique(), history_field=history,
                            note='Fixed descriptive support; no model selection' if supported else 'Missing required columns or spread-unit contract'))
    return pd.DataFrame(records)


def _maturity_support(targets):
    # CONFIGURATION LOGIC: Preserve short maturities and explicit unknowns in both full and large-trade populations.
    field = next((name for name in ['MATURITY_YEARS', 'YRS_TO_MATURITY'] if name in targets), None)
    term = numeric(targets, field)
    names = ['<=3 months', '3-12 months', '>1 year', 'Unknown/invalid']
    # CORE LOGIC: STEP 1 — Assign mutually exclusive maturity groups, leaving negative values unknown.
    # Input: term=[.1,.25,1,2,-1,NaN].
    # Output: ['<=3 months','<=3 months','3-12 months','>1 year','Unknown/invalid','Unknown/invalid'].
    # Explanation: Quarter-year and one-year boundaries are closed above; negative terms are invalid.
    # Trick: A missing maturity column assigns every target to Unknown, without dropping that cohort.
    labels = pd.Series(np.select([term.ge(0) & term.le(.25), term.gt(.25) & term.le(1), term.gt(1)],
                                 names[:3], default=names[-1]), index=targets.index)
    cohorts = [('All', pd.Series(True, index=targets.index)), ('>=1MM', numeric(targets, 'QUANTITY').ge(1e6))]
    # CORE LOGIC: STEP 2 — Keep all four bins, including zero-count groups, within both cohorts.
    # Input: one 2MM target has term=2 and bond X.
    # Output: each cohort has bin counts [0,0,1,0], and the >1-year bin has bonds=1.
    # Explanation: Empty groups describe zero observed support, not zero model error.
    # Trick: Iterate declared bins rather than observed-only groupby so empty/unknown categories remain inspectable.
    rows = []
    for cohort, eligible in cohorts:
        for group in names:
            subset = targets.loc[eligible & labels.eq(group)]
            rows.append(dict(cohort=cohort, maturity_group=group, n=len(subset), bonds=subset.cusip.nunique(), field_present=field is not None))
    return pd.DataFrame(rows)


def _cases(events, seed):
    # CONFIGURATION LOGIC: Case selection is repeatable for the same event table, not a prevalence estimator.
    columns = EVENT_KEYS + ['rows', 'candidate_count', 'gap', 'positive_quantity_count', 'incomplete']
    if events.empty:
        return pd.DataFrame(columns=['case_kind', *columns])
    # CORE LOGIC: STEP 1 — Select a seeded random event, a median-range event and two widest-range events.
    # Input: sorted event gaps=[0,2,4], random_seed=2026.
    # Output: typical gap=2; high-range gaps=[4,2]; random gap=4 with pandas random_state=2026.
    # Explanation: Typical minimizes distance to the median; high-range ranks ambiguity, not model harm.
    # Trick: The same event may serve multiple case roles; never add these rows to estimate incidence.
    ordered = events.sort_values(EVENT_KEYS, kind='stable').reset_index(drop=True)
    distance = (ordered.gap - (ordered.gap.median() if ordered.gap.notna().any() else np.nan)).abs()
    typical = ordered.loc[distance.dropna().nsmallest(1).index]
    samples = [('Random', ordered.sample(n=1, random_state=seed)), ('Typical candidate range', typical),
               ('High candidate range', ordered.loc[ordered.gap.notna()].sort_values('gap', ascending=False, kind='stable').head(2))]
    return pd.concat([part.reindex(columns=columns).assign(case_kind=label) for label, part in samples], ignore_index=True)


def _case_pair_tables(targets, events, config):
    # SETUP LOGIC: Reuse the exact same paired-slot calculator as feature construction, without fitting.
    from .state import pair_snapshots, pair_policy_comparison
    # CORE LOGIC: STEP 1 — Freeze at most eighteen targets without inspecting labels or model losses.
    # Input: three targets have abs(pair_gap)=[0,2,4]; each role has a six-target cap.
    # Output: all three row identities are retained once; the final query count is 3, at most 18.
    # Explanation: Seeded random, median-gap and largest-gap selections explain different conditions.
    # Trick: Duplicate roles retain their first label; selection depends on features only, never target errors.
    ordered = targets.sort_values(['cusip', 'time'], kind='stable')
    score = numeric(ordered, 'bcq_pair_gap').abs()
    typical = (score - (score.median() if score.notna().any() else np.nan)).abs().dropna().nsmallest(6).index
    parts = [ordered.sample(n=min(6, len(ordered)), random_state=config.get('case_seed', 2026)).assign(case_kind='Random target'),
             ordered.loc[typical].assign(case_kind='Typical absolute pair gap'),
             ordered.loc[score.dropna().nlargest(6).index].assign(case_kind='High absolute pair gap')]
    queries = pd.concat(parts).drop_duplicates('row_id')[['row_id', 'cusip', 'time', 'case_kind']]
    # CONFIGURATION LOGIC: Bound expensive reconstruction by selected target identities and unique bond/time queries.
    policies, effects = [], []
    for bond, group in queries.groupby('cusip', sort=False, observed=True):
        history = events.loc[events.cusip.eq(bond)]
        pairs = pair_snapshots(history, group.time, config.get('age_min', 30), config.get('sync_min', 1),
                               config.get('allow_exact', True), value_kind=config.get('value_kind', 'spread'))
        for target in group.itertuples(index=False):
            # CORE LOGIC: STEP 2 — Separate slot selection from same-slot positive-quantity matching.
            # Input: A uses two gaps [-10,4]; B uses original gap [4]; C matches that B slot to gap [2].
            # Output: A mean=-3,n=2; B mean=4,n=1; C mean=2,n=1; selection delta=7, matching delta=-2.
            # Explanation: A-to-B changes support; B-to-C changes candidate matching on identical slots.
            # Trick: These are selected case snapshots, not all-population event rates or causal size estimates.
            table = pair_policy_comparison(pairs.loc[pairs.time.eq(target.time)]).reset_index()
            table = table.rename(columns={'gap_mean_bps': 'gap_mean_value', 'mid_mean_bps': 'mid_mean_value'})
            a, b, c = [row for _, row in table.iterrows()]
            assert b.n_slots == c.n_slots
            policies.append(table.assign(row_id=target.row_id, cusip=bond, time=target.time, case_kind=target.case_kind))
            effects.append(dict(row_id=target.row_id, cusip=bond, time=target.time, original_slots=a.n_slots, matched_slots=b.n_slots,
                                selection_delta_gap=b.gap_mean_value-a.gap_mean_value, same_slot_match_delta_gap=c.gap_mean_value-b.gap_mean_value))
    # REPORTING LOGIC: Empty selections remain explicit tables and never trigger a wider case search.
    policy_columns = ['policy', 'n_slots', 'n_dealers', 'gap_mean_value', 'mid_mean_value', 'none_fraction',
                      'some_fraction', 'all_fraction', 'row_id', 'cusip', 'time', 'case_kind']
    effect_columns = ['row_id', 'cusip', 'time', 'original_slots', 'matched_slots', 'selection_delta_gap', 'same_slot_match_delta_gap']
    return {'case_pair_queries': queries, 'case_pair_policy': pd.concat(policies, ignore_index=True) if policies else pd.DataFrame(columns=policy_columns),
            'case_pair_effect': pd.DataFrame(effects, columns=effect_columns)}


def diagnostic_tables(tx, quotes, events, features, config=None):
    """Return raw/event/target diagnostics; preserve inputs and never fit or infer a causal cleaning rule."""
    # VALIDATION LOGIC: Require canonical keys; the upstream adapter owns unit and known-time normalization.
    required_tx, required_quotes = {'row_id', 'cusip', 'time', 'target'}, set(EVENT_KEYS + ['spread', 'quantity'])
    if not required_tx.issubset(tx) or not required_quotes.issubset(quotes):
        raise ValueError('Diagnostics need canonical transactions and raw quote fields.')
    options = settings(config)
    tx = diagnostic_view(tx, options)
    universe = _universe(diagnostic_view(tx, options, preserve_priority_sources=False))
    targets, event_rows = _aligned_features(tx, features), _event_view(events)
    # CORE LOGIC: STEP 1 — Restrict quote diagnostics to the traded universe while auditing excluded records.
    # Input: transaction bonds=[X,Y]; quote bonds=[X,X,Z]; events have X/Z.
    # Output: in-universe quote rows=2,outside-universe rows=1; retained events belong to X.
    # Explanation: Bond Y remains in the universe even without quotes; Z is reported as outside the intended population.
    # Trick: Universe restriction does not remove zero/negative values, incomplete states or duplicates within X.
    in_universe = quotes.cusip.isin(universe.index)
    raw = quotes.loc[in_universe]
    event_rows = event_rows.loc[event_rows.cusip.isin(universe.index)]
    bonds = _bond_table(universe, raw, event_rows, targets)
    classes = raw['quantity_kind'] if 'quantity_kind' in raw else quantity_classes(raw.quantity)
    quantity = classes.value_counts().reindex(QUANTITY_KINDS, fill_value=0).rename_axis('quantity_kind').reset_index(name='raw_rows')
    # CORE LOGIC: STEP 2 — Keep raw-row, event, bond and target denominators distinct in the overview.
    # Input: targets=3,bonds=2; in-universe raw rows=4 with one duplicate; events=2 with one multi-price event.
    # Output: global records retain target_rows=3,bonds=2,raw_rows=4,raw_duplicate_rows=1,events=2,multi_price_events=1.
    # Explanation: Prefer the ingestion flag measured before generated IDs; repeats do not create distinct price votes.
    # Trick: A wide source duplicate expands to two duplicate side rows; these counts are not unique source messages.
    duplicates = raw.source_duplicate.fillna(False) if 'source_duplicate' in raw else raw.duplicated()
    counts = dict(target_rows=len(tx), bonds=len(universe), raw_rows=len(raw), raw_rows_outside_universe=int((~in_universe).sum()),
                  raw_duplicate_rows=int(duplicates.sum()), invalid_spread_rows=int(numeric(raw, 'spread').isna().sum()),
                  events=len(event_rows), multi_price_events=int(event_rows.multi_price.sum()),
                  incomplete_events=int(event_rows.incomplete.sum()), no_quote_bonds=int(bonds.no_quotes_in_file.sum()))
    global_table = pd.DataFrame([{'metric': name, 'count': value} for name, value in counts.items()])
    # REPORTING LOGIC: Field names explicitly identify saved target-level aggregates, not unobserved pair-level causes.
    pair_fields = ['bcq_n_pair', 'bcq_n_size_time_pair', 'bcq_pair_gap', 'bcq_pair_gap_low', 'bcq_pair_gap_high',
                   'bcq_pair_cross_some', 'bcq_pair_cross_all', 'bcq_pair_time_gap', 'bcq_pair_unknown_size', 'bcq_size_time_gap']
    side_fields = [f'bcq_{side}_{field}' for side in ['bid', 'ask'] for field in
                   ['median_message_age_min', 'median_change_age_min', 'dispersion_bps', 'mean_candidate_gap', 'n_peer_supported', 'n_incomplete']]
    tables = dict(global_summary=global_table, bond=bonds, sector=_metadata_summary(bonds, 'SECTOR'),
                  issuer=_metadata_summary(bonds, 'ISSUER'), dealer=_dealer_summary(raw, event_rows), quantity=quantity,
                  event_quality=_numeric_summary(event_rows, ['rows', 'repeats', 'candidate_count', 'positive_quantity_count', 'gap']),
                  target_coverage=_coverage(targets), pair_sources=_numeric_summary(targets, pair_fields + side_fields),
                  feature_availability=_numeric_summary(targets, [c for c in targets if c.startswith('bcq_')]),
                  priority_support=_priority_support(targets, options),
                  maturity_support=_maturity_support(diagnostic_view(targets, options, preserve_priority_sources=False)),
                  history_coverage=_numeric_summary(targets, ['prior_trade_count_30d', 'history_days_available', 'history_30d_complete']),
                  cases=_cases(event_rows, options.get('case_seed', 2026)))
    # REPORTING LOGIC: An omitted mapping disables its standard summary instead of creating a guessed group.
    for name in ['sector', 'issuer']:
        if 'transactions' in options and options['transactions'].get(name) is None:
            tables[name] = pd.DataFrame([dict(supported=False, note=f'No {name} column configured')])
    # ORCHESTRATION LOGIC: Reconstruct only the frozen bounded case queries with the shared pair policies.
    tables.update(_case_pair_tables(targets, event_rows, options))
    return tables
