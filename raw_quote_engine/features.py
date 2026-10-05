"""One causal feature assembly, preserving the complete transaction population."""
# SETUP LOGIC: Numerical kernels are shared across diagnosis and model experiments.
import numpy as np
import pandas as pd
from .state import build_quote_features
from .movement import build_movement_features, DIRECTION_FEATURES, ISSUER_FEATURES
from .path import build_path_features, PATH_FEATURES


def history_features(tx):
    # CORE LOGIC: STEP 1 — Count strictly earlier transactions within the trailing 30 calendar days.
    # Input: X trades Jan 1 10:00, Jan 2 10:00, Jan 2 10:00, Feb 2 10:00.
    # Output: prior_trade_count_30d=[0,1,1,0].
    # Explanation: Equal-time trades cannot count each other; Jan 2 is outside the Feb 2 window.
    # Trick: Both search boundaries use left, defining [t-30days,t), with no target values read.
    result = pd.Series(0, index=tx.index, dtype='int64')
    for _, group in tx.groupby('cusip', sort=False, observed=True):
        stamps = group.time.array.as_unit('ns').asi8
        ordered = np.sort(stamps)
        left = np.searchsorted(ordered, stamps - 30 * 86400 * 10**9, side='left')
        right = np.searchsorted(ordered, stamps, side='left')
        result.loc[group.index] = right-left
    # CORE LOGIC: STEP 2 — Expose truncation of the available transaction history.
    # Input: supplied data starts Jan 1 10:00; query Jan 2 10:00 has one prior trade.
    # Output: prior_trade_count_30d=1, history_days_available=1.0, history_30d_complete=False.
    # Explanation: A low observed count after one day cannot establish 30-day illiquidity.
    # Trick: This checks dataset coverage only; it cannot certify completeness of the supplied feed.
    elapsed = (tx.time-tx.time.min()).dt.total_seconds()/86400
    return pd.DataFrame({'prior_trade_count_30d': result,
                         'history_days_available': elapsed.clip(upper=30),
                         'history_30d_complete': elapsed.ge(30)}, index=tx.index)


def assemble_features(tx, quotes, events, config, progress=None):
    # ORCHESTRATION LOGIC: All targets are requested explicitly; these kernels never read target values.
    current = build_quote_features(quotes, tx, config.age_min, config.sync_min, config.allow_exact,
                                   progress, events, config.value_kind, config.clip_floor)
    movement, movement_meta = build_movement_features(
        tx, events, lookback_min=config.lookback_min, age_min=config.age_min,
        allow_exact=config.allow_exact, include_issuer=config.transactions.issuer is not None, progress=progress,
        query_positions=np.arange(len(tx)))
    path, path_meta = build_path_features(events, tx, np.arange(len(tx)), progress,
                                         config.age_min, config.lookback_min, config.allow_exact)
    # CORE LOGIC: STEP 1 — Join each feature family by immutable transaction identity.
    # Input: tx row_id=[9,7]; current rows=[(7,100),(9,110)] with bid center values.
    # Output: tx row_id=[9,7], bid center=[110,100]; both original transactions survive.
    # Explanation: One-to-one joins attach calculations rather than relying on group iteration order.
    # Trick: Explicit column selection avoids suffixing time/bond/split or replacing base features.
    frame = tx.copy()
    for family in (current, movement, path):
        columns = [c for c in family if c.startswith('bcq_')]
        frame = frame.merge(family[['row_id']+columns], on='row_id', how='left', validate='one_to_one')
    history = history_features(frame)
    frame = pd.concat([frame, history], axis=1)
    # CORE LOGIC: STEP 2 — Express quote levels relative to a supplied prediction-time anchor.
    # Input: bid center=105, ask center=103, anchor=100, mapped rollover_adjustment=2.
    # Output: bcq_bid_center_equal_to_anchor=7, bcq_ask_center_equal_to_anchor=5.
    # Explanation: The benchmark-adjusted anchor is 100-2=98; without a mapping the differences would be 5 and 3.
    # Trick: Anchor and adjustment must be observable at prediction time; target is never read and an unmapped source column is ignored.
    if config.transactions.anchor is not None:
        adjustment = frame.rollover_adjustment if config.transactions.rollover_adjustment is not None else 0.
        effective_anchor = frame.anchor-adjustment
        levels = [c for c in current if 'center_' in c and not c.endswith('n_changed_centers')]
        levels += [c for c in ('bcq_pair_mid', 'bcq_size_time_mid') if c in current]
        for name in levels:
            frame[name+'_to_anchor'] = frame[name]-effective_anchor
    # METADATA LOGIC: Return the exact columns used to construct each nested ablation.
    groups = feature_groups(frame)
    metadata = {'movement': movement_meta, 'path': path_meta, 'groups': groups}
    return frame, metadata


def feature_groups(frame):
    # CONFIGURATION LOGIC: Fixed ablations avoid searching a large family of models on evaluation slices.
    core = ['bcq_has_quote', 'bcq_n_pair', 'bcq_pair_gap', 'bcq_pair_cross_all',
            'bcq_pair_cross_some', 'bcq_pair_time_gap', 'bcq_pair_unknown_size']
    side_fields = ['center_max_age', 'center_decay', 'n_dealers', 'n_fresh_dealers',
                   'median_message_age_min', 'median_change_age_min', 'dispersion_bps',
                   'unknown_quantity_fraction', 'zero_quantity_fraction', 'multi_fraction',
                   'mean_candidate_gap', 'n_incomplete', 'unknown_change_age_fraction']
    # CORE LOGIC: STEP 1 — Prefer anchor-relative levels when the anchor exists.
    # Input: columns contain bcq_bid_center_max_age and bcq_bid_center_max_age_to_anchor.
    # Output: the Quote feature list contains only bcq_bid_center_max_age_to_anchor for that field.
    # Explanation: Non-level reliability statistics remain unchanged, and missing optional columns are omitted.
    # Trick: This is a fixed schema rule; no validation error or target determines which level is used.
    for side in ('bid', 'ask'):
        for field in side_fields:
            name = f'bcq_{side}_{field}'
            relative = name+'_to_anchor'
            core.append(relative if relative in frame else name)
    core = [name for name in core if name in frame]
    path = [name for name in PATH_FEATURES if name in frame]
    cross = [name for name in DIRECTION_FEATURES+ISSUER_FEATURES if name in frame]
    return {'Quote': core, 'Quote+Path': core+path, 'Quote+CrossBond': core+path+cross}


def feature_dictionary(frame, groups, config):
    """Define every generated feature explicitly; diagnostic-only outputs keep empty model membership."""
    # CONFIGURATION LOGIC: c_i is a dealer's median distinct candidate value; age_i is minutes since its latest event.
    side_specs = {
        'center_equal': ('quote value unit', 'Arithmetic mean of c_i across all latest complete same-day dealer states.'),
        'n_dealers': ('count', 'Number of latest complete same-day dealer states; freshness is not required.'),
        'n_fresh_dealers': ('count', 'Number of latest complete same-day dealer states with age_i <= age_min.'),
        'n_incomplete': ('count', 'Number of latest same-day dealer states containing a nonfinite candidate or no finite candidate.'),
        'center_decay': ('quote value unit', 'sum(w_i*c_i)/sum(w_i), w_i=2**(-age_i/age_min), over all latest complete same-day states.'),
        'center_max_age': ('quote value unit', 'Arithmetic mean of c_i restricted to latest complete same-day states with age_i <= age_min.'),
        'dispersion_bps': ('quote value unit', 'sqrt(mean((c_i-mean(c_i))**2)) across latest complete same-day dealers; population ddof=0.'),
        'mean_candidate_gap': ('quote value unit', 'Arithmetic mean of max(candidate values)-min(candidate values) within each latest complete dealer event.'),
        'multi_fraction': ('fraction', 'Dealers with more than one distinct finite candidate / complete same-day dealers.'),
        'zero_quantity_fraction': ('fraction', 'Complete same-day dealers whose candidate quantity tags include literal zero / complete same-day dealers.'),
        'unknown_quantity_fraction': ('fraction', 'Complete same-day dealers with at least one Missing or Other quantity tag / complete same-day dealers; zero has its own fraction.'),
        'median_message_age_min': ('minutes', 'Median age_i across latest complete same-day dealer states.'),
        'median_change_age_min': ('minutes', 'Median elapsed minutes since the latest observed value-set change, among complete dealers with a known change in their current <=60-minute-gap history segment.'),
        'unknown_change_age_fraction': ('fraction', 'Complete same-day dealers with no observed value-set change in their current continuous history segment / complete same-day dealers.'),
        'center_lower': ('quote value unit', 'Arithmetic mean of each latest complete dealer event minimum candidate value.'),
        'center_upper': ('quote value unit', 'Arithmetic mean of each latest complete dealer event maximum candidate value.'),
        'max_decay_weight_share': ('fraction', 'max(w_i)/sum(w_i), w_i=2**(-age_i/age_min), across complete same-day dealers.'),
        'decay_effective_dealers': ('effective dealer count', '(sum(w_i)**2)/sum(w_i**2), w_i=2**(-age_i/age_min); effective support is not a raw dealer count.'),
        'center_candidate_clip': ('quote value unit', 'Mean of dealer medians after clipping each candidate into its leave-one-dealer-out peer interval; unsupported dealers retain their original median.'),
        'center_dealer_downweight': ('quote value unit', 'sum(v_i*c_i)/sum(v_i), where supported dealer v_i=min(1,radius_i/abs(c_i-peer_median_i)); zero residual or unsupported dealers use v_i=1.'),
        'n_peer_supported': ('count', 'Complete same-day dealers with at least three other fresh complete dealer centers for a leave-one-dealer-out peer reference.'),
        'n_clipped_dealers': ('count', 'Peer-supported dealers with any candidate outside their peer clipping interval; counts candidate changes even when the median stays unchanged.'),
        'n_changed_centers': ('count', 'Peer-supported dealers whose candidate-clipped median differs from the original c_i with rtol=0 and atol=1e-9.'),
    }
    # CONFIGURATION LOGIC: Width is bid-minus-ask for the supported spread input.
    pair_specs = {
        'has_quote': ('indicator', '1 when either side has at least one latest complete same-day dealer state, otherwise 0; freshness is not required.'),
        'n_pair': ('count', 'Dealers with both latest sides complete on the same day and max(bid_age,ask_age) <= age_min.'),
        'n_size_time_pair': ('count', 'Fresh complete dealer pairs additionally meeting abs(bid_time-ask_time) <= sync_min and at least one shared finite positive quantity.'),
        'pair_gap': ('quote value unit', 'Equal-dealer mean oriented width between bid and ask candidate medians over fresh complete pairs.'),
        'pair_gap_low': ('quote value unit', 'Equal-dealer mean minimum oriented width over all bid/ask candidate combinations in each fresh complete pair.'),
        'pair_gap_high': ('quote value unit', 'Equal-dealer mean maximum oriented width over all bid/ask candidate combinations in each fresh complete pair.'),
        'pair_mid': ('quote value unit', 'Equal-dealer mean of (median(bid candidates)+median(ask candidates))/2 over fresh complete pairs.'),
        'pair_mid_range': ('quote value unit', 'Equal-dealer mean of ((max_bid-min_bid)+(max_ask-min_ask))/2 over fresh complete pairs.'),
        'pair_cross_some': ('fraction', 'Fresh complete pairs with minimum oriented width < 0 and maximum width >= 0 / fresh complete pairs.'),
        'pair_cross_all': ('fraction', 'Fresh complete pairs with maximum oriented width < 0 / fresh complete pairs; locked zero width is not crossing.'),
        'pair_time_gap': ('minutes', 'Median absolute bid-versus-ask latest message-time difference over fresh complete dealer pairs.'),
        'pair_unknown_size': ('fraction', 'Fresh complete pairs with no shared positive quantity / fresh complete pairs; includes unmatched positive sizes, not only unknown quantities.'),
        'size_time_gap': ('quote value unit', 'For each eligible size/time pair, average oriented differences of bid/ask candidate medians within each shared positive quantity, then average dealers equally.'),
        'size_time_mid': ('quote value unit', 'For each eligible size/time pair, average (bid median+ask median)/2 within shared positive quantities, then average dealers equally.'),
        'size_time_cross_some': ('fraction', 'Eligible size/time pairs whose matched-quantity oriented width bounds straddle zero (low < 0 <= high) / eligible size/time pairs.'),
        'size_time_cross_all': ('fraction', 'Eligible size/time pairs whose maximum matched-quantity oriented width is strictly negative / eligible size/time pairs.'),
    }
    # CONFIGURATION LOGIC: Support rules apply without modifying source quotes or using target labels.
    state_support = 'One dealer per vote; latest known-time event on the configured local day; an incomplete latest event blocks older valid state.'
    peer_support = f' Peer interval is median(other fresh centers) +/- max({config.clip_floor:g},4*1.4826*MAD); at least three other fresh dealers are required.'
    pair_support = 'Same dealer and local day; both latest sides complete and fresh. Size/time features additionally require synchronization and shared positive quantity; quantities never weight the averages.'
    # CORE LOGIC: STEP 1 — Register explicit side and pair definitions under their actual feature names.
    # Input: side_specs['n_dealers']=('count','Number of latest complete same-day dealer states; freshness is not required.').
    # Output: bcq_bid_n_dealers and bcq_ask_n_dealers both have unit='count', family='Quote state', and that exact definition.
    # Explanation: Side prefixes identify separate computations; the same definition applies symmetrically.
    # Trick: Metadata is keyed by complete names, so substring matches cannot confuse center, count, or age fields.
    details = {}
    for side in ('bid', 'ask'):
        for suffix, (unit, definition) in side_specs.items():
            support = state_support + (peer_support if suffix in {'center_candidate_clip', 'center_dealer_downweight', 'n_peer_supported', 'n_clipped_dealers', 'n_changed_centers'} else '')
            details[f'bcq_{side}_{suffix}'] = dict(unit=unit, definition=definition, support=support, family='Quote state')
    for suffix, (unit, definition) in pair_specs.items():
        details['bcq_'+suffix] = dict(unit=unit, definition=definition, support=pair_support if suffix != 'has_quote' else state_support, family='Quote pairs')
    # REPORTING LOGIC: Reuse authoritative movement/path definitions rather than independently paraphrasing names.
    from .movement import _dictionary
    from .path import PATH_DEFINITIONS
    # CORE LOGIC: STEP 2 — Attach endpoint movement rules and the correct dealer/bond vote denominator.
    # Input: movement definition bcq_bid_move_common_n has unit='count' and definition='Dealer count complete and fresh at both endpoints'.
    # Output: the same name keeps that definition/unit and receives family='Dealer movement'.
    # Explanation: Issuer names instead receive family='Other-bond issuer movement' and retain their target-exclusion/minimum-two-bond support.
    # Trick: Window and age values come from the actual config; legacy _bps names do not assert basis-point input.
    for name, specification in _dictionary(include_issuer=True).items():
        specification = dict(specification)
        specification['family'] = 'Other-bond issuer movement' if '_issuer_move_' in name else 'Dealer movement'
        specification['support'] += f' Endpoints are query and query-{config.lookback_min:g}min; each endpoint age <= {config.age_min:g}min.'
        details[name] = specification
    # CORE LOGIC: STEP 3 — Document the path fractions and their explicit comparable-dealer counts.
    # Input: side='bid'; PATH_DEFINITIONS['flat_refresh_share'] has unit='fraction'.
    # Output: bcq_path_bid_flat_refresh_share has family='Quote path', unit='fraction'; bcq_path_bid_comparable_n has unit='count'.
    # Explanation: The count is the denominator for refresh/direction shares; last-move median uses only known nonzero moves.
    # Trick: Comparable means consecutive complete same-day states, gap <=60min, unchanged quantity set/candidate count, and latest age <= age_min.
    path_support = f'Latest event age <= {config.age_min:g}min; consecutive complete same-day states no more than 60min apart with unchanged quantity set and candidate count; direction shares use (query-{config.lookback_min:g}min,query].'
    for side in ('bid', 'ask'):
        for suffix, specification in PATH_DEFINITIONS.items():
            details[f'bcq_path_{side}_{suffix}'] = dict(specification, family='Quote path', support=path_support)
        details[f'bcq_path_{side}_comparable_n'] = dict(unit='count', definition='Number of currently fresh complete dealer states with a comparable preceding state; denominator for all three path fractions.', family='Quote path', support=path_support)
    # CONFIGURATION LOGIC: Distinguish model-error units from the normalized quote-value features themselves.
    value_unit = 'normalized '+config.value_kind+' units'
    conventions = (f'Quote spreads use quote_scale={config.quote_scale:g}; target/anchor/proxy/rollover_adjustment use target_scale={config.target_scale:g} into compatible units. '
                   f'Prediction errors alone multiply by error_scale={config.error_scale:g} to produce {config.unit}. '
                   f'Local timezone={config.timezone}; allow_exact={config.allow_exact}; age_min={config.age_min:g}; sync_min={config.sync_min:g}; lookback_min={config.lookback_min:g}. '
                   'Legacy _bps/_30m names do not override these units or windows; positive quote movement means spread widening.')
    # CORE LOGIC: STEP 4 — Require a formal definition for every actually generated quote column.
    # Input: frame columns=['row_id','bcq_has_quote','bcq_bid_center_equal_to_anchor']; details contains both base names.
    # Output: dictionary visits bcq_has_quote and bcq_bid_center_equal_to_anchor, using bcq_bid_center_equal as the latter's formula.
    # Explanation: Original source/base columns do not enter this quote-feature dictionary.
    # Trick: An undocumented new generated field raises; it never falls back to vague words copied from its name.
    records = []
    for name in [column for column in frame if column.startswith('bcq_')]:
        base = name.removesuffix('_to_anchor')
        if base not in details:
            raise ValueError(f'Missing formal feature definition for {name!r}.')
        specification = dict(details[base])
        # CORE LOGIC: STEP 5 — Add anchor subtraction, missing-value behavior, and exact ablation membership.
        # Input: name='bcq_bid_center_equal_to_anchor', groups={'Quote':['bcq_bid_center_equal_to_anchor']}, value_kind='spread'.
        # Output: models='Quote', unit='normalized spread units'; with no rollover mapping, definition adds 'Subtract the supplied prediction-time anchor from that result.'.
        # Explanation: A feature may be diagnostic-only (empty models); unsupported numeric outputs remain NaN, while known counts/indicators are zero.
        # Trick: Membership uses complete column names from actual model groups, not inference from feature prefixes.
        if name.endswith('_to_anchor'):
            formula = 'anchor minus rollover_adjustment' if config.transactions.rollover_adjustment is not None else 'anchor'
            specification['definition'] += f' Subtract the supplied prediction-time {formula} from that result.'
            specification['support'] += ' Anchor and any mapped rollover adjustment must be observable at query time; missing anchor yields NaN.'
        original_unit = specification['unit']
        specification['unit'] = value_unit if original_unit == 'quote value unit' else original_unit
        specification['missingness'] = 'Zero when no qualifying support exists.' if original_unit in {'count', 'indicator'} else 'NaN when the defined support is unavailable; a measured zero remains zero.'
        members = ', '.join(model for model, columns in groups.items() if name in columns)
        records.append(dict(feature=name, models=members, conventions=conventions, **specification))
    # REPORTING LOGIC: Stable columns make even an empty dictionary explicit and machine-readable.
    return pd.DataFrame(records, columns=['feature', 'family', 'models', 'unit', 'definition', 'support', 'missingness', 'conventions'])
