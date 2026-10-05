"""Deterministic synthetic smoke data; never evidence that real quotes improve a pricer."""
# SETUP LOGIC: The demo creates no files until the caller chooses a run output directory.
import numpy as np
import pandas as pd
from .config import PipelineConfig, TransactionColumns, QuoteColumns


def demo_inputs(seed=2026):
    # CONFIGURATION LOGIC: Include a no-quote bond, multiple issuers and both short/long maturities.
    random = np.random.default_rng(seed)
    dates = pd.bdate_range('2026-01-05', periods=16, tz='America/New_York')
    bonds = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'NO_QUOTE']
    trades, quotes = [], []
    # CORE LOGIC: STEP 1 — Generate a transparent quote signal and a noisy transaction target.
    # Input: base=100, move=2, target_noise=1, dealer_bias=-.25.
    # Output: target=103, bid=102.75, ask=100.75; anchor=100.
    # Explanation: Spread bids exceed asks by two; both track the same synthetic move.
    # Trick: This deliberately informative signal tests plumbing only; its gains must never enter a real-data conclusion.
    for day in dates:
        for minute in range(0, 181, 30):
            for index, bond in enumerate(bonds):
                known = day+pd.Timedelta(hours=9, minutes=minute)
                base = 80+index*15
                move = float(random.normal(0, 3))
                target = base+move+float(random.normal())
                trades.append(dict(bond=bond, time=known+pd.Timedelta(minutes=5), y=target,
                    baseline=base, size=2e6 if index % 2 else 5e5, issuer=bond[0],
                    sector='Energy' if index < 3 else 'Bank', maturity=.5 if index == 0 else 3.0, cpp=base+.5))
                # CORE LOGIC: STEP 2 — Give every quoted bond two explicit dealer observations.
                # Input: bond='A1', base=80, move=2, biases=[-.25,.25].
                # Output: dealers D1/D2 quote bid=[82.75,83.25], ask=[80.75,81.25], size=[1MM,0].
                # Explanation: Zero size is an observed tag, not a reason to discard D2.
                # Trick: The NO_QUOTE bond gets transactions but no quotes, exercising full-population retention.
                if bond == 'NO_QUOTE':
                    continue
                for dealer, bias in [('D1', -.25), ('D2', .25)]:
                    quotes.append(dict(bond=bond, known=known, dealer=dealer,
                        bid=base+move+1+bias, ask=base+move-1+bias, size=1e6 if dealer == 'D1' else 0))
    # CORE LOGIC: STEP 3 — Add a duplicate, a same-event alternative and an incomplete latest update.
    # Input: first row bid=82, ask=80, size=1MM, known=09:00.
    # Output: duplicate at09:00; another candidate bid94/ask80 at09:00; bid='bad'/ask=NaN at09:01.
    # Explanation: One event is multiprice; the later malformed bid invalidates the current bid state.
    # Trick: The wide missing ask is absent, not an inferred cancellation; the invalid present bid remains an incomplete event.
    quotes.append(dict(quotes[0]))
    quotes.append(dict(quotes[0], bid=quotes[0]['bid']+12))
    quotes.append(dict(quotes[0], known=quotes[0]['known']+pd.Timedelta(minutes=1), bid='bad', ask=np.nan))
    # CONFIGURATION LOGIC: The same public mappings used for real inputs drive the smoke run.
    config = PipelineConfig(
        TransactionColumns('bond', 'time', 'y', issuer='issuer', sector='sector', quantity='size',
                           maturity_years='maturity', anchor='baseline', cpp='cpp'),
        QuoteColumns('bond', 'known', 'dealer', bid='bid', ask='ask', quantity='size'), synthetic=True)
    return pd.DataFrame(trades), pd.DataFrame(quotes), ['baseline', 'maturity', 'size'], config
