"""Clearly synthetic data for learning the workbench; never evidence of research gains."""
# SETUP LOGIC: Demo generation is deterministic and needs no private files or trained models.
import numpy as np
import pandas as pd


def make_demo(n=1200, seed=2026):
    # TEST LOGIC: Simulate correlated predictions and heterogeneous populations solely for interface examples.
    rng = np.random.default_rng(seed)
    actual = rng.normal(110,25,n)
    reference_error = rng.normal(0,2,n)
    sector = rng.choice(['Energy','Financials','Technology','Industrials'],n)
    candidate_error = .85*reference_error+rng.normal(0,.6,n)+np.where(sector=='Energy',1,0)
    return pd.DataFrame(dict(trade_id=np.arange(n),actual=actual,BASE=actual+reference_error,
                             Candidate=actual+candidate_error,SECTOR=sector,ISSUER=rng.choice(['Issuer A','Issuer B','Issuer C','Issuer D','Issuer E'],n),
                             CUSIP=rng.choice([f'SYNTH{i:03d}' for i in range(30)],n),
                             QUANTITY=rng.choice([100000,500000,1000000,2000000,5000000],n),
                             YRS_TO_MATURITY=rng.uniform(.1,15,n),new_feature=rng.normal(0,1,n),
                             time=pd.date_range('2026-04-01 08:00',periods=n,freq='15min',tz='America/New_York')))
