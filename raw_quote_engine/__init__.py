"""End-to-end known-time quote research for transaction-level regression."""
# SETUP LOGIC: Public imports; numerical work starts only when explicitly requested.
from .config import PipelineConfig, QuoteColumns, TransactionColumns
from .training import TrainingConfig
from .pipeline import ResearchRun, run_research, load_run, finalize_test
from bond_pricer import Model, Slice, compare_models, compare_predictions, show_comparison

__version__ = '0.1.1'
__all__ = ['PipelineConfig', 'QuoteColumns', 'TransactionColumns', 'TrainingConfig',
           'ResearchRun', 'run_research', 'load_run', 'finalize_test', 'Model', 'Slice',
           'compare_models', 'compare_predictions', 'show_comparison']
