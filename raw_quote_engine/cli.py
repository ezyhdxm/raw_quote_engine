"""Small command-line entry point for JSON-configured runs and a clearly labelled synthetic smoke run."""
# SETUP LOGIC: No work is performed when the module is imported.
import argparse
import json
from pathlib import Path
from .pipeline import run_research, load_run, finalize_test
from .demo import demo_inputs
from .training import TrainingConfig


def main():
    # UI LOGIC: Required inputs are declared in one JSON file rather than a large set of positional flags.
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run', help='Read raw files, run Steps 1–5 and generate the report')
    run.add_argument('--config', required=True)
    run.add_argument('--final-test', action='store_true', help='Freeze on validation, then evaluate only that candidate on test')
    demo = commands.add_parser('demo', help='Run deterministic synthetic data; no real-data conclusions')
    demo.add_argument('--output', default='runs/synthetic_demo')
    final = commands.add_parser('final-test', help='Evaluate the existing frozen choice without re-training')
    final.add_argument('output')
    refresh = commands.add_parser('report', help='Regenerate the saved report without raw inputs or training')
    refresh.add_argument('output')
    args = parser.parse_args()
    # ORCHESTRATION LOGIC: Subcommands use the exact same API as the notebook.
    if args.command == 'demo':
        tx, quotes, base, config = demo_inputs()
        result = run_research(tx, quotes, base, {'n_estimators': 25, 'n_jobs': 2, 'min_child_samples': 10},
                              config, output=args.output, evaluate_test=True)
    elif args.command == 'final-test':
        result = finalize_test(args.output)
    elif args.command == 'report':
        from .report import write_report
        result = load_run(args.output)
        write_report(result.output, result.tables, result.comparisons, result.metadata)
    else:
        result = run_file(args.config, args.final_test)
    # UI LOGIC: Print only the artifact location, never a large results table.
    print(f'Open {result.report}')


def run_file(path, evaluate_test=False):
    # FILE IO LOGIC: Resolve relative data/output paths beside the config file, including on Windows.
    path = Path(path).expanduser().resolve()
    specification = json.loads(path.read_text(encoding='utf-8'))
    def resolve(value):
        # FILE IO LOGIC: Absolute paths remain absolute; relative paths are configuration-relative.
        return path.parent/Path(value).expanduser()
    return run_research(resolve(specification['transactions_path']), resolve(specification['quotes_path']),
                        specification['base_features'], specification.get('model_params', {}), specification['config'],
                        base_cat_features=specification.get('base_cat_features', []),
                        output=resolve(specification.get('output', 'runs/research')),
                        training=TrainingConfig(**specification.get('training', {})), evaluate_test=evaluate_test)


# CLI LOGIC: Direct module execution invokes one explicit subcommand.
if __name__ == '__main__':
    main()
