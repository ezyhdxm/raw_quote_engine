"""Content-addressed local stage cache and compact progress reporting."""
# SETUP LOGIC: Cache files are private runtime artifacts, never packaged or pushed.
import ast
import hashlib
import json
import os
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd


def code_signature(names):
    # CACHEING LOGIC: Comments do not invalidate costly stage work; executable syntax changes do.
    source = Path(__file__).parent
    trees = [ast.dump(_without_docstrings(ast.parse((source/name).read_text())), include_attributes=False) for name in names]
    return hashlib.sha256('\n'.join(trees).encode()).hexdigest()


def _without_docstrings(tree):
    # CACHEING LOGIC: Documentation literals do not change numerical behavior or need expensive recomputation.
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            node.body = [item for item in node.body if not (
                isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant) and isinstance(item.value.value, str))]
    return tree


def fingerprint(frame):
    # CACHEING LOGIC: Preserve row order, column names, dtypes and every scalar value at full precision.
    schema = repr([(str(c), str(t)) for c, t in frame.dtypes.items()]).encode()
    values = pd.util.hash_pandas_object(frame, index=True, categorize=True).to_numpy().tobytes()
    return hashlib.sha256(schema+values).hexdigest()


def portable(value):
    # SERIALIZATION LOGIC: Produce strict JSON; no executable object representations are accepted.
    if isinstance(value, dict):
        return {str(k): portable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [portable(v) for v in value]
    if isinstance(value, np.generic):
        return portable(value.item())
    if isinstance(value, np.ndarray):
        return portable(value.tolist())
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def digest(*values):
    # CACHEING LOGIC: Bind a stage to all its declared inputs and numerical settings.
    payload = json.dumps(portable(values), sort_keys=True, allow_nan=False, separators=(',', ':'))
    return hashlib.sha256(payload.encode()).hexdigest()


def write_json(path, data):
    # FILE IO LOGIC: A successful replace is the commit point; partial files are never treated as complete.
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(portable(data), indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


class StageCache:
    # CACHEING LOGIC: Only load files produced by this engine in a trusted local cache directory.
    def __init__(self, folder, progress):
        self.folder, self.progress = Path(folder), progress
        self.folder.mkdir(parents=True, exist_ok=True)

    def get(self, stage, key, calculate):
        # CACHEING LOGIC: Verify the complete payload digest before unpickling our own cached output.
        path = self.folder/f'{stage}-{key}.pkl'
        receipt = path.with_suffix('.json')
        if path.exists() and receipt.exists():
            payload = path.read_bytes()
            if hashlib.sha256(payload).hexdigest() == json.loads(receipt.read_text())['sha256']:
                self.progress(stage, 1, 1, 'Reusing completed stage')
                return pickle.loads(payload)
        self.progress(stage, None, None, 'Computing stage')
        value = calculate()
        # FILE IO LOGIC: Write data before its integrity receipt; failed/incomplete stages are recomputed.
        payload = pickle.dumps(value, protocol=5)
        temporary = path.with_suffix('.tmp')
        temporary.write_bytes(payload)
        os.replace(temporary, path)
        write_json(receipt, {'key': key, 'sha256': hashlib.sha256(payload).hexdigest()})
        self.progress(stage, 1, 1, 'Saved completed stage')
        return value


class Progress:
    # PROGRESS LOGIC: Throttle console output and allow the same callback in notebook widgets.
    def __init__(self, callback=None):
        self.started, self.last, self.stage = time.monotonic(), 0, None
        self.callback = callback

    def __call__(self, stage, done, total, detail):
        # PROGRESS LOGIC: Unknown work units remain indeterminate; no fabricated ETA or percentages.
        now = time.monotonic()
        if stage != self.stage or now-self.last >= 2 or (total is not None and done == total):
            if self.callback:
                self.callback(stage, done, total, detail)
            else:
                count = f'{done:,}/{total:,}' if done is not None and total else 'working'
                print(f'[{now-self.started:,.0f}s] {stage}: {count} — {detail}', flush=True)
            self.last, self.stage = now, stage
