"""Common utility helpers used across pyspect.

This module provides small dictionary and sequence utilities:
- iterwin: windowed iteration over indexable sequences
- setdefaults: set default values on a dict using several calling styles
- collect_keys: pick selected keys (optionally filling a default)
- collect_prefix: extract and remove keys by prefix (optionally strip prefix)
- prefix_keys: add a common prefix to all keys
- flatten: flatten nested dicts with joined keys
"""

from typing import Any, Dict, Mapping

from .idict import *


def iterwin(seq, winlen=1):
    """Yield fixed-size windows from an indexable sequence by striding.

    Equivalent to: zip(*(seq[i::winlen] for i in range(winlen))).
    For generic iterables, prefer: zip(*[iter(seq)] * winlen).

    Parameters:
    - seq: indexable sequence (supports slicing)
    - winlen: positive window/stride length

    Yields:
    - Tuples of length `winlen` with elements at positions i mod winlen
    """
    # Works for indexables; for iterables, use: zip(*[iter(seq)]*winlen)
    slices = [seq[i::winlen] for i in range(winlen)]
    yield from zip(*slices)

def setdefaults(d: dict, *args, **kwds) -> None:
    """Set default key/value pairs on dict `d` without overwriting existing keys.

    Calling conventions:
    - Keyword form: setdefaults(d, a=1, b=2)
    - Dict form:    setdefaults(d, {'a': 1, 'b': 2})
    - Variadic kv:  setdefaults(d, 'a', 1, 'b', 2)  # even number of args

    Raises:
    - TypeError/ValueError if the calling convention is invalid.
    """
    if not args:
        if not kwds:
            raise TypeError("setdefaults expected defaults")
        defaults = kwds
    elif len(args) == 1:
        (defaults,) = args
        if not isinstance(defaults, dict):
            raise TypeError("single-arg form must be a dict")
        if kwds:
            raise TypeError("cannot mix dict arg with keyword args")
    else:
        if kwds:
            raise TypeError("cannot mix variadic kv with keyword args")
        if len(args) % 2 != 0:
            raise ValueError("variadic kv form needs even number of args")
        defaults = {k: v for k, v in iterwin(args, 2)}
    for k, v in defaults.items():
        d.setdefault(k, v)

def collect_keys(d: dict, *keys, default=...):
    """Collect a subset of keys from `d`.

    Behavior:
    - If `default` is Ellipsis (the default), include only keys that exist in `d`.
    - Otherwise, include all requested `keys`, filling missing ones with `default`.

    Returns:
    - New dict containing the selected keys.
    """
    if default is Ellipsis:
        return {k: d[k] for k in keys if k in d}
    else:
        return {k: d.get(k, default) for k in keys}

def collect_prefix(d: Dict[str, Any], prefix: str, remove=False) -> Dict[str, Any]:
    """Extract and remove items whose keys start with `prefix`.

    Side effect:
    - Matching items are popped from `d`.

    Parameters:
    - d: source dictionary (mutated)
    - prefix: string to match at the start of each key
    - remove: if True, strip `prefix` from keys in the returned dict;
              if False, keep original keys

    Returns:
    - New dict of the extracted items.
    """
    if remove:
        return {k.removeprefix(prefix): d.pop(k)
                for k in list(d) if k.startswith(prefix)}
    else:
        return {k: d.pop(k)
                for k in list(d) if k.startswith(prefix)}

def prefix_keys(d: Mapping[str, Any], prefix: str) -> Dict[str, Any]:
    """Return a new dict with `prefix` added to every key in `d`."""
    return {f"{prefix}{k}": v for k, v in d.items()}

def flatten(nested: Mapping[str, Any], *, sep: str = "_", inplace: bool = False) -> Dict[str, Any]:
    """Flatten nested mappings into a single level by joining keys with `sep`.

    Example:
    - {"a": {"b": 1}, "c": 2} -> {"a_b": 1, "c": 2}  (sep="_")

    Parameters:
    - nested: mapping to flatten; nested values that are mappings are expanded
    - sep: string inserted between joined key parts
    - inplace: if True and `nested` is mutable, mutate it in place; otherwise return a new dict

    Returns:
    - A flat dict (or the mutated input when `inplace=True`).
    """
    if inplace:
        for k, v in list(nested.items()):
            if isinstance(v, Mapping):
                for kk, vv in flatten(v, sep=sep).items():
                    nested[f"{k}{sep}{kk}"] = vv
                del nested[k]
        return nested
    else:
        out: Dict[str, Any] = {}
        for k, v in nested.items():
            if isinstance(v, Mapping):
                for kk, vv in flatten(v, sep=sep).items():
                    out[f"{k}{sep}{kk}"] = vv
            else:
                out[k] = v
        return out

