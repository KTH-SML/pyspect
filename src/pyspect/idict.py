from collections.abc import Mapping
from typing import Dict, Generic, TypeVar

__all__ = (
    'idict',
)

class idict[KT, VT](Mapping):
    """Immutable Dictionary."""

    _data: Dict[KT, VT]
    _hash: int

    def __init__(self, *args, **kwargs):
        self._data = dict(*args, **kwargs)
        self._hash = None

    def __getitem__(self, key: KT) -> VT:
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return f'{self.__class__.__name__}({self._data})'

    def __hash__(self):
        if self._hash is None:
            # Hash the tuple of sorted items to ensure order doesn't matter
            self._hash = hash(tuple(sorted(self._data.items())))
        return self._hash

    def __eq__(self, other):
        if isinstance(other, idict):
            return self._data == other._data
        return False

    def __or__(self, other):
        if isinstance(other, Mapping):
            return idict(self._data | dict(other))
        return NotImplemented