import numpy as np
import hj_reachability as hj

class AxesImpl:
    
    def __init__(self, names, min_bounds=..., max_bounds=...):
        names, is_periodic = zip(*[
            (name[1:], True) if name[0] == '*' else (name, False)
            for name in names 
        ])
        self._ndim = len(names)
        self._axis_name = names
        self._axis_isperiodic = is_periodic
        self._min_bounds = (min_bounds if min_bounds is not Ellipsis else
                            [-float('inf')] * self._ndim)
        self._max_bounds = (max_bounds if max_bounds is not Ellipsis else
                            [+float('inf')] * self._ndim)
        
    @property
    def ndim(self):
        return self._ndim

    def assert_axis(self, ax: int | str) -> None:
        match ax:
            case int(i):
                assert -len(self._axis_name) <= i < len(self._axis_name), \
                    f'Axis ({i=}) does not exist.'
            case str(name):
                assert name in self._axis_name, \
                    f'Axis ({name=}) does not exist.'

    def axis(self, ax: int | str) -> int:
        self.assert_axis(ax)
        match ax:
            case int(i):
                return i
            case str(name):
                return self._axis_name.index(name)

    def axis_name(self, i: int) -> str:
        self.assert_axis(i)
        return self._axis_name[i]

    def axis_bounds(self, ax: int | str) -> bool:
        i = self.axis(ax)
        amin = self._min_bounds[i]
        amax = self._max_bounds[i]
        return amin, amax

    def axis_is_periodic(self, ax: int | str) -> bool:
        i = self.axis(ax)
        return self._axis_isperiodic[i]
