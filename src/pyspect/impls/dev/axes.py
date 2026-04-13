"""Implementation of the AxesImpl abstract base class."""

import math
from .base import Impl

__all__ = ['AxesImpl', 'Axis']


type Axis = int | str

class AxesImpl[R](Impl[R]):
    """
    Base class for axis metadata containers.

    AxesImpl is a lightweight container for axis metadata (names, bounds,
    periodicity flags, and units). It does NOT store or manipulate the underlying
    numerical data; instead it standardizes how set representations (of type `R`)
    expose axis-aware interactions (selection, projection, formatting).
    
    Construction (new):
    ```python
    AxesImpl([
        dict(name=   'x', bounds=[xmin, xmax], unit='m'),
        dict(name='*phi', bounds=[0, 2*pi],    unit='rad'),
        dict(name=   'y', bounds=[-5, 5]),  # unit optional
    ])
    ```

    Also accepted (shorthand): `AxesImpl(['x', 'y', 'z'])  # all (-inf,+inf), no units`

    Conventions:
        - Leading '*' in name => periodic axis (name stored without '*').
        - bounds omitted or Ellipsis => (-inf, +inf)
        - unit omitted => ''
        - Each axis spec must be a dict with at minimum a 'name' key (unless list[str] form used).

    Key properties / methods:
        - `ndim`: number of axes
        - `axis(ax)`: resolve axis identifier (int index or name) to int
        - `axis_name(i)`: canonical name at index i
        - `axis_bounds(ax)`: (min, max) tuple for the resolved axis
        - `axis_is_periodic(ax)`: True if marked periodic (via leading '*')
        - `project_onto(inp, axes)`: subclasses implement projection onto one or multiple axes.
    """

    def __init__(self, specs: list[dict] | list[str]):
        """Construct AxesImpl.

        Parameters:
            axes: Either a list of axis spec dicts (preferred) or list of axis name strings.
                  Dict form keys: name (required), bounds (optional), unit (optional), periodic (optional bool).
                  Periodicity may also be encoded by leading '*' in the name.
        """

        if not isinstance(specs, list):
            raise TypeError("AxesImpl expects a list[dict] or list[str]")

        names: list[str] = []
        periodicity: list[bool] = []
        min_bounds: list[float] = []
        max_bounds: list[float] = []
        units: list[str] = []

        for spec in specs:
            if isinstance(spec, str):
                spec = dict(name=spec)
            elif not isinstance(spec, dict):
                raise TypeError(f"Axis spec must be dict, got {type(spec)}")
            elif 'name' not in spec:
                raise ValueError("Axis spec missing 'name'")
            
            if not isinstance(spec['name'], str) or spec['name'] == '':
                raise TypeError("Axis 'name' must be non-empty str")
            is_periodic = spec['name'].startswith('*')
            name = spec['name'][1:] if is_periodic else spec['name']

            if bounds := spec.get('bounds', None):
                try:
                    mn, mx = float(bounds[0]), float(bounds[1])
                except (TypeError, ValueError, IndexError):
                    raise ValueError(f"Axis '{name}' bounds must be a 2-element list/tuple of numbers")
            else:
                mn, mx = -math.inf, math.inf
            
            unit = spec.get('unit', '')
            if not isinstance(unit, str):
                raise TypeError(f"Axis '{name}' unit must be a string")
            
            names.append(name)
            periodicity.append(is_periodic)
            min_bounds.append(mn)
            max_bounds.append(mx)
            units.append(unit)

        self._ndim = len(names)
        self._axis_name = tuple(names)
        self._axis_isperiodic = tuple(periodicity)
        self._min_bounds = tuple(min_bounds)
        self._max_bounds = tuple(max_bounds)
        self._units = tuple(units)

    def _axes_from_lists(self,
                         names: list[str],
                         min_bounds: list[float] = ...,
                         max_bounds: list[float] = ...,
                         units: list[str] | None = None):
        """Legacy helper: build list[dict] from separate lists and re-init."""
        if min_bounds is Ellipsis:
            min_bounds = [-float('inf')] * len(names)
        if max_bounds is Ellipsis:
            max_bounds = [ float('inf')] * len(names)
        if units is None:
            units = [''] * len(names)
        specs = [
            dict(name=name, bounds=[mn, mx], unit=unit)
            for name, mn, mx, unit in zip(names, min_bounds, max_bounds, units)
        ]
        self.__init__(specs)

    @property
    def ndim(self):
        return self._ndim

    def assert_axis(self, ax: Axis) -> None:
        """Assert that the given axis identifier is valid."""
        match ax:
            case int(i):
                assert -len(self._axis_name) <= i < len(self._axis_name), \
                    f'Axis ({i=}) does not exist.'
            case str(name):
                assert name in self._axis_name, \
                    f'Axis ({name=}) does not exist.'

    def axis(self, ax: Axis) -> int:
        """Resolve the given axis identifier to an integer index."""
        self.assert_axis(ax)
        match ax:
            case int(i):
                return i
            case str(name):
                return self._axis_name.index(name)

    def axis_name(self, i: int) -> str:
        """Get the canonical name of the axis at the given index."""
        self.assert_axis(i)
        return self._axis_name[i]
    
    def axis_unit(self, ax: Axis) -> str:
        """Get the unit string of the axis at the given index."""
        i = self.axis(ax)
        return self._units[i]

    def axis_bounds(self, ax: Axis) -> bool:
        """Get the (min, max) bounds tuple of the given axis."""
        i = self.axis(ax)
        amin = self._min_bounds[i]
        amax = self._max_bounds[i]
        return amin, amax

    def axis_is_periodic(self, ax: Axis) -> bool:
        """Return True if the given axis is marked periodic."""
        i = self.axis(ax)
        return self._axis_isperiodic[i]

    def project_onto(self, inp: R, axes: Axis | tuple[Axis, ...], **kwds) -> R:
        """Project the input set representation onto the specified axes.
        
        This method should be implemented in subclasses.
        The `axes` argument may be a single axis identifier or a tuple of them.
        Additional keyword arguments may be accepted by subclasses.
        
        Parameters:
            inp (R): The input set to project.
            axes (Axis | tuple[Axis, ...]): The axis or axes to project onto.
            **kwds: Additional keyword arguments for subclass-specific behavior.

        Returns:
            The projected set.
        """
        raise NotImplementedError("project_onto not implemented")
