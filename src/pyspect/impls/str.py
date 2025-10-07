"""String-based AxesImpl for demonstration and debugging.

This module provides StrImpl, an AxesImpl[str] implementation that represents
sets and set operations as human-readable strings. It is meant to illustrate
that concrete implementations can use any set representation (even plain text),
and it doubles as a lightweight pretty-printer for debugging and tests.
"""

from .axes import AxesImpl

class StrImpl(AxesImpl[str]):
    """AxesImpl that encodes all operations as formatted strings.

    Notes:
    - Outputs are not computable sets; they are descriptive strings.
    - Useful for tracing formulas, unit tests, and debugging TLT construction.
    """

    def plane_cut(self, normal, offset, axes=None):
        """Return a string describing a hyperplane cut.

        Parameters:
        - normal: iterable of coefficients along each axis
        - offset: iterable of offsets along each axis
        - axes: optional iterable of axis indices; defaults to range(self.ndim)

        Behavior:
        - Ensures normal, offset, and axes have matching lengths.
        - Drops axes where both the normal and offset are zero (no effect).
        - Returns: "Plane<[normal], [offset], [axes]>".
        """
        axes = axes or list(range(self.ndim))
        assert len(normal) == len(offset) == len(axes)
        axes, normal, offset = zip(*[
            (i, k, m)
            for i, k, m in zip(axes, normal, offset)
            if k != 0 or m != 0
        ])

        return f'Plane<{list(normal)}, {list(offset)}, {list(axes)}>'

    def empty(self):
        """Return a textual representation of the empty set."""
        return 'Empty'
    
    def complement(self, vf):
        """Return the complement as a postfix-marked string."""
        return f'{vf}ꟲ'
    
    def intersect(self, vf1, vf2):
        """Return the intersection as a parenthesized infix string."""
        return f'({vf1} ∩ {vf2})'

    def union(self, vf1, vf2):
        """Return the union as a parenthesized infix string."""
        return f'({vf1} ∪ {vf2})'

    def reach(self, target, constraints=None):
        """Return a string describing a reachability query."""
        return f'Reach({target}, {constraints})'

    def avoid(self, target, constraints=None):
        """Return a string describing an avoidability query."""
        return f'Avoid({target}, {constraints})'
