from .axes import AxesImpl

class StrImpl(AxesImpl):

    def plane_cut(self, normal, offset, axes=None):
        axes = axes or list(range(self.ndim))
        assert len(normal) == len(offset) == len(axes)
        axes, normal, offset = zip(*[
            (i, k, m)
            for i, k, m in zip(axes, normal, offset)
            if k != 0 or m != 0
        ])

        return f'Plane<{list(normal)}, {list(offset)}, {list(axes)}>'

    def empty(self):
        return 'Empty'
    
    def complement(self, vf):
        return f'{vf}ꟲ'
    
    def intersect(self, vf1, vf2):
        return f'({vf1} ∩ {vf2})'

    def union(self, vf1, vf2):
        return f'({vf1} ∪ {vf2})'

    def reach(self, target, constraints=None):
        return f'Reach({target}, {constraints})'

    def avoid(self, target, constraints=None):
        return f'Avoid({target}, {constraints})'
