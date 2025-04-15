import itertools
import hj_reachability.shapes as shp
from numpy.linalg import matrix_rank
from scipy.linalg import null_space, svd, pinv
from scipy.ndimage import distance_transform_edt
from pyspect.plotting.levelsets import *

def _hz2hj(hz, min_bounds, max_bounds, grid_shape):

    # traditional -> iterate over all binary combinations
    # non-traditional -> slab-constraints
    traditional_method = False

    c, Gc, Gb, Ac, Ab, b = hz.astuple()

    nz = c.shape[0]
    ng = Gc.shape[1]
    nb = Gb.shape[1]
    nc = b.shape[0]

    # print(f'Info: {nz=}, {ng=}, {nb=}, {nc=}')

    assert  c.shape == (nz,  1), '(hz2hj) Wrong shape: c'
    assert Gc.shape == (nz, ng), '(hz2hj) Wrong shape: c'
    assert Gb.shape == (nz, nb), '(hz2hj) Wrong shape: c'
    assert Ac.shape == (nc, ng), '(hz2hj) Wrong shape: c'
    assert Ab.shape == (nc, nb), '(hz2hj) Wrong shape: c'
    assert  b.shape == (nc,  1), '(hz2hj) Wrong shape: c'
    
    NA = null_space(Ac) # ;   print('null_space(Ac)')
    Aci = pinv(Ac)      # ;   print('pinv(Ac)')

    GNi = pinv(Gc @ NA) # ;   print('pinv(Gc @ NA)')

    H = NA @ GNi            # (ng, nz)
    m = c + Gc @ Aci @ b    # (nz, 1)
    D = Gb - Gc @ Aci @ Ab  # (nz, nb)

    # print(f'{ H.shape = } ?=', (ng, nz))
    # print(f'{ m.shape = } ?=', (nz, 1))
    # print(f'{ D.shape = } ?=', (nz, nb))
    # if H.size: print(' Rank(H) =', matrix_rank(H))
    # if D.size: print(' Rank(D) =', matrix_rank(D))

    coords = [
        np.linspace(a, b, n)
        for n, a, b in zip(grid_shape, min_bounds, max_bounds)
    ]
    assert len(coords) == nz

    x = np.array(np.meshgrid(*coords, indexing='ij'))

    #       A_c^\dagger b + H (x - m) - (A_c^\dagger A_b + H D) \delta
    # (g =) M                         - K \delta

    M = (+ (Aci @ b).reshape(-1, *[1] * nz) 
         + shp.tmul(H, x - m.reshape(-1, *[1] * nz))) # (ng, ..X)

    K = Aci @ Ab + H @ D # (ng, nb)

    fixed = np.zeros((nb, *grid_shape)) # (nb, ..X)

    # fixed array has following values:
    # *   0: both axis directions (+1/-1) are free
    # *  +1: binary generators only exist on positive side of axis
    # *  -1: binary generators only exist on negative side of axis
    # * nan: there is no solution; binary generators degenerate zonotopic conditions

    # print('Branch analysis...')

    for j in range(ng):

        # *) Select g_j = _m - _k.T delta with unknown delta
        # *) Inf-norm asserts -1 <= _m - _k xib <= +1
        # *) Goal: Study upper- and lower-bounds _m - _k xib <= +1 and -1 <= _m - _K xib, respectively.
        # *) Upper- and lower-bounds form a half-space constraints
        # *) Iterate over axes (i) to find axis-aligned conditions wrt. half-space constraints
        #   *) Need to know where half-space hyperplane cuts the selected axis
        #   *) Need to know where half-space hyperplane cuts the null space (?) 

        k  = K[j]       # (nb,)
        mu = (M[j] + 1) # (..X)
        ml = (M[j] - 1) # (..X)

        ## We'll use these later

        _Ml = np.array([
            ml, 
            np.ones(grid_shape),
        ]) # (2, ..X)
        _Mu = np.array([
            mu, 
            np.ones(grid_shape),
        ]) # (2, ..X)

        for s, i in (itertools.product([+1], range(nb))):

            # unit vector
            e = np.zeros((nb,)) # (nb,)
            e[i] = 1

            _K = np.array([k, s*e]) # (2, nb)
            _Ki = pinv(_K) # (nb, 2)

            ## Compute closes point p on line directed by e

            # we seek to know if |p.T e| < 1: within +/- 1

            ##
            # *) Consider signed basis vector e
            # *) Consider hyperplanes Pl: (k, ml) and Pu: (k, mu)
            # *) Consider axis-aligned hyperplane E: e x = 1
            # *) We seek closest point p in L = intersect(P, E) to e
            # *) We seek value along axis which cuts P, i.e. v = m / (k e)
            # *) If lowerbound: k *= -1
            # *) Check cases:
            #   0) If k e = 0: skip
            #   1) If k e > 0 and v < -1                 : degenerate if sqrt(n) < |p-e| else skip
            #   2) If k e > 0 and     -1 <= v < +1       : fix delta e < 0 if sqrt(n) < |p-e| else skip
            #   3) If k e > 0 and               +1 < v   : skip
            #   4) If k e < 0 and v <= -1                : skip
            #   5) If k e < 0 and      -1 < v <= +1      : fix delta e < 0 if sqrt(n) < |p-e| else skip
            #   6) If k e < 0 and                +1 < v  : degenerate if sqrt(n) < |p-e| else skip

            d = k @ e # (1,)
            if d == 0: continue # Case 0)

            # k = k.reshape(-1, *[1]*nz) # (nb, ..X)
            e = e.reshape(-1, *[1]*nz) # (nb, ..X)

            ## Upper bound

            ub = np.zeros(grid_shape)

            v = mu / (k.flatten() @ e.flatten()) # (..X)
            p = shp.tmul(_Ki, _Mu) # (nb, ..X)

            # is_free = fixed[i] == 0.0
            is_outside = np.sqrt(nb) < np.linalg.norm(p-e, axis=0) # (..X)

            if d > 0:
                # assert np.all(is_free[_sel := is_outside & (v < -1)                ]), (
                #     f'{fixed[i].flatten()[_idx := np.argmax(_sel & ~is_free)]}'
                #     +' at (%d, %d)' % np.unravel_index(_idx, grid_shape)
                # )
                # assert np.all(is_free[_sel := is_outside &     (-1 <= v) & (v < +1)]), (
                #     f'{fixed[i].flatten()[_idx := np.argmax(_sel & ~is_free)]}'
                #     +' at (%d, %d)' % np.unravel_index(_idx, grid_shape)
                # )
                ub[is_outside & (v < -1)                ]     = np.nan    # Case 1)
                ub[is_outside &     (-1 <= v) & (v < +1)]     = -1        # Case 2)
                # skip                                                    # Case 3)
                
            if d < 0:
                # assert np.all(is_free[_sel := is_outside & (-1 < v) & (v <= +1)    ]), (
                #     f'{fixed[i].flatten()[_idx := np.argmax(_sel & ~is_free)]}'
                #     +' at (%d, %d)' % np.unravel_index(_idx, grid_shape)
                # )
                # assert np.all(is_free[_sel := is_outside &                 (+1 < v)]), (
                #     f'{fixed[i].flatten()[_idx := np.argmax(_sel & ~is_free)]}'
                #     +' at (%d, %d)' % np.unravel_index(_idx, grid_shape)
                # )
                # skip                                                     # Case 4)
                ub[is_outside & (-1 < v) & (v <= +1)    ]      = +1        # Case 5)
                ub[is_outside &                 (+1 < v)]      = np.nan    # Case 6)

            ## Lower bound
            
            lb = np.zeros(grid_shape)

            v = ml / (k.flatten() @ e.flatten()) # (..X)
            p = shp.tmul(_Ki, _Ml) # (nb, ..X)

            # is_free = fixed[i] == 0.0
            is_outside = np.sqrt(nb) < np.linalg.norm(p-e, axis=0) # (..X)

            if d > 0:
                # assert np.all(is_free[_sel := is_outside & (-1 <= v) & (v < +1)    ]), (
                #     f'{fixed[i].flatten()[_idx := np.argmax(_sel & ~is_free)]}'
                #     +' at (%d, %d)' % np.unravel_index(_idx, grid_shape)
                # )
                # assert np.all(is_free[_sel := is_outside &                 (+1 < v)]), (
                #     f'{fixed[i].flatten()[_idx := np.argmax(_sel & ~is_free)]}'
                #     +' at (%d, %d)' % np.unravel_index(_idx, grid_shape)
                # )
                # skip                                                     # Case 1)
                lb[is_outside & (-1 <= v) & (v < +1)    ]      = +1        # Case 2)
                lb[is_outside &                 (+1 < v)]      = np.nan    # Case 3)
                
            if d < 0:
                # assert np.all(is_free[_sel := is_outside & (v < -1)                ]), (
                #     f'{fixed[i].flatten()[_idx := np.argmax(_sel & ~is_free)]}'
                #     +' at (%d, %d)' % np.unravel_index(_idx, grid_shape)
                # )
                # assert np.all(is_free[_sel := is_outside &     (-1 < v) & (v <= +1)]), (
                #     f'{fixed[i].flatten()[_idx := np.argmax(_sel & ~is_free)]}'
                #     +' at (%d, %d)' % np.unravel_index(_idx, grid_shape)
                # )
                lb[is_outside & (v < -1)                ]      = np.nan    # Case 4)
                lb[is_outside &     (-1 < v) & (v <= +1)]      = -1        # Case 5)
                # skip                                                     # Case 6)
 
            ## Combining bound conditions

            fixed[i, np.isnan(ub)] = np.nan
            fixed[i, np.isnan(lb)] = np.nan
            
            # Disagreement between ub and lb, effectively doing XOR
            fixed[i, (ub * lb) == -1] = np.nan

            _good = np.abs(ub + lb) == 1
            fixed[i, _good] = ub[_good] + lb[_good]
            fixed[i, _good] = ub[_good] + lb[_good]

    # print('Branch analysis done')

    if not traditional_method:

        # Now we go through fixed list to create the sub-zero level set
        vf = np.inf * np.ones(tuple(map(len, coords)))
        
        # 1) Degenerate states
        mask = np.isnan(fixed).any(axis=0) # (..X)
        # print(f'Info: There are {mask.sum()} degenerate/ill-conditioned states')

        # 2) Iterate over the non-degenerate states
        mask = np.logical_not(mask) # (..X)
        while (n := mask.sum()) > 0:
            # print(f'{n} states left to check!')

            # only care about X idx
            idx = np.unravel_index(np.argmax(mask), mask.shape)
            
            # binary generator
            xib = fixed[(..., *idx)].reshape(nb, *[1]*nz) # (nb, ..X)

            submask = np.all(fixed == xib, axis=0) # (..X)
            # print(f'  Info: Found 1 binary generator covering {submask.sum()} states')

            # Remove collected states from mask
            mask[submask] = False

            nfree = (xib == 0).sum()
            # print('Number of free:', nfree)

            zeroes = np.where(xib == 0)[0]
            for zs in itertools.product([-1, 1], repeat=len(zeroes)):
                
                _xib = xib.copy()

                if nfree > 0:
                    _xib[zeroes] = np.array(zs).reshape(-1, *[1]*nz)

                # M: (ng, ..X)
                # K: (ng, nb)

                E = M[:, submask] - shp.tmul(K, _xib.reshape(-1, 1)) # (ng, msk)

                # narrow down submask with final condition on binary generators
                # _mask = np.zeros_like(mask)
                # _mask[submask] = np.max(np.abs(E), axis=0) <= 1
                submask[submask] = np.max(np.abs(E), axis=0) <= 1
                
                # print(f'    ==> ... of which {submask.sum()} are well-conditioned')

                # # Not necessary
                # # # Step 1: Compute distance transform
                # # mask = mask.astype(int)
                # # _vf = distance_transform_edt(mask == 0) - distance_transform_edt(mask == 1)
                # _vf = np.where(mask, -1, +1)


                vf = np.minimum(vf, np.where(submask, -1, +1))

    else:

        # Even when nb = 0, this loop will run once, and use E = M to produce the right output

        vf = np.inf * np.ones(tuple(map(len, coords)))
        for i in range(2**nb):

            delta = np.array(bin2f(i, nb)).reshape(nb, 1)

            E = M - shp.tmul(K, delta.reshape(-1, *[1] * nz))

            mask = np.max(np.abs(E), axis=0) <= 1

            # Not necessary
            # # Step 1: Compute distance transform
            # mask = mask.astype(int)
            # _vf = distance_transform_edt(mask == 0) - distance_transform_edt(mask == 1)

            vf = np.minimum(vf, np.where(mask, -1, +1))

            ninvalid = np.isnan(fixed[:, mask]).any(axis=0).sum()

            if ninvalid > 0:
                # print('Looking at:', delta.flatten())
                # print(f'INFO: {ninvalid} states falsely reported degenerate!')

                Ml = (M - 1) # (ng, ..X)
                Mu = (M + 1) # (ng, ..X)

                # # Temporary debugging
                idx = np.unravel_index(np.argmax(mask), mask.shape)
                # print('One example:', fixed[(..., *idx)], 'at (%d, %d)' % idx)
                for _i in range(ng):
                    namel = 'C_{%s}' % f'{_i+1}L'
                    nameu = 'C_{%s}' % f'{_i+1}U'
                    rhs = " + ".join(f'{k:.2f}{x}' for k, x in zip(K[_i], 'xyz'))
                    # print(f'{namel}: {Ml[(_i, *idx)]:.2f} = {rhs}')
                    # print(f'{nameu}: {Mu[(_i, *idx)]:.2f} = {rhs}')

    return vf
