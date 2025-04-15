import contextlib
import functools

from flax import struct
from jax_tqdm import scan_tqdm
from jax.experimental import io_callback
import jax
import jax.numpy as jnp
import numpy as np

import hj_reachability.shapes as shp
from hj_reachability import artificial_dissipation
from hj_reachability import time_integration
from hj_reachability.finite_differences import upwind_first

from typing import Callable, Text

# Hamiltonian postprocessors.
identity = lambda *x: x[-1]  # Returns the last argument so that this may also be used as a value postprocessor.
backwards_reachable_tube = lambda x: jnp.minimum(x, 0)

# Value postprocessors.
static_obstacle = lambda obstacle: (lambda t, v: jnp.maximum(v, obstacle))


@struct.dataclass
class SolverSettings:
    upwind_scheme: Callable = struct.field(
        default=upwind_first.WENO5,
        pytree_node=False,
    )
    artificial_dissipation_scheme: Callable = struct.field(
        default=artificial_dissipation.global_lax_friedrichs,
        pytree_node=False,
    )
    hamiltonian_postprocessor: Callable = struct.field(
        default=identity,
        pytree_node=False,
    )
    time_integrator: Callable = struct.field(
        default=time_integration.third_order_total_variation_diminishing_runge_kutta,
        pytree_node=False,
    )
    value_postprocessor: Callable = struct.field(
        default=identity,
        pytree_node=False,
    )
    CFL_number: float = 0.75

    @classmethod
    def with_accuracy(cls, accuracy: Text, **kwargs) -> "SolverSettings":
        if accuracy == "low":
            upwind_scheme = upwind_first.first_order
            time_integrator = time_integration.first_order_total_variation_diminishing_runge_kutta
        elif accuracy == "medium":
            upwind_scheme = upwind_first.ENO2
            time_integrator = time_integration.second_order_total_variation_diminishing_runge_kutta
        elif accuracy == "high":
            upwind_scheme = upwind_first.WENO3
            time_integrator = time_integration.third_order_total_variation_diminishing_runge_kutta
        elif accuracy == "very_high":
            upwind_scheme = upwind_first.WENO5
            time_integrator = time_integration.third_order_total_variation_diminishing_runge_kutta
        return cls(upwind_scheme=upwind_scheme, time_integrator=time_integrator, **kwargs)


@functools.partial(jax.jit, static_argnames=("dynamics", "progress_bar"))
def step(solver_settings, dynamics, grid, time, values, target_time, progress_bar=True):

    def sub_step(time_values):
        t, v = solver_settings.time_integrator(solver_settings, dynamics, grid, *time_values, target_time)
        return t, v

    return jax.lax.while_loop(lambda time_values: jnp.abs(target_time - time_values[0]) > 0, sub_step,
                                (time, values))[1]

# @functools.partial(jax.jit, static_argnames=("dynamics", "progress_bar"))
# def solve(solver_settings, dynamics, grid, times, target, constraints=None, preempt_saturatation=False, progress_bar=True):
#     with (_try_get_progress_bar(times[0], times[-1])
#           if progress_bar is True else contextlib.nullcontext(progress_bar)) as bar:
        
#         is_target_invariant = shp.is_invariant(grid, times, target)
#         is_constraints_invariant = shp.is_invariant(grid, times, constraints)
#         initial = target if is_target_invariant else target[0, ...]
#         if constraints is not None:
#             initial = jnp.maximum(initial, constraints if is_constraints_invariant else constraints[0, ...])
        
#         def isempty(a):
#             return jnp.all(a > 0)
        
#         i, values = 1, jnp.ones(times.shape + grid.shape)
#         values = values.at[0].set(initial)
#         def pred(val):
#             i, _ = val
#             return (0 <= i) & (i < len(times))
#         def body(val):
#             i, values = val
#             values = step(solver_settings, dynamics, grid, times[i-1], values, times[i], bar)
#             if not is_target_invariant:
#                 values = jnp.minimum(values, target[i, ...])
#             if constraints is not None:
#                 c = constraints if is_constraints_invariant else constraints[i, ...]
#                 values = jnp.maximum(values, c)
#                 if preempt_saturatation and isempty(shp.setminus(c, values)):
#                     i = -i-1
#             return (i+1, values)
#         i, values = jax.lax.while_loop(pred, body, (i, values))
#         return values

@functools.partial(jax.jit, static_argnames=("dynamics", "progress_bar"))
def solve(solver_settings, dynamics, grid, times, target, constraints=None, preempt_saturatation=False, progress_bar=True):
    with (_try_get_progress_bar(times[0], times[-1])
          if progress_bar is True else contextlib.nullcontext(progress_bar)) as bar:
        
        is_target_invariant = target.shape == grid.shape
        is_constraints_invariant = constraints is None or constraints.shape == grid.shape
        initial = target if is_target_invariant else target[0, ...]
        if constraints is not None:
            initial = jnp.maximum(initial, constraints if is_constraints_invariant else constraints[0, ...])
        
        def setminus(a, b):
            return jnp.maximum(a, -b)
        
        def isempty(a):
            return jnp.all(a > 0)
        
        @functools.partial(jax.jit, static_argnames=("grid", "dynamics", "progress_bar"))
        def ident(solver_settings, dynamics, grid, time, values, target_time, progress_bar=True):
            return values
        
        def f(carry, j):
            i, values = carry
            args = (solver_settings, dynamics, grid, times[i], values, times[j], bar)
            values = jax.lax.switch((i < 0).astype(int), [ident, step], *args)
            if not is_target_invariant:
                values = jnp.minimum(values, target[j, ...])
            if constraints is not None:
                c = constraints if is_constraints_invariant else constraints[j, ...]
                values = jnp.maximum(values, c)
                if preempt_saturatation and isempty(setminus(c, values)):
                    j *= -1
            return ((j, values), values)
        
        (i, _), values = jax.lax.scan(f, (0, initial), np.arange(1, len(times)))
        return jnp.concatenate([
            initial[np.newaxis],
            values[:jnp.abs(i)]
        ])

@functools.partial(jax.jit, static_argnames=("dynamics", "progress_bar"))
def solve(solver_settings, dynamics, grid, times, target, constraints=None, progress_bar=True):
    ctx = (_try_get_progress_bar(times[0], times[-1]) if progress_bar is True else
           contextlib.nullcontext(progress_bar))
    with ctx as bar:
        is_target_invariant = shp.is_invariant(grid, times, target)
        is_constraints_invariant = shp.is_invariant(grid, times, constraints)
        initial_values = target if is_target_invariant else target[0, ...]
        if constraints is not None:
            initial_values = jnp.maximum(initial_values, constraints if is_constraints_invariant else constraints[0, ...])
        def f(time_values, i): 
            values = step(solver_settings, dynamics, grid, *time_values, times[i], bar)
            if not is_target_invariant:
                values = jnp.minimum(values, target[i, ...])
            if not is_constraints_invariant:
                values = jnp.maximum(values, constraints[i, ...])
            elif constraints is not None:
                values = jnp.maximum(values, constraints)
            return ((times[i], values), values)
        return jnp.concatenate([
            initial_values[np.newaxis],
            jax.lax.scan(f, (times[0], initial_values), np.arange(1, len(times)))[1]
        ])
    
@functools.partial(jax.jit, static_argnames=("dynamics", "progress_bar"))
def solve(solver_settings, dynamics, grid, times, target, constraint=None, progress_bar=True):
        
    is_target_invariant = shp.is_invariant(grid, times, target)
    is_constraint_invariant = shp.is_invariant(grid, times, constraint)
    
    # ctx = (_try_get_progress_bar(times[0], times[-1]) if progress_bar is True else
    #        contextlib.nullcontext(progress_bar))
    # with ctx as bar:
    
    target = jnp.array(target)
    vf = target if is_target_invariant else target[0]
    
    if constraint is not None:
        constraint = jnp.array(constraint)
        vf = jnp.maximum(vf, constraint if is_constraint_invariant else constraint[0])
    
    if constraint is None:
        def f(carry, j):
            i, vf = carry
            vf = step(solver_settings, dynamics, grid, times[i], vf, times[j+1])
            vf = jnp.minimum(vf, target if is_target_invariant else target[j+1])
            return (j, vf), vf
    else:
        def f(carry, j):
            i, _vf = carry
            _vf = step(solver_settings, dynamics, grid, times[i], _vf, times[j+1])
            _vf = jnp.minimum(_vf, target if is_target_invariant else target[j+1])
            _vf = jnp.maximum(_vf, constraint if is_constraint_invariant else constraint[j+1])
            return (j, _vf), _vf

    if progress_bar:
        decorator = scan_tqdm(len(times)-1)
        f = decorator(f)

    return jnp.concatenate([
        vf[jnp.newaxis],
        jax.lax.scan(f, (0, vf), jnp.arange(len(times)-1))[1]
    ])

### NumPy thing ###

# def solve(solver_settings, dynamics, grid, timeline, target_vf, constraint_vf=None, progress_bar=True):
    
#     is_target_invariant = shp.is_invariant(grid, timeline, target_vf)
#     is_constraint_invariant = shp.is_invariant(grid, timeline, constraint_vf)
    
#     def scan(f, init, xs):
#         carry, ys = init, []
#         for x in xs:
#             carry, y = f(carry, x)
#             ys.append(y)
#         return np.stack(ys)

#     if constraint_vf is None:
#         def f(carry, j):
#             i, vf = carry
#             vf = step(solver_settings, dynamics, grid, timeline[i], vf, timeline[j])
#             vf = jnp.minimum(vf, target_vf if is_target_invariant else target_vf[j])
#             return (j, vf), np.array(vf)
#     else:
#         def f(carry, j):
#             i, vf = carry
#             vf = step(solver_settings, dynamics, grid, timeline[i], vf, timeline[j])
#             vf = jnp.minimum(vf, target_vf if is_target_invariant else target_vf[j])
#             vf = jnp.maximum(vf, constraint_vf if is_constraint_invariant else constraint_vf[j])
#             return (j, vf), np.array(vf)
            
#     vf = target_vf if is_target_invariant else target_vf[0]
#     if constraint_vf is not None:
#         vf = np.maximum(vf, constraint_vf if is_constraint_invariant else constraint_vf[0])
#     return np.concatenate([
#         vf[np.newaxis],
#         scan(f, (0, jnp.array(vf)), range(1, len(timeline))),
#     ])

# @functools.partial(jax.jit, static_argnames=("dynamics", "progress_bar"))
# def solve(solver_settings, dynamics, grid, times, initial_values, progress_bar=True):
#     with (_try_get_progress_bar(times[0], times[-1])
#           if progress_bar is True else contextlib.nullcontext(progress_bar)) as bar:
#         make_carry_and_output_slice = lambda t, v: ((t, v), v)
#         return jnp.concatenate([
#             initial_values[np.newaxis],
#             jax.lax.scan(
#                 lambda time_values, target_time: make_carry_and_output_slice(
#                     target_time, step(solver_settings, dynamics, grid, *time_values, target_time, bar)),
#                 (times[0], initial_values), times[1:])[1]
#         ])


def add_progress_bar(f, steps, style='auto'):
    decorator = scan_tqdm(
        t0,
        total=jnp.abs(t1 - t0),
        unit="sim_s",
        bar_format="{l_bar}{bar}| {n:7.4f}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        ascii=True
    )
    return decorator(f)

def _try_get_progress_bar(reference_time, target_time):
    try:
        import tqdm
    except ImportError:
        raise ImportError("The option `progress_bar=True` requires the 'tqdm' package to be installed.")
    return TqdmWrapper(tqdm,
                       reference_time,
                       total=jnp.abs(target_time - reference_time),
                       unit="sim_s",
                       bar_format="{l_bar}{bar}| {n:7.4f}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
                       ascii=True)


class TqdmWrapper:

    _tqdm = None

    def __init__(self, tqdm, reference_time, total, *args, **kwargs):
        self.reference_time = reference_time
        callback = lambda total: self._create_tqdm(tqdm, total, *args, **kwargs)
        io_callback(callback, None, total)

    def _create_tqdm(self, tqdm, total, *args, **kwargs):
        self._tqdm = tqdm.tqdm(total=total, *args, **kwargs)

    def update_to(self, n):
        callback = lambda n: self._tqdm.update(n - self._tqdm.n) if self._tqdm is not None else None
        io_callback(callback, None, n)
        return n

    def close(self):
        callback = lambda: self._tqdm.close() if self._tqdm is not None else None
        io_callback(callback, None)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
