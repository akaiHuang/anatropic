"""
Benchmark the 3D self-gravity Jeans run on the NumPy and MLX backends.

Runs the Simulation3D driver (gravity-Strang + dimensional sweeps + HLLE
Riemann + FFT Poisson) for a fixed number of timesteps on a sequence of
cube grids and reports wall time per step and the MLX speedup over
NumPy. Each backend is launched in a subprocess so ``ANATROPIC_BACKEND``
is honoured at module import time.

Usage::

    python3.11 benchmarks/bench_3d.py
    python3.11 benchmarks/bench_3d.py --grids 64 96 --steps 30

Notes
-----
* MLX runs in float32 / complex64 (the only precisions Metal supports);
  NumPy runs in float64 / complex128. The two backends therefore solve
  the same equations at different precisions, which is the entire point
  of the comparison.
* The first MLX run includes one-shot Metal kernel compilation. We add
  a short warm-up step before timing.
* Timings are printed to stdout as a Markdown table so they can be
  pasted into the README.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import textwrap
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _child_script(grid: int, steps: int) -> str:
    """Return the Python source the subprocess will execute."""
    return textwrap.dedent(f"""
        import json, time, sys
        sys.path.insert(0, {str(REPO_ROOT)!r})

        from anatropic._backend import BACKEND
        from anatropic.eos import IsothermalEOS
        from anatropic.simulation3d import Simulation3D
        import anatropic._backend as _be

        N = {grid}
        n_steps = {steps}

        # ---- Warm-up: one full step (MLX kernel compile, NumPy cache prime) ----
        warm = Simulation3D()
        warm.setup(Nx=N, Ny=N, Nz=N, Lx=1.0, Ly=1.0, Lz=1.0, rho0=1.0,
                   eos=IsothermalEOS(cs=0.05))
        warm.add_random_perturbation(amplitude=1e-3, seed=0)
        warm.run(t_end=1e9, cfl=0.3, max_steps=1, print_every=10**9)
        _be.evaluate(warm.state.rho)
        del warm

        # ---- Timed run ----
        sim = Simulation3D()
        sim.setup(Nx=N, Ny=N, Nz=N, Lx=1.0, Ly=1.0, Lz=1.0, rho0=1.0,
                  eos=IsothermalEOS(cs=0.05))
        sim.add_random_perturbation(amplitude=1e-3, seed=42)

        t0 = time.perf_counter()
        sim.run(t_end=1e9, cfl=0.3, max_steps=n_steps, print_every=10**9)
        _be.evaluate(sim.state.rho, sim.state.vx, sim.state.vy,
                     sim.state.vz, sim.state.eint)
        elapsed = time.perf_counter() - t0

        out = {{
            "backend": BACKEND,
            "grid": N,
            "steps": int(sim.step),
            "elapsed_s": float(elapsed),
            "per_step_ms": float(1000.0 * elapsed / max(sim.step, 1)),
        }}
        print("BENCH_RESULT_JSON " + json.dumps(out))
    """)


def run_backend(backend: str, grid: int, steps: int, timeout_s: int = 1800):
    env = os.environ.copy()
    env["ANATROPIC_BACKEND"] = backend
    code = _child_script(grid, steps)

    proc = subprocess.run(
        ["python3.11", "-c", code],
        env=env, capture_output=True, text=True, timeout=timeout_s,
    )
    if proc.returncode != 0:
        print(f"  [{backend} N={grid}] FAILED:")
        print(proc.stdout)
        print("STDERR:", proc.stderr)
        return None

    for line in proc.stdout.splitlines():
        if line.startswith("BENCH_RESULT_JSON "):
            return json.loads(line[len("BENCH_RESULT_JSON "):])

    print(f"  [{backend} N={grid}] no BENCH_RESULT_JSON line in stdout:")
    print(proc.stdout)
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grids", type=int, nargs="+", default=[64, 96, 128],
        help="Cubic grid sizes to test (default: 64 96 128).",
    )
    parser.add_argument(
        "--steps", type=int, default=50,
        help="Timed steps per (backend, grid) point (default: 50).",
    )
    parser.add_argument(
        "--backends", nargs="+", default=["numpy", "mlx"],
        help="Backends to run (default: numpy mlx).",
    )
    args = parser.parse_args()

    print(f"# Anatropic 3D backend benchmark")
    print(f"# host:   {platform.machine()} / {platform.processor()} / "
          f"{platform.system()} {platform.release()}")
    print(f"# python: {platform.python_version()}")
    print(f"# steps per measurement: {args.steps}")
    print()

    rows = []
    for grid in args.grids:
        for backend in args.backends:
            print(f"[run] backend={backend:5s}  N={grid:4d}  steps={args.steps} ...",
                  flush=True)
            result = run_backend(backend, grid, args.steps)
            if result is None:
                continue
            rows.append(result)
            print(f"      {result['elapsed_s']:7.3f} s total"
                  f"   {result['per_step_ms']:7.2f} ms/step")
        print()

    # Pivot into a markdown table: rows by grid, cols by backend.
    by_grid = {}
    for r in rows:
        by_grid.setdefault(r["grid"], {})[r["backend"]] = r

    print("## Results")
    print()
    print("| Grid | NumPy ms/step | MLX ms/step | MLX speedup |")
    print("|------|---------------|-------------|-------------|")
    for grid in sorted(by_grid):
        np_r = by_grid[grid].get("numpy")
        mx_r = by_grid[grid].get("mlx")
        np_s = f"{np_r['per_step_ms']:.1f}" if np_r else "-"
        mx_s = f"{mx_r['per_step_ms']:.1f}" if mx_r else "-"
        if np_r and mx_r:
            spd = f"{np_r['per_step_ms'] / mx_r['per_step_ms']:.2f}x"
        else:
            spd = "-"
        print(f"| {grid}^3 | {np_s} | {mx_s} | {spd} |")

    print()
    print("Raw results (for archival):")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
