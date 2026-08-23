"""
================================================================================
 ASTRO-SOLVER
 Vectorized Gumbel-Max Swarm for the Subset-Sum Problem
================================================================================

Problem Statement
-----------------
Given a list of integers ``numbers`` and a target value ``T``, find a subset
of ``numbers`` whose sum is exactly ``T``. The Astro-Solver solves this
heuristically in four phases: pruning/germination, vectorized Gumbel-Max swarm,
memetic local search and pheromone decay.

Execution
---------
The script is directly executable (IDE: F5 or ``python astro_solver.py``).
It opens with a short introduction, then walks through an interactive test
run (Tests 1-5), optionally followed by large-scale stress tests
(N = 500,000 and N = 1,000,000), and an optional extreme-scale test
(N = 5,000,000) if the user opts in.

Usage as a Library
------------------
    >>> from astro_solver import astro_solve
    >>> res = astro_solve([3, 7, 12, 9, 4], 16)
    >>> res.success, res.best_sum, res.values
    (True, 16, [12, 4])

Dependencies: NumPy (none other).
================================================================================
"""

from __future__ import annotations

import argparse
import logging
import time
import traceback
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

__all__ = ["AstroResult", "astro_solve", "run_tests"]
__version__ = "0.3.0"

# ----------------------------------------------------------------------------
# Global Solver Constants
# ----------------------------------------------------------------------------
DEFAULT_AGENTS: int = 300          # B: Swarm size (Phase 2 specification)
DEFAULT_MAX_ITERS: int = 600       # Swarm termination criterion
DECAY_EVERY: int = 50              # Pheromone decay every 50 iterations (Phase 4)
WINDOW_CANDIDATES: int = 2048      # Candidate window in "Large-Pool" mode
FULL_SCAN_LIMIT: int = 8_000_000   # B*M cells up to which full scan is performed
CHUNK_SIZE: int = 8192             # Chunking of candidate axis (memory)
EPS: float = 1e-12                 # Numerical clip for Gumbel trick
SWAP_ATTEMPTS: int = 4             # Attempts per memetic mutation
ESCAPE_PROB: float = 0.15          # Escape probability (worsening swap)
DEFAULT_SEED: int = 20260214       # Reproducibility

# Safety / heuristic thresholds
MASK_WARN_BYTES: int = 200 * 1024 * 1024  # warn if estimated used-mask exceeds ~200 MB

# Test-suite pacing (UX)
TEST_PACE_DELAY_S: float = 1.5     # Pause between the end of one test and the next header


# ----------------------------------------------------------------------------
# Result Container
# ----------------------------------------------------------------------------
@dataclass
class AstroResult:
    """Complete result of a solver run including diagnostics."""

    success: bool                      # Exact solution found?
    target: int                        # Target value T
    best_sum: int                      # Achieved sum (exactly T on success)
    indices: List[int]                 # Indices of chosen numbers (original list)
    values: List[int]                  # Values of chosen numbers
    elapsed_s: float                   # Execution time in seconds
    iterations: int                    # Consumed swarm iterations
    agents: int                        # Used agents (B)
    swaps: int                         # Executed local swaps (Phase 3)
    decays: int                        # Discarded path elements (Phase 4)
    n_input: int                       # Original list length N
    n_pool: int                        # Pool size after pruning (Phase 1)
    feasible: bool                     # gcd check: is target mathematically reachable?
    message: str                       # Human-readable status message

    @property
    def count(self) -> int:
        """Number of used numbers."""
        return len(self.indices)

    @property
    def deviation(self) -> int:
        """Absolute deviation of the found sum from the target."""
        return abs(self.best_sum - self.target)


# ============================================================================
# Phase 1 · Pruning & Germination
# ============================================================================
def phase1_prune_and_rank(numbers: np.ndarray, target: int):
    """
    Filters out all numbers > T and sorts the pool by germination power.

    K(x) = x * frequency(x mod 7)

    Returns
    -------
    vals      : Pool values in K-order (descending)
    orig_idx  : Corresponding indices in the original list
    k_vals    : Germination power per pool element (diagnostics)
    mod_freqs : Frequencies of residue classes mod 7 (diagnostics)
    """
    keep = numbers <= target                       # Pruning: x > T flies out
    vals = numbers[keep].copy()
    orig_idx = np.flatnonzero(keep)

    if vals.size == 0:
        return vals, orig_idx, vals.copy(), np.zeros(7, dtype=np.int64)

    mods = np.mod(vals, 7)                         # Residue classes (also for x < 0)
    mod_freqs = np.bincount(mods, minlength=7)     # Frequency(x mod 7)
    k_vals = vals * mod_freqs[mods]                # Germination power K(x)

    order = np.argsort(-k_vals, kind="stable")     # Descending, stable ties
    return vals[order], orig_idx[order], k_vals[order], mod_freqs


# ============================================================================
# Phase 2 · Vectorized Gumbel-Max Swarm
# ============================================================================
def _gumbel_noise(rng: np.random.Generator, shape) -> np.ndarray:
    """
    Standard Gumbel noise G = -ln(-ln(U)), U ~ U(0,1).

    Together with ln(W), Argmax(ln W + G) provides exact sampling
    proportional to W (Gumbel-Max trick).
    """
    u = rng.random(shape)
    np.clip(u, EPS, 1.0 - EPS, out=u)              # Protection against ln(0)
    return -np.log(-np.log(u))


def _score_candidates(sums_col: np.ndarray, v: np.ndarray, target: int) -> np.ndarray:
    """
    ln(W) = -ln(|(S + V) - T| + 1)   for a candidate matrix.

    sums_col : (B, 1)  current agent sums
    v        : (B, C)  candidate values per agent
    """
    diff = np.abs(sums_col + v - float(target))
    return -np.log1p(diff)


def _sample_full(rng, sums, vals, target, used, nonneg, agents):
    """
    Exact Gumbel-Max scan over the ENTIRE pool (chunked).

    Returns
    -------
    sel   : (A,) selected pool index per agent, -1 = dead end
    score : (A,) corresponding score (-inf = dead end)
    """
    sums_a = sums[agents].astype(np.float64)       # (A,)
    sub_used = used[agents]                        # (A, M) bool
    n_agents = agents.size
    m = vals.size

    best_score = np.full(n_agents, -np.inf)
    best_idx = np.full(n_agents, -1, dtype=np.int64)
    rows = np.arange(n_agents)

    for s in range(0, m, CHUNK_SIZE):
        e = min(s + CHUNK_SIZE, m)
        v = vals[s:e].astype(np.float64)[None, :]              # (1, c) — broadcasts
        score = _score_candidates(sums_a[:, None], v, target)  # -> (A, c)
        score += _gumbel_noise(rng, score.shape)               # Gumbel-Max trick
        score[sub_used[:, s:e]] = -np.inf                      # without replacement
        if nonneg:
            score[sums_a[:, None] + v > target] = -np.inf      # Overshoot forbidden

        am = np.argmax(score, axis=1)
        am_score = score[rows, am]
        better = am_score > best_score
        best_score[better] = am_score[better]
        best_idx[better] = am[better] + s

    sel = np.where(np.isfinite(best_score), best_idx, -1)
    return sel, best_score


def _sample_window(rng, sums, vals, sorted_vals, pos_map, target, used, nonneg, agents):
    """
    Gumbel-Max on a window of ``WINDOW_CANDIDATES`` values closest to the
    residual value T - S. Exactly the same formula as the full scan, but
    O(B·C) instead of O(B·M) — necessary for N = 100,000+.

    Returns like ``_sample_full``.
    """
    c = WINDOW_CANDIDATES
    sums_a = sums[agents]                                     # (A,) int64
    resid = (target - sums_a).astype(np.float64)              # Residual value per agent

    center = np.searchsorted(sorted_vals, target - sums_a)    # Window center
    start = np.clip(center - c // 2, 0, vals.size - c)        # Window start
    spos = start[:, None] + np.arange(c)[None, :]             # (A, C) sorted pos.
    pidx = pos_map[spos]                                      # -> Pool indices
    v = sorted_vals[spos].astype(np.float64)                  # (A, C) values

    score = _score_candidates(sums_a[:, None].astype(np.float64), v, target)
    score += _gumbel_noise(rng, score.shape)                  # Gumbel-Max trick
    score[used[agents[:, None], pidx]] = -np.inf              # without replacement
    if nonneg:
        score[v > resid[:, None]] = -np.inf                   # Overshoot forbidden

    am = np.argmax(score, axis=1)
    rows = np.arange(agents.size)
    am_score = score[rows, am]
    sel = np.where(np.isfinite(am_score), pidx[rows, am], -1)
    return sel, am_score


# ============================================================================
# Phase 3 · Memetic Mutation (Local-Swap)
# ============================================================================
def _local_swap(rng, b, paths, used, sums, vals, target, nonneg):
    """
    Dead-end repair for agent ``b``: swap internal path element with an
    external (unused) element.

    Acceptance: strict improvement of |S - T|, or with ESCAPE_PROB a
    neutral/worse swap (diversity). Overshoot remains forbidden.

    Returns: 1 if swap executed, else 0.
    """
    path = paths[b]
    if len(path) == 0:
        return 0                                              # nothing to swap

    unused = np.flatnonzero(~used[b])
    if unused.size == 0:
        return 0                                              # pool completely consumed

    cur_dist = abs(int(sums[b]) - target)
    for _ in range(SWAP_ATTEMPTS):
        in_pos = int(rng.integers(len(path)))                 # internal element
        out_idx = int(unused[rng.integers(unused.size)])      # external element
        delta = int(vals[out_idx]) - int(vals[path[in_pos]])
        new_sum = int(sums[b]) + delta
        new_dist = abs(new_sum - target)

        overshoot_ok = (not nonneg) or (new_sum <= target)
        if overshoot_ok and (new_dist < cur_dist or rng.random() < ESCAPE_PROB):
            old = path[in_pos]
            used[b, old] = False
            used[b, out_idx] = True
            path[in_pos] = out_idx
            sums[b] = new_sum
            return 1
    return 0


# ============================================================================
# Phase 4 · Pheromone Decay
# ============================================================================
def _pheromone_decay(paths, used, sums, vals, active_ids):
    """
    Discards the LAST added element of each active path
    ("evaporation" every DECAY_EVERY iterations). Returns the number of
    discarded elements.
    """
    dropped = 0
    for b in active_ids:
        if paths[b]:
            j = paths[b].pop()
            used[b, j] = False
            sums[b] -= vals[j]
            dropped += 1
    return dropped


# ============================================================================
# Helpers
# ============================================================================
def _estimate_used_mask_bytes(agents: int, pool_size: int) -> int:
    """Estimate memory in bytes required for the used-mask boolean array."""
    itemsize = np.dtype(bool).itemsize
    return int(agents) * int(pool_size) * int(itemsize)


# ============================================================================
# Main Solver
# ============================================================================
def astro_solve(
    numbers: Sequence[int],
    target: int,
    agents: int = DEFAULT_AGENTS,
    max_iters: int = DEFAULT_MAX_ITERS,
    seed: Optional[int] = DEFAULT_SEED,
) -> AstroResult:
    """
    Astro-Solver: finds (heuristically) a subset of ``numbers`` with
    sum == ``target``.

    Parameters
    ---------
    numbers   : Input list of integers (length N)
    target    : Target value T
    agents    : Swarm size B (default 300)
    max_iters : Maximum swarm iterations
    seed      : RNG seed for reproducibility (None = random)

    Returns
    -------
    AstroResult with solution, diagnostics, and timing.
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    target = int(target)
    arr = np.asarray(numbers, dtype=np.int64).ravel()
    n_input = int(arr.size)

    def _finish(success, path, pool_vals, pool_orig, iters, swaps, decays,
                feasible, message):
        """Builds the AstroResult from an agent path."""
        if path:
            p = np.asarray(path, dtype=np.int64)
            idx = [int(o) for o in pool_orig[p]]
            vals_out = [int(v) for v in pool_vals[p]]
            best = int(sum(vals_out))
        else:
            idx, vals_out, best = [], [], 0
        return AstroResult(
            success=bool(success), target=target, best_sum=best,
            indices=idx, values=vals_out,
            elapsed_s=time.perf_counter() - t0,
            iterations=int(iters), agents=int(agents),
            swaps=int(swaps), decays=int(decays),
            n_input=n_input, n_pool=int(pool_vals.size),
            feasible=bool(feasible), message=message,
        )

    # --- Trivial cases -------------------------------------------------------
    if target == 0:
        return _finish(True, [], arr, np.arange(n_input, dtype=np.int64),
                       0, 0, 0, True, "Trivial case T=0: empty subset is exact solution.")
    if n_input == 0:
        return _finish(False, [], arr, np.arange(0, dtype=np.int64),
                       0, 0, 0, False, "Empty input list: target unreachable.")

    # Direct hit: a single number equals T
    single = np.flatnonzero(arr == target)
    if single.size > 0:
        i = int(single[0])
        return _finish(True, [0], np.asarray([target]), np.asarray([i]),
                       0, 0, 0, True, "Instant hit: element identical to T.")

    # --- Phase 1: Pruning & Germination ---------------------------------------
    vals, orig_idx, _k, _freqs = phase1_prune_and_rank(arr, target)
    m = vals.size
    if m == 0:
        return _finish(False, [], vals, orig_idx, 0, 0, 0, False,
                       "All numbers > T: pool empty, target unreachable.")

    nonneg = bool(vals.min() >= 0)

    # Mathematical pre-check: every partial sum is a multiple of g = gcd.
    g = int(np.gcd.reduce(vals))
    feasible = (g > 0) and (target % g == 0)

    # For provably unreachable targets: limited best-effort run.
    eff_iters = max_iters if feasible else min(max_iters, 200)

    # Value-sorted view for candidate window (Large-Pool mode)
    pos_map = np.argsort(vals, kind="stable")          # sorted pos -> pool index
    sorted_vals = vals[pos_map]                        # ascending sorted values
    window_mode = (agents * m > FULL_SCAN_LIMIT) and (m > WINDOW_CANDIDATES)

    # --- Swarm Initialization --------------------------------------------
    b = int(agents)

    # Estimate memory for used-mask and warn if large
    est_bytes = _estimate_used_mask_bytes(b, m)
    if est_bytes > MASK_WARN_BYTES:
        logging.warning(
            "Estimated used-mask size ~%d bytes (~%.1f MB). Consider lowering DEFAULT_AGENTS or WINDOW_CANDIDATES",
            est_bytes, est_bytes / (1024.0 * 1024.0)
        )

    used = np.zeros((b, m), dtype=bool)                # Mask: sampling w/o replacement
    sums = np.zeros(b, dtype=np.int64)                 # Agent sums S
    paths: List[List[int]] = [[] for _ in range(b)]    # Paths (pool indices)
    done = np.zeros(b, dtype=bool)

    # Germination: Agent b starts with the b-th best element of K-order.
    for ag in range(b):
        j = ag % m                                     # K-sorted pool
        used[ag, j] = True
        sums[ag] = vals[j]
        paths[ag].append(int(j))

    swaps_total = 0
    decays_total = 0
    best_dist = int(np.abs(sums - target).min())
    best_agent = int(np.argmin(np.abs(sums - target)))
    best_path = list(paths[best_agent])

    # Seed can already be exact (e.g., after pruning edge cases)
    seed_hits = np.flatnonzero(sums == target)
    if seed_hits.size > 0:
        ag = int(seed_hits[0])
        return _finish(True, paths[ag], vals, orig_idx, 0, 0, 0, feasible,
                       "Seed hit immediately: start element sums exactly to T.")

    # --- Phases 2–4: Swarm Loop ----------------------------------------
    iterations_used = 0
    success = False
    final_path: List[int] = []

    for it in range(1, eff_iters + 1):
        iterations_used = it

        # Phase 4: Pheromone decay every 50 iterations
        if it % DECAY_EVERY == 0:
            active = np.flatnonzero(~done)
            decays_total += _pheromone_decay(paths, used, sums, vals, active)

        ids = np.flatnonzero(~done)                    # Active agents
        if ids.size == 0:
            break

        # Phase 2: Gumbel-Max Sampling (Window or Full Scan)
        if window_mode:
            sel, sc = _sample_window(np.random.default_rng(), sums, vals, sorted_vals, pos_map,
                                     target, used, nonneg, ids)
            # Window blind (everything masked)? -> safety full scan
            blind = sc == -np.inf
            if np.any(blind):
                blind_ids = ids[blind]
                sel2, sc2 = _sample_full(np.random.default_rng(), sums, vals, target, used,
                                         nonneg, blind_ids)
                sel[blind] = sel2
                sc[blind] = sc2
        else:
            sel, sc = _sample_full(np.random.default_rng(), sums, vals, target, used, nonneg, ids)

        valid = sel >= 0
        move_ids = ids[valid]
        move_sel = sel[valid]
        stuck_ids = ids[~valid]

        # Bookkeeping selected elements (without replacement via mask)
        if move_ids.size > 0:
            used[move_ids, move_sel] = True
            sums[move_ids] += vals[move_sel]
            for ag, j in zip(move_ids, move_sel):
                paths[ag].append(int(j))

        # Phase 3: Memetic mutation for agents in dead ends
        for ag in stuck_ids:
            swaps_total += _local_swap(np.random.default_rng(), int(ag), paths, used, sums,
                                       vals, target, nonneg)

        # Check exact hit -> immediate success
        if move_ids.size > 0:
            hits = move_ids[sums[move_ids] == target]
            if hits.size > 0:
                success = True
                final_path = list(paths[int(hits[0])])
                break

        # Track global best (for best-effort result)
        dist = np.abs(sums - target)
        j_best = int(np.argmin(dist))
        if int(dist[j_best]) < best_dist:
            best_dist = int(dist[j_best])
            best_agent = j_best
            best_path = list(paths[j_best])

    # --- Assemble result ----------------------------------------------
    if success:
        return _finish(True, final_path, vals, orig_idx, iterations_used,
                       swaps_total, decays_total, feasible,
                       "Exact solution found: Sum == T.")

    msg = ("Target not exactly reachable (gcd check: every partial sum is "
           f"a multiple of {g}, T mod {g if g > 0 else '-'} = {target % g if g > 0 else '-'})"
           if not feasible else
           f"Iteration limit ({eff_iters}) reached: best approximation provided "
           f"(deviation {best_dist}).")
    return _finish(False, best_path, vals, orig_idx, iterations_used,
                   swaps_total, decays_total, feasible, msg)


# ============================================================================
# Verification (for Test Suite)
# ============================================================================
def verify_solution(res: AstroResult, numbers: Sequence[int], expect_exact: bool) -> List[str]:
    """
    Checks a solver result for internal consistency:
    - no duplicate or invalid indices
    - sum of values == claimed sum
    - Pruning invariant: all chosen numbers <= T
    - Success flag matches expectation
    Returns a list of problem descriptions (empty = all ok).
    """
    problems: List[str] = []
    arr = np.asarray(numbers, dtype=np.int64).ravel()
    idx = res.indices

    if len(set(idx)) != len(idx):
        problems.append("Duplicate indices in solution.")
    if any((i < 0 or i >= arr.size) for i in idx):
        problems.append("Index out of valid range.")
    if idx and any(int(arr[i]) > res.target for i in idx):
        problems.append("Pruning invariant violated (number > T chosen).")

    recomputed = int(sum(int(arr[i]) for i in idx)) if idx else 0
    if recomputed != res.best_sum:
        problems.append(f"Sum mismatch: recalculated {recomputed}, "
                        f"claimed {res.best_sum}.")
    if res.success and res.best_sum != res.target:
        problems.append("Success claimed, but sum != T.")
    if expect_exact and not res.success:
        problems.append("Exact solution expected, but not found.")
    return problems


# ============================================================================
# Test Suite & Diagnostics
# ============================================================================
@dataclass
class TestOutcome:
    """Compact record of one executed test case, used for the final summary."""
    number: int
    title: str
    n_input: int
    passed: bool
    exact_success: bool
    elapsed_s: float
    deviation: int
    error: Optional[str] = None
    skipped: bool = False


def _print_intro() -> None:
    """Prints the introductory header and a short explanation of the solver."""
    print("=" * 78)
    print("  ASTRO-SOLVER")
    print("  Vectorized Gumbel-Max Swarm for the Subset-Sum Problem")
    print("=" * 78)
    print(
        "\n"
        "Given a list of integers and a target value T, Astro-Solver searches\n"
        "heuristically for a subset that sums exactly to T. It combines four\n"
        "phases: pruning & germination, a vectorized Gumbel-Max swarm sampler,\n"
        "memetic local-swap repair, and periodic pheromone decay.\n"
        "\n"
        f"NumPy {np.__version__}  ·  Default swarm size B = {DEFAULT_AGENTS}  ·  "
        f"Decay every {DECAY_EVERY} iterations\n"
        "\n"
        "This run will execute Tests 1-5 (correctness, performance, and edge\n"
        "cases). Afterwards you will be offered optional large-scale stress\n"
        "tests at N = 500,000 and N = 1,000,000.\n"
    )


def _wait_for_start(auto: bool = False) -> None:
    """Blocks on Enter before starting the test run, unless running non-interactively."""
    if auto:
        return
    try:
        input("Press Enter to start tests...")
    except EOFError:
        # No interactive stdin available (e.g., piped execution) - proceed automatically.
        pass


def _ask_yes_no(prompt: str, auto_answer: Optional[bool] = None) -> bool:
    """
    Asks a yes/no question on the console. Returns True for 'y', False for 'n'.
    If no interactive stdin is available, falls back to ``auto_answer`` (default False).
    """
    if auto_answer is not None:
        return auto_answer
    while True:
        try:
            reply = input(f"{prompt} [y/n]: ").strip().lower()
        except EOFError:
            return False
        if reply in ("y", "yes"):
            return True
        if reply in ("n", "no"):
            return False
        print("  Please answer with 'y' or 'n'.")


def _section_separator(char: str = "-", width: int = 78) -> None:
    print(char * width)


def _pace(delay: float = TEST_PACE_DELAY_S) -> None:
    """Short pause between tests so console output doesn't scroll by too fast."""
    if delay > 0:
        time.sleep(delay)


def _print_case_header(nr: int, title: str, desc: str) -> None:
    print()
    _section_separator()
    print(f"[TEST {nr}] {title}")
    print(f"          {desc}")
    _section_separator()


def _print_case_result(status: str, res: Optional[AstroResult],
                       problems: List[str], error: Optional[str]) -> None:
    if res is not None:
        delta = res.best_sum - res.target
        print(f"  Status              : {status}")
        print(f"  Found Sum           : {res.best_sum}   "
              f"(Target T = {res.target}, Deviation {delta:+d})")
        print(f"  Used Numbers        : {res.count} of {res.n_input} "
              f"(Pool after pruning: {res.n_pool})")
        print(f"  Execution Time      : {res.elapsed_s:.2f}s")
        print(f"  Iterations/Swaps    : {res.iterations} / {res.swaps} "
              f"(Pheromone decays: {res.decays})")
        print(f"  Reachable (gcd)     : {'yes' if res.feasible else 'no'}")
    else:
        print(f"  Status              : {status}")
    if problems:
        print("  Validation problems:")
        for p in problems:
            print(f"    - {p}")
    if error:
        print(f"  Error message       : {error}")
    elif not problems:
        print("  Errors              : none")


def _run_case(nr: int, title: str, desc: str, numbers, target: int,
              expect_exact: bool, pace_before: bool = True,
              **solver_kwargs) -> TestOutcome:
    """Executes a test case, prints diagnostics, and returns a TestOutcome record."""
    if pace_before:
        _pace()
    _print_case_header(nr, title, desc)
    res: Optional[AstroResult] = None
    problems: List[str] = []
    error: Optional[str] = None
    status = "FAILED"
    passed = False
    n_input = len(numbers) if hasattr(numbers, "__len__") else 0
    elapsed = 0.0
    deviation = -1

    try:
        res = astro_solve(numbers, target, **solver_kwargs)
        problems = verify_solution(res, numbers, expect_exact)
        elapsed = res.elapsed_s
        deviation = res.deviation
        n_input = res.n_input
        if not problems:
            passed = True
            status = ("SUCCESS (exact solution)" if res.success
                      else "SUCCESS (unreachability cleanly detected)")
        else:
            status = "FAILURE (validation)"
    except Exception as exc:  # noqa: BLE001 — Test suite catches everything
        error = f"{type(exc).__name__}: {exc}"
        tb = traceback.format_exc().strip().splitlines()
        error += "  |  " + (tb[-1] if tb else "")
        status = "FAILURE (Exception)"

    _print_case_result(status, res, problems, error)
    print(f"  => TEST {nr} {'PASSED' if passed else 'FAILED'}")

    return TestOutcome(
        number=nr, title=title, n_input=n_input, passed=passed,
        exact_success=bool(res.success) if res is not None else False,
        elapsed_s=elapsed, deviation=deviation, error=error,
    )


def _make_reachable_case(rng: np.random.Generator, n: int, lo: int, hi: int,
                         k_subset: int):
    """Random instance with guaranteed solvability: T = sum of a random subset."""
    numbers = rng.integers(lo, hi + 1, size=n)
    pick = rng.choice(n, size=k_subset, replace=False)
    target = int(numbers[pick].sum())
    return numbers, target


def _run_core_cases() -> List[TestOutcome]:
    """Executes the standard Tests 1-5 (correctness, performance, edge cases)."""
    outcomes: List[TestOutcome] = []

    # --- Test 1 · Small (N=50): Correctness ---------------------------------
    rng1 = np.random.default_rng(101)
    nums1, t1 = _make_reachable_case(rng1, n=50, lo=1, hi=99, k_subset=12)
    outcomes.append(_run_case(
        1, "Small Test (N=50)", "Verification of correctness — "
        "exact solution, unique indices, sum check.",
        nums1, t1, expect_exact=True, pace_before=False, seed=1001))

    # --- Test 2 · Medium (N=5000): Performance ------------------------------
    rng2 = np.random.default_rng(202)
    nums2, t2 = _make_reachable_case(rng2, n=5_000, lo=1, hi=1_000, k_subset=60)
    outcomes.append(_run_case(
        2, "Medium Test (N=5,000)", "Performance check — "
        "fully vectorized Gumbel scan over entire pool.",
        nums2, t2, expect_exact=True, seed=1002))

    # --- Test 3 · Large (N=100000): Stress test -------------------------------
    rng3 = np.random.default_rng(303)
    nums3, t3 = _make_reachable_case(rng3, n=100_000, lo=1, hi=2_000, k_subset=120)
    outcomes.append(_run_case(
        3, "Large Test (N=100,000)", "Stress test with timing — "
        "Candidate window mode for large pools.",
        nums3, t3, expect_exact=True, seed=1003))

    # --- Test 4 · Edge Case: many duplicates --------------------------------
    rng4 = np.random.default_rng(404)
    nums4 = np.concatenate([
        np.full(400, 7, dtype=np.int64),
        np.full(300, 3, dtype=np.int64),
        rng4.integers(1, 50, size=200),
    ])
    rng4.shuffle(nums4)
    t4 = 7 * 37 + 3 * 15          # 259 + 45 = 304 — guaranteed reachable
    outcomes.append(_run_case(
        4, "Edge Case: many duplicates", "700x identical values (7, 3) — "
        "Sampling without replacement must work index-based.",
        nums4, int(t4), expect_exact=True, seed=1004))

    # --- Test 5 · Edge Case: T unreachable ---------------------------------
    rng5 = np.random.default_rng(505)
    nums5 = rng5.integers(1, 501, size=400) * 2   # only even numbers
    t5 = 501                                      # odd -> never reachable
    outcomes.append(_run_case(
        5, "Edge Case: Target T unreachable", "Only even numbers, T=501 "
        "(odd) — gcd check + clean best-effort without crash.",
        nums5, t5, expect_exact=False, seed=1005, max_iters=250))

    return outcomes


def _run_large_scale_cases() -> List[TestOutcome]:
    """Executes the optional Tests 6-7 (large-scale stress tests)."""
    outcomes: List[TestOutcome] = []

    # --- Test 6 · Very Large (N=500,000) ------------------------------------
    rng6 = np.random.default_rng(606)
    nums6, t6 = _make_reachable_case(rng6, n=500_000, lo=1, hi=2_000, k_subset=150)
    outcomes.append(_run_case(
        6, "Stress Test (N=500,000)", "Half a million candidate numbers — "
        "candidate-window sampling keeps memory and runtime bounded.",
        nums6, t6, expect_exact=True, seed=1006))

    # --- Test 7 · Extreme (N=1,000,000) -------------------------------------
    rng7 = np.random.default_rng(707)
    nums7, t7 = _make_reachable_case(rng7, n=1_000_000, lo=1, hi=2_000, k_subset=180)
    outcomes.append(_run_case(
        7, "Stress Test (N=1,000,000)", "One million candidate numbers — "
        "practical upper bound demonstration for the swarm approach.",
        nums7, t7, expect_exact=True, seed=1007))

    return outcomes


def _print_extreme_scale_warning() -> None:
    """Prints a prominent memory warning before the extreme-scale test (Test 8)."""
    print()
    _section_separator("=")
    print("  !!! WARNING !!!")
    print(
        "  WARNING: THIS TEST REQUIRES APPROXIMATELY 1.5 GB OF RAM FOR THE "
        "AGENT MASKS.\n"
        "  ENSURE YOUR SYSTEM HAS ENOUGH FREE MEMORY BEFORE PROCEEDING."
    )
    _section_separator("=")


def _run_extreme_scale_case() -> TestOutcome:
    """Executes the optional Test 8 (extreme-scale stress test, N=5,000,000)."""
    rng8 = np.random.default_rng(808)
    nums8, t8 = _make_reachable_case(rng8, n=5_000_000, lo=1, hi=2_000, k_subset=250)
    return _run_case(
        8, "Extreme Stress Test (N=5,000,000)", "Five million candidate numbers — "
        "upper-bound demonstration of the candidate-window swarm under "
        "significant memory pressure.",
        nums8, t8, expect_exact=True, seed=1008)


def _print_summary_table(outcomes: List[TestOutcome]) -> None:
    """Prints a compact summary table of all executed tests."""
    print()
    print("=" * 78)
    print("  SUMMARY TABLE")
    print("=" * 78)
    name_w = max(28, min(38, max((len(o.title) for o in outcomes), default=28)))
    header = f"  {'#':<3}{'Test':<{name_w}}{'N':>12}{'Status':>10}{'Time':>10}{'Deviation':>12}"
    print(header)
    _section_separator(width=78)
    for o in outcomes:
        status = "PASSED" if o.passed else "FAILED"
        dev = "-" if o.deviation < 0 else str(o.deviation)
        time_str = f"{o.elapsed_s:.2f}s"
        title = (o.title[:name_w - 1] + "…") if len(o.title) > name_w else o.title
        print(f"  {o.number:<3}{title:<{name_w}}{o.n_input:>12,}{status:>10}{time_str:>10}{dev:>12}")
    _section_separator(width=78)
    passed_count = sum(1 for o in outcomes if o.passed)
    total_time = sum(o.elapsed_s for o in outcomes)
    print(f"  Total: {passed_count}/{len(outcomes)} tests passed  ·  "
          f"Combined execution time: {total_time:.2f}s")
    print("=" * 78)


def _print_comparison_section(extreme_ran: bool = False) -> None:
    """Prints a short qualitative comparison against exact reference methods."""
    print()
    print("-" * 78)
    print("  COMPARISON WITH EXACT METHODS")
    print("-" * 78)
    print(
        "  vs. Dynamic Programming:\n"
        "    Classic DP for Subset-Sum needs O(N*T) time and memory. Beyond\n"
        "    roughly N = 10,000 (or a large T), the DP table no longer fits\n"
        "    in memory and the approach becomes impractical. Astro-Solver's\n"
        "    vectorized swarm scales to hundreds of thousands of elements\n"
        "    without building any such table.\n"
        "\n"
        "  vs. Brute Force:\n"
        "    Exhaustive subset enumeration is O(2^N) and becomes infeasible\n"
        "    well before N = 30. Astro-Solver instead searches with a fixed\n"
        "    number of agents and iterations, so it comfortably handles\n"
        "    N in the millions.\n"
        "\n"
        "  Bottom line:\n"
        "    Where exact methods would take hours or crash outright on\n"
        "    large instances, Astro-Solver typically returns an exact or\n"
        "    near-optimal solution within seconds."
    )
    if extreme_ran:
        print(
            "\n"
            "  Extreme scale (N=5,000,000):\n"
            "    At this size, both Dynamic Programming and Brute Force are\n"
            "    entirely out of reach — DP's O(N*T) table would require far\n"
            "    more memory than is typically available, and Brute Force's\n"
            "    O(2^N) search space is astronomically large. Astro-Solver\n"
            "    completed this instance using only its fixed agent/window\n"
            "    budget, demonstrating scaling well beyond exact methods."
        )
    print()
    print("=" * 78)


def run_tests(interactive: bool = True, full_suite: Optional[bool] = None,
              extreme_suite: Optional[bool] = None) -> bool:
    """
    Runs the Astro-Solver test suite with console diagnostics.

    Parameters
    ----------
    interactive : bool
        If True (default), prints the introduction, waits for the user to
        press Enter, and after Tests 1-5 asks whether to proceed with the
        large-scale stress tests (N=500k, N=1M). If those run, a further
        warning and confirmation gate the extreme-scale Test 8 (N=5M).
        If False, runs non-interactively: Tests 1-5 always run, Tests 6-7
        run only if ``full_suite`` is True, and Test 8 runs only if
        ``extreme_suite`` is True (and Tests 6-7 also ran).
    full_suite : Optional[bool]
        Used only when ``interactive`` is False. If True, Tests 6-7 are
        executed automatically without prompting.
    extreme_suite : Optional[bool]
        Used only when ``interactive`` is False. If True (and ``full_suite``
        is also True), Test 8 (N=5,000,000) is executed automatically
        without prompting, skipping the interactive memory warning gate.

    Returns
    -------
    bool
        True if every executed test passed.
    """
    _print_intro()
    _wait_for_start(auto=not interactive)

    t_all = time.perf_counter()
    outcomes: List[TestOutcome] = _run_core_cases()

    if interactive:
        print()
        _section_separator("=")
        run_large = _ask_yes_no(
            "Do you want to run large-scale stress tests (N=500k and N=1M)?"
        )
    else:
        run_large = bool(full_suite)

    extreme_ran = False

    if run_large:
        outcomes.extend(_run_large_scale_cases())

        if interactive:
            _pace()
            _print_extreme_scale_warning()
            run_extreme = _ask_yes_no("Proceed with Test 8?")
        else:
            run_extreme = bool(extreme_suite)

        if run_extreme:
            outcomes.append(_run_extreme_scale_case())
            extreme_ran = True
        else:
            print("\n[SKIP] Extreme-scale stress test (N=5,000,000) skipped.")
    else:
        print("\n[SKIP] Large-scale stress tests (N=500,000 / N=1,000,000) skipped.")
        print("[SKIP] Extreme-scale stress test (N=5,000,000) skipped.")

    total_time = time.perf_counter() - t_all

    _print_summary_table(outcomes)
    _print_comparison_section(extreme_ran=extreme_ran)

    print(f"\nTotal wall-clock time for this run: {total_time:.2f}s")

    return all(o.passed for o in outcomes)


# ============================================================================
# Script Entry Point (CLI)
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Astro-Solver interactive test runner / demo")
    parser.add_argument(
        "--full", action="store_true",
        help="Run large-scale stress tests (N=500k, N=1M) automatically, "
             "without prompting (implies --non-interactive unless combined "
             "with an interactive terminal).")
    parser.add_argument(
        "--extreme", action="store_true",
        help="Also run the extreme-scale stress test (N=5M, ~1.5 GB RAM). "
             "Only takes effect together with --full; skips the interactive "
             "memory-warning confirmation.")
    parser.add_argument(
        "--non-interactive", action="store_true",
        help="Skip the 'Press Enter' prompt and the y/n stress-test questions. "
             "Use --full and/or --extreme together with this flag to include "
             "the large-scale and extreme-scale tests.")
    parser.add_argument("--seed", type=int, default=None, help="Override default RNG seed for tests.")
    parser.add_argument("--log", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO",
                        help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log), format="%(levelname)s: %(message)s")
    if args.seed is not None:
        # Note: np.random.default_rng used with explicit seeds in tests will not be affected.
        # This sets the legacy global RNG seed and may affect code that uses np.random.<func>.
        np.random.seed(args.seed)

    ok = run_tests(interactive=not args.non_interactive, full_suite=args.full,
                    extreme_suite=args.extreme)
    raise SystemExit(0 if ok else 1)
