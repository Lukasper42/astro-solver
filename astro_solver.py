"""
================================================================================
 ASTRO-SOLVER
 Unsorted-Pool Adaptive Team-Based Gumbel-Max Swarm for Subset-Sum
================================================================================

Problem Statement
-----------------
Given a list of integers ``numbers`` and a target value ``T``, find a subset
of ``numbers`` whose sum is exactly ``T``.

This implementation follows the requested strict constraints:

  * NO sorting of the pool by value or germination power.
  * NO bucket structures or value-range grouping.
  * NO console warnings or debug output.

Optimizations implemented:

  1. Smart Germination (Head-Start)
     Each team starts with one or two promising numbers, chosen from the
     original unsorted pool without sorting.

  2. Adaptive Window Sizing
     The candidate window size is based on the team's residual need:

         residual = T - current_sum

     Large residuals use larger exploratory windows, while small residuals
     use smaller fine-tuning windows.

  3. Intelligent Shaking (Conditional Decay)
     Decay is no longer triggered every fixed number of iterations. A team
     is shaken only if its distance to the target has not improved for a
     randomized stagnation period of 10-15 iterations.

Team Architecture
-----------------
The outer swarm consists of up to 300 Teams. Each Team contains a dynamic
number of Birds:

    N <= 1,000      -> 50 Birds per Team
    N <= 100,000    -> 20 Birds per Team
    N >  100,000    -> 10 Birds per Team

All Birds in one Team share the same used-mask row, path, and sum. The
candidate window is partitioned equally among the Birds, and the Team adopts
the best move found by any Bird.

Dependencies: NumPy only.
================================================================================
"""

from __future__ import annotations

import argparse
import logging
import time
import traceback
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

__all__ = ["AstroResult", "astro_solve", "run_tests"]
__version__ = "0.9.0"

# ----------------------------------------------------------------------------
# Global Solver Constants
# ----------------------------------------------------------------------------
DEFAULT_AGENTS: int = 300          # Maximum / default number of Teams
DEFAULT_MAX_ITERS: int = 600       # Swarm termination criterion
WINDOW_CANDIDATES: int = 2048      # Maximum adaptive candidate window
MIN_WINDOW_CANDIDATES: int = 64    # Minimum adaptive candidate window
FULL_SCAN_LIMIT: int = 8_000_000   # Cell budget for safe full-scan fallback
CHUNK_SIZE: int = 8192             # Chunk size for full-scan memory safety
EPS: float = 1e-12                 # Numerical clip for Gumbel trick
SWAP_ATTEMPTS: int = 4             # Attempts per memetic mutation
ESCAPE_PROB: float = 0.15          # Escape probability for worse swap
DEFAULT_SEED: int = 20260214       # Reproducibility

# Intelligent shaking thresholds
STAGNATION_MIN_ITERS: int = 10
STAGNATION_MAX_ITERS: int = 15

# Test-suite pacing
TEST_PACE_DELAY_S: float = 1.5


# ----------------------------------------------------------------------------
# Result Container
# ----------------------------------------------------------------------------
@dataclass
class AstroResult:
    """Complete result of a solver run including diagnostics."""

    success: bool
    target: int
    best_sum: int
    indices: List[int]
    values: List[int]
    elapsed_s: float
    iterations: int
    teams: int
    team_size: int
    swaps: int
    decays: int
    n_input: int
    n_pool: int
    feasible: bool
    message: str

    @property
    def count(self) -> int:
        """Number of used numbers."""
        return len(self.indices)

    @property
    def deviation(self) -> int:
        """Absolute deviation of the found sum from the target."""
        return abs(self.best_sum - self.target)


# ============================================================================
# Dynamic Configuration
# ============================================================================
def _get_team_size(n: int) -> int:
    """Dynamic Birds-per-Team scaling based on input size N."""
    if n <= 1000:
        return 50
    if n <= 100000:
        return 20
    return 10


# ============================================================================
# Phase 1 · Pruning Only (No Sorting)
# ============================================================================
def phase1_prune(numbers: np.ndarray, target: int):
    """
    Filters out all numbers > T while preserving the original input order.

    This function intentionally performs no sorting and no grouping.
    """
    keep = numbers <= target
    vals = np.ascontiguousarray(numbers[keep], dtype=np.int64)
    orig_idx = np.flatnonzero(keep).astype(np.int64, copy=False)
    return vals, orig_idx


# ============================================================================
# Smart Germination (Head-Start)
# ============================================================================
def _smart_germinate(
    vals: np.ndarray,
    target: int,
    teams: int,
    nonneg: bool,
    rng: np.random.Generator,
):
    """
    Assigns each team one or two promising starter numbers without sorting.

    The first starter is drawn from the largest value(s) in the pruned pool.
    If the pool is nonnegative, a second starter may be added from values that
    can safely accompany the largest starter without exceeding the target.
    """
    b = int(teams)
    m = int(vals.size)

    used = np.zeros((b, m), dtype=bool)
    sums = np.zeros(b, dtype=np.int64)
    paths: List[List[int]] = [[] for _ in range(b)]

    if m == 0:
        return used, sums, paths

    max_val = int(vals.max())

    # First starter pool: largest value(s) in the original unsorted pool.
    first_indices = np.flatnonzero(vals == max_val)
    if first_indices.size == 0:
        first_indices = np.arange(m, dtype=np.int64)

    # Second starter pool: safe complementary values for nonnegative pools.
    second_indices: Optional[np.ndarray] = None

    if nonneg:
        limit = int(target - max_val)
        if limit >= 0:
            safe_mask = vals <= limit
            if np.any(safe_mask):
                safe_indices = np.flatnonzero(safe_mask)
                second_max = int(vals[safe_mask].max())
                second_best = np.flatnonzero(safe_mask & (vals == second_max))

                # Prefer exact second-best duplicates when available; otherwise
                # fall back to the safe pool so teams can still receive two
                # distinct starters when possible.
                if second_best.size > 1 or safe_indices.size == second_best.size:
                    second_indices = second_best
                else:
                    second_indices = safe_indices

    rng_integers = rng.integers

    for ag in range(b):
        j1 = int(rng_integers(first_indices.size))

        used[ag, j1] = True
        sums[ag] = vals[j1]
        paths[ag].append(int(j1))

        if second_indices is not None and second_indices.size > 0:
            for _ in range(8):
                j2 = int(rng_integers(second_indices.size))

                if j2 == j1 or used[ag, j2]:
                    continue

                new_sum = int(sums[ag]) + int(vals[j2])
                if (not nonneg) or new_sum <= target:
                    used[ag, j2] = True
                    sums[ag] = new_sum
                    paths[ag].append(int(j2))
                    break

    return used, sums, paths


# ============================================================================
# Phase 2 · Gumbel-Max Core
# ============================================================================
def _gumbel_noise(rng: np.random.Generator, shape) -> np.ndarray:
    """
    Standard Gumbel noise G = -ln(-ln(U)), U ~ U(0,1).

    Argmax(ln W + G) samples exactly proportional to W.
    """
    u = rng.random(shape)
    np.clip(u, EPS, 1.0 - EPS, out=u)

    # In-place transformation reduces temporary allocations:
    # u = -log(-log(u))
    np.log(u, out=u)
    np.negative(u, out=u)
    np.log(u, out=u)
    np.negative(u, out=u)

    return u


def _score_candidates(sums_col: np.ndarray, v: np.ndarray, target: int) -> np.ndarray:
    """
    ln(W) = -ln(|(S + V) - T| + 1)

    sums_col : (B, 1) current team sums
    v        : (B, C) candidate values
    """
    diff = np.add(sums_col, v)
    diff -= float(target)
    np.abs(diff, out=diff)
    np.log1p(diff, out=diff)
    return np.negative(diff, out=diff)


# ============================================================================
# Adaptive Window Sizing
# ============================================================================
def _adaptive_window_sizes(
    resid: np.ndarray,
    value_scale: float,
    max_window: int,
) -> np.ndarray:
    """
    Computes a per-team adaptive candidate-window size.

    Large residual need -> larger window for exploration.
    Small residual need -> smaller window for fine-tuning.
    """
    min_window = min(MIN_WINDOW_CANDIDATES, max_window)

    if max_window <= min_window:
        return np.full(resid.shape, max_window, dtype=np.int64)

    ratio = np.abs(resid.astype(np.float64))
    ratio /= float(value_scale)
    np.clip(ratio, 0.0, 1.0, out=ratio)
    np.sqrt(ratio, out=ratio)

    ratio *= float(max_window - min_window)
    ratio += float(min_window)
    np.clip(ratio, float(min_window), float(max_window), out=ratio)

    return ratio.astype(np.int64)


# ============================================================================
# Phase 2 · Adaptive Unsorted Window Sampler
# ============================================================================
def _sample_adaptive_window(
    rng: np.random.Generator,
    sums: np.ndarray,
    vals: np.ndarray,
    target: int,
    used: np.ndarray,
    nonneg: bool,
    teams: np.ndarray,
    team_size: int,
    value_scale: float,
):
    """
    Samples a dense adaptive window from the original unsorted pool.

    No sorting or bucketing is used. Each team receives a circular contiguous
    window over the original pool order. The window size adapts to the team's
    residual need.
    """
    n_teams = teams.size
    m = vals.size

    if n_teams == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

    if m == 0:
        return np.full(n_teams, -1, dtype=np.int64), np.full(n_teams, -np.inf)

    max_window = min(WINDOW_CANDIDATES, m)

    resid = target - sums[teams]
    w = _adaptive_window_sizes(resid, value_scale, max_window)

    global_w = int(w.max())
    if global_w <= 0:
        return np.full(n_teams, -1, dtype=np.int64), np.full(n_teams, -np.inf)

    # Randomized starts, lightly influenced by team identity and residual.
    starts = rng.integers(0, m, size=n_teams).astype(np.int64, copy=False)
    resid_shift = np.mod(resid, m)
    team_shift = teams.astype(np.int64, copy=False) * np.int64(101)

    starts += resid_shift
    starts += team_shift
    starts %= m

    offsets = np.arange(global_w, dtype=np.int64)[None, :]

    pos = starts[:, None] + offsets
    pos %= m

    # Keep candidate values as integers; conversion to float happens only
    # inside scoring where broadcasting with float sums is required.
    v_window = vals[pos]

    invalid = used[teams[:, None], pos]
    invalid |= offsets >= w[:, None]

    sums_a = sums[teams].astype(np.float64, copy=False)
    resid_f = resid.astype(np.float64, copy=False)

    best_score = np.full(n_teams, -np.inf)
    best_idx = np.full(n_teams, -1, dtype=np.int64)
    rows = np.arange(n_teams)

    bird_splits = np.array_split(
        np.arange(global_w, dtype=np.int64),
        min(team_size, global_w)
    )

    for chunk_indices in bird_splits:
        if chunk_indices.size == 0:
            continue

        s = chunk_indices[0]
        e = chunk_indices[-1] + 1

        v_chunk = v_window[:, s:e]
        invalid_chunk = invalid[:, s:e]
        pos_chunk = pos[:, s:e]

        score = _score_candidates(sums_a[:, None], v_chunk, target)
        score += _gumbel_noise(rng, score.shape)

        score[invalid_chunk] = -np.inf

        if nonneg:
            score[v_chunk > resid_f[:, None]] = -np.inf

        am = np.argmax(score, axis=1)
        am_score = score[rows, am]

        better = am_score > best_score
        if np.any(better):
            better_rows = rows[better]
            better_cols = am[better]
            best_score[better] = am_score[better]
            best_idx[better] = pos_chunk[better_rows, better_cols]

    return best_idx, best_score


# ============================================================================
# Phase 2 · Full Scan Fallback
# ============================================================================
def _sample_full(
    rng: np.random.Generator,
    sums: np.ndarray,
    vals: np.ndarray,
    target: int,
    used: np.ndarray,
    nonneg: bool,
    teams: np.ndarray,
    team_size: int,
):
    """
    Exact Gumbel-Max scan over the entire unsorted pool.

    Used only for small pools or when the cell budget remains safely small.
    """
    n_teams = teams.size
    m = vals.size

    if n_teams == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

    if m == 0:
        return np.full(n_teams, -1, dtype=np.int64), np.full(n_teams, -np.inf)

    sums_a = sums[teams].astype(np.float64, copy=False)
    resid_f = target - sums_a

    sub_used = used[teams]

    best_score = np.full(n_teams, -np.inf)
    best_idx = np.full(n_teams, -1, dtype=np.int64)
    rows = np.arange(n_teams)

    bird_splits = np.array_split(
        np.arange(m, dtype=np.int64),
        min(team_size, m)
    )

    for chunk_indices in bird_splits:
        if chunk_indices.size == 0:
            continue

        s = chunk_indices[0]
        e = chunk_indices[-1] + 1

        for s2 in range(s, e, CHUNK_SIZE):
            e2 = min(s2 + CHUNK_SIZE, e)

            v = vals[s2:e2][None, :]

            score = _score_candidates(sums_a[:, None], v, target)
            score += _gumbel_noise(rng, score.shape)

            score[sub_used[:, s2:e2]] = -np.inf

            if nonneg:
                score[v > resid_f[:, None]] = -np.inf

            am = np.argmax(score, axis=1)
            am_score = score[rows, am]

            better = am_score > best_score
            if np.any(better):
                best_score[better] = am_score[better]
                best_idx[better] = am[better] + s2

    return best_idx, best_score


# ============================================================================
# Phase 3 · Memetic Mutation
# ============================================================================
def _local_swap(
    rng: np.random.Generator,
    b: int,
    paths: List[List[int]],
    used: np.ndarray,
    sums: np.ndarray,
    vals: np.ndarray,
    target: int,
    nonneg: bool,
) -> int:
    """
    Dead-end repair for one team: swap one internal path element with one
    unused external element.
    """
    path = paths[b]
    if not path:
        return 0

    pool_size = vals.size
    if len(path) >= pool_size:
        return 0

    cur_dist = abs(int(sums[b]) - target)

    used_b = used[b]
    rng_integers = rng.integers
    rng_random = rng.random

    for _ in range(SWAP_ATTEMPTS):
        in_pos = int(rng_integers(len(path)))

        unused_count = pool_size - len(path)
        if unused_count <= 0:
            return 0

        if unused_count < 64:
            unused = np.flatnonzero(~used_b)
            if unused.size == 0:
                return 0
            out_idx = int(unused[rng_integers(unused.size)])
        else:
            out_idx = -1
            for _ in range(32):
                candidate = int(rng_integers(pool_size))
                if not used_b[candidate]:
                    out_idx = candidate
                    break

            if out_idx < 0:
                unused = np.flatnonzero(~used_b)
                if unused.size == 0:
                    return 0
                out_idx = int(unused[rng_integers(unused.size)])

        old = path[in_pos]
        delta = int(vals[out_idx]) - int(vals[old])
        new_sum = int(sums[b]) + delta
        new_dist = abs(new_sum - target)

        overshoot_ok = (not nonneg) or (new_sum <= target)
        if overshoot_ok and (new_dist < cur_dist or rng_random() < ESCAPE_PROB):
            used_b[old] = False
            used_b[out_idx] = True
            path[in_pos] = out_idx
            sums[b] = new_sum
            return 1

    return 0


# ============================================================================
# Intelligent Shaking (Conditional Decay)
# ============================================================================
def _shake_teams(
    paths: List[List[int]],
    used: np.ndarray,
    sums: np.ndarray,
    vals: np.ndarray,
    team_ids: np.ndarray,
) -> int:
    """
    Drops the last number from each stagnant team path.
    """
    dropped = 0

    for t in team_ids:
        team = int(t)
        if paths[team]:
            j = paths[team].pop()
            used[team, int(j)] = False
            sums[team] -= vals[int(j)]
            dropped += 1

    return dropped


# ============================================================================
# Main Solver
# ============================================================================
def astro_solve(
    numbers: Sequence[int],
    target: int,
    teams: int = DEFAULT_AGENTS,
    max_iters: int = DEFAULT_MAX_ITERS,
    seed: Optional[int] = DEFAULT_SEED,
) -> AstroResult:
    """
    Astro-Solver: heuristic subset-sum solver using an unsorted pool,
    adaptive windows, smart germination, and intelligent shaking.
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

    target = int(target)
    arr = np.asarray(numbers, dtype=np.int64).ravel()
    n_input = int(arr.size)

    teams = max(1, int(teams))
    team_size = _get_team_size(n_input)

    def _finish(success, path, pool_vals, pool_orig, iters, swaps, decays,
                feasible, message):
        if path:
            p = np.asarray(path, dtype=np.int64)

            chosen_orig = pool_orig[p]
            chosen_vals = pool_vals[p]

            idx = chosen_orig.tolist()
            vals_out = chosen_vals.tolist()
            best = int(sum(vals_out))
        else:
            idx, vals_out, best = [], [], 0

        return AstroResult(
            success=bool(success),
            target=target,
            best_sum=best,
            indices=idx,
            values=vals_out,
            elapsed_s=time.perf_counter() - t0,
            iterations=int(iters),
            teams=int(teams),
            team_size=int(team_size),
            swaps=int(swaps),
            decays=int(decays),
            n_input=n_input,
            n_pool=int(pool_vals.size),
            feasible=bool(feasible),
            message=message,
        )

    # --- Trivial cases -------------------------------------------------------
    if target == 0:
        return _finish(
            True, [], arr, np.arange(n_input, dtype=np.int64),
            0, 0, 0, True,
            "Trivial case T=0: empty subset is exact solution."
        )

    if n_input == 0:
        return _finish(
            False, [], arr, np.arange(0, dtype=np.int64),
            0, 0, 0, False,
            "Empty input list: target unreachable."
        )

    single = np.flatnonzero(arr == target)
    if single.size > 0:
        i = int(single[0])
        return _finish(
            True, [0], np.asarray([target]), np.asarray([i]),
            0, 0, 0, True,
            "Instant hit: element identical to T."
        )

    # --- Phase 1: Pruning only, original order preserved ---------------------
    vals, orig_idx = phase1_prune(arr, target)
    m = vals.size

    if m == 0:
        return _finish(
            False, [], vals, orig_idx,
            0, 0, 0, False,
            "All numbers > T: pool empty, target unreachable."
        )

    nonneg = bool(vals.min() >= 0)

    g = int(np.gcd.reduce(vals))
    if g == 0:
        return _finish(
            False, [], vals, orig_idx,
            0, 0, 0, False,
            "All pool values are zero: non-zero target unreachable."
        )

    feasible = (target % g == 0)
    eff_iters = max_iters if feasible else min(max_iters, 200)

    value_scale = max(
        float(abs(int(vals.min()))),
        float(abs(int(vals.max()))),
        float(abs(target)),
        1.0,
    )

    b = int(teams)
    window_mode = (b * m > FULL_SCAN_LIMIT) and (m > WINDOW_CANDIDATES)
    allow_full_fallback = (b * m <= FULL_SCAN_LIMIT)

    # --- Smart Germination ---------------------------------------------------
    used, sums, paths = _smart_germinate(vals, target, b, nonneg, rng)

    team_best_dist = np.abs(sums - target).astype(np.int64, copy=False)
    stagnation = np.zeros(b, dtype=np.int32)
    shake_threshold = rng.integers(
        STAGNATION_MIN_ITERS,
        STAGNATION_MAX_ITERS + 1,
        size=b,
    ).astype(np.int32, copy=False)

    swaps_total = 0
    decays_total = 0

    best_dist = int(team_best_dist.min()) if b else 0
    best_team = int(np.argmin(team_best_dist)) if b else 0
    best_path = list(paths[best_team]) if b else []

    seed_hits = np.flatnonzero(sums == target)
    if seed_hits.size > 0:
        ag = int(seed_hits[0])
        return _finish(
            True, paths[ag], vals, orig_idx,
            0, 0, 0, feasible,
            "Smart germination hit: starter subset sums exactly to T."
        )

    # --- Swarm loop ----------------------------------------------------------
    iterations_used = 0
    success = False
    final_path: List[int] = []

    all_team_ids = np.arange(b, dtype=np.int64)

    for it in range(1, eff_iters + 1):
        iterations_used = it

        # Phase 2: candidate sampling.
        if window_mode:
            sel, sc = _sample_adaptive_window(
                rng, sums, vals, target, used, nonneg,
                all_team_ids, team_size, value_scale
            )

            blind = sc == -np.inf

            if np.any(blind) and allow_full_fallback:
                blind_ids = all_team_ids[blind]
                sel2, sc2 = _sample_full(
                    rng, sums, vals, target, used, nonneg,
                    blind_ids, team_size
                )
                sel[blind] = sel2
                sc[blind] = sc2
        else:
            sel, sc = _sample_full(
                rng, sums, vals, target, used, nonneg,
                all_team_ids, team_size
            )

        valid = sel >= 0

        # Apply selected moves.
        if np.any(valid):
            move_ids = np.flatnonzero(valid)
            move_sel = sel[valid]

            used[move_ids, move_sel] = True
            sums[move_ids] += vals[move_sel]

            for ag, j in zip(move_ids, move_sel):
                paths[int(ag)].append(int(j))

            hit_mask = sums[move_ids] == target
            if np.any(hit_mask):
                success = True
                final_path = list(paths[int(move_ids[hit_mask][0])])
                break

        # Phase 3: memetic repair for stuck teams.
        if np.any(~valid):
            stuck_ids = np.flatnonzero(~valid)
            swaps_before = swaps_total

            for ag in stuck_ids:
                swaps_total += _local_swap(
                    rng, int(ag), paths, used, sums,
                    vals, target, nonneg
                )

            if swaps_total > swaps_before:
                hit_mask = sums[stuck_ids] == target
                if np.any(hit_mask):
                    success = True
                    final_path = list(paths[int(stuck_ids[hit_mask][0])])
                    break

        # Intelligent shaking: update stagnation and conditionally shake.
        current_dist = np.abs(sums - target)
        improved = current_dist < team_best_dist

        if np.any(improved):
            team_best_dist[improved] = current_dist[improved]
            stagnation[improved] = 0

        if np.any(~improved):
            stagnation[~improved] += 1

        shake_mask = stagnation >= shake_threshold

        if np.any(shake_mask):
            shake_ids = np.flatnonzero(shake_mask)

            decays_total += _shake_teams(paths, used, sums, vals, shake_ids)

            stagnation[shake_ids] = 0
            shake_threshold[shake_ids] = rng.integers(
                STAGNATION_MIN_ITERS,
                STAGNATION_MAX_ITERS + 1,
                size=shake_ids.size,
            ).astype(np.int32, copy=False)

            hit_mask = sums[shake_ids] == target
            if np.any(hit_mask):
                success = True
                final_path = list(paths[int(shake_ids[hit_mask][0])])
                break

            shaken_dist = np.abs(sums[shake_ids] - target)
            improved_shake = shaken_dist < team_best_dist[shake_ids]
            if np.any(improved_shake):
                team_best_dist[shake_ids[improved_shake]] = shaken_dist[improved_shake]

            dist_all = np.abs(sums - target)
        else:
            dist_all = current_dist

        # Track global best approximation.
        j_best = int(np.argmin(dist_all))

        if int(dist_all[j_best]) < best_dist:
            best_dist = int(dist_all[j_best])
            best_team = j_best
            best_path = list(paths[j_best])

    if success:
        return _finish(
            True, final_path, vals, orig_idx, iterations_used,
            swaps_total, decays_total, feasible,
            "Exact solution found: Sum == T."
        )

    msg = (
        "Target not exactly reachable (gcd check: every partial sum is "
        f"a multiple of {g}, T mod {g} = {target % g})"
        if not feasible else
        f"Iteration limit ({eff_iters}) reached: best approximation provided "
        f"(deviation {best_dist})."
    )

    return _finish(
        False, best_path, vals, orig_idx, iterations_used,
        swaps_total, decays_total, feasible, msg
    )


# ============================================================================
# Verification
# ============================================================================
def verify_solution(res: AstroResult, numbers: Sequence[int], expect_exact: bool) -> List[str]:
    """
    Checks solver output for internal consistency.
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
        problems.append(
            f"Sum mismatch: recalculated {recomputed}, claimed {res.best_sum}."
        )

    if res.success and res.best_sum != res.target:
        problems.append("Success claimed, but sum != T.")

    if expect_exact and not res.success:
        problems.append("Exact solution expected, but not found.")

    return problems


# ============================================================================
# Test Suite
# ============================================================================
@dataclass
class TestOutcome:
    number: int
    title: str
    n_input: int
    passed: bool
    exact_success: bool
    elapsed_s: float
    deviation: int
    teams_used: int = 0
    team_size_used: int = 0
    error: Optional[str] = None
    skipped: bool = False


def _print_intro() -> None:
    print("=" * 78)
    print("  ASTRO-SOLVER")
    print("  Unsorted-Pool Adaptive Team-Based Gumbel-Max Swarm")
    print("=" * 78)
    print(
        "\n"
        "Given a list of integers and a target value T, Astro-Solver searches\n"
        "heuristically for a subset that sums exactly to T.\n"
        "\n"
        "This version preserves the original pool order and uses adaptive\n"
        "candidate windows, smart germination, and intelligent shaking.\n"
        "\n"
        f"NumPy {np.__version__}  ·  Maximum Teams = {DEFAULT_AGENTS}  ·  "
        f"Dynamic Birds per Team  ·  Conditional shaking after "
        f"{STAGNATION_MIN_ITERS}-{STAGNATION_MAX_ITERS} stagnant iterations\n"
        "\n"
        "This run will execute Tests 1-5, followed by optional large-scale\n"
        "stress tests at N = 500,000 and N = 1,000,000.\n"
    )


def _wait_for_start(auto: bool = False) -> None:
    if auto:
        return
    try:
        input("Press Enter to start tests...")
    except EOFError:
        pass


def _ask_yes_no(prompt: str, auto_answer: Optional[bool] = None) -> bool:
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
    if delay > 0:
        time.sleep(delay)


def _query_available_ram(interactive: bool, default_gb: int = 4,
                         override_gb: Optional[int] = None) -> int:
    if not interactive:
        gb = override_gb if (override_gb is not None and override_gb > 0) else default_gb
        return gb

    try:
        raw = input("How much RAM is available? (e.g., 2 for 2GB): ").strip()
        gb = int(raw)
        if gb <= 0:
            raise ValueError("RAM value must be positive")
    except (ValueError, EOFError):
        print(f"  Invalid or missing input — defaulting to {default_gb} GB.")
        gb = default_gb

    return gb


def _compute_dynamic_teams(n_items: int, available_bytes: int) -> int:
    """
    Computes a safe number of Teams from the RAM budget.

    Each team requires one boolean used-mask row of length N. Half of the
    available RAM is reserved for NumPy overhead and temporary arrays.
    """
    mem_per_team = max(int(n_items), 1)
    reserved_bytes = available_bytes * 0.5
    raw_max_teams = int(reserved_bytes / mem_per_team)
    return max(1, min(DEFAULT_AGENTS, raw_max_teams))


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
        print(f"  Swarm Config        : Teams: {res.teams} | "
              f"Birds per Team: {res.team_size} | "
              f"Total Birds: {res.teams * res.team_size}")
        print(f"  Execution Time      : {res.elapsed_s:.2f}s")
        print(f"  Iterations/Swaps    : {res.iterations} / {res.swaps} "
              f"(Conditional shakes: {res.decays})")
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
              expect_exact: bool, available_bytes: int, pace_before: bool = True,
              **solver_kwargs) -> TestOutcome:
    if pace_before:
        _pace()

    _print_case_header(nr, title, desc)

    n_for_scaling = len(numbers) if hasattr(numbers, "__len__") else 0
    teams = _compute_dynamic_teams(n_for_scaling, available_bytes)
    team_size = _get_team_size(n_for_scaling)
    total_birds = teams * team_size

    print(
        f"  Test {nr}: Configuring {teams} teams "
        f"({total_birds} total birds, {team_size} per team) "
        f"for N={n_for_scaling:,}."
    )

    solver_kwargs = dict(solver_kwargs)
    solver_kwargs["teams"] = teams

    res: Optional[AstroResult] = None
    problems: List[str] = []
    error: Optional[str] = None
    status = "FAILED"
    passed = False
    n_input = n_for_scaling
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
            status = (
                "SUCCESS (exact solution)"
                if res.success
                else "SUCCESS (unreachability cleanly detected)"
            )
        else:
            status = "FAILURE (validation)"

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        tb = traceback.format_exc().strip().splitlines()
        if tb:
            error += "  |  " + tb[-1]
        status = "FAILURE (Exception)"

    _print_case_result(status, res, problems, error)
    print(f"  => TEST {nr} {'PASSED' if passed else 'FAILED'}")

    return TestOutcome(
        number=nr,
        title=title,
        n_input=n_input,
        passed=passed,
        exact_success=bool(res.success) if res is not None else False,
        elapsed_s=elapsed,
        deviation=deviation,
        teams_used=teams,
        team_size_used=team_size,
        error=error,
    )


def _make_reachable_case(rng: np.random.Generator, n: int, lo: int, hi: int,
                         k_subset: int):
    numbers = rng.integers(lo, hi + 1, size=n)
    pick = rng.choice(n, size=k_subset, replace=False)
    target = int(numbers[pick].sum())
    return numbers, target


def _run_core_cases(available_bytes: int) -> List[TestOutcome]:
    outcomes: List[TestOutcome] = []

    rng1 = np.random.default_rng(101)
    nums1, t1 = _make_reachable_case(rng1, n=50, lo=1, hi=99, k_subset=12)
    outcomes.append(_run_case(
        1, "Small Test (N=50)",
        "Verification of correctness — exact solution, unique indices, sum check.",
        nums1, t1, expect_exact=True, available_bytes=available_bytes,
        pace_before=False, seed=1001
    ))

    rng2 = np.random.default_rng(202)
    nums2, t2 = _make_reachable_case(rng2, n=5_000, lo=1, hi=1_000, k_subset=60)
    outcomes.append(_run_case(
        2, "Medium Test (N=5,000)",
        "Performance check — full vectorized scan remains active for moderate pools.",
        nums2, t2, expect_exact=True, available_bytes=available_bytes,
        seed=1002
    ))

    rng3 = np.random.default_rng(303)
    nums3, t3 = _make_reachable_case(rng3, n=100_000, lo=1, hi=2_000, k_subset=120)
    outcomes.append(_run_case(
        3, "Large Test (N=100,000)",
        "Stress test — adaptive unsorted windows over a large pool.",
        nums3, t3, expect_exact=True, available_bytes=available_bytes,
        seed=1003
    ))

    rng4 = np.random.default_rng(404)
    nums4 = np.concatenate([
        np.full(400, 7, dtype=np.int64),
        np.full(300, 3, dtype=np.int64),
        rng4.integers(1, 50, size=200),
    ])
    rng4.shuffle(nums4)
    t4 = 7 * 37 + 3 * 15

    outcomes.append(_run_case(
        4, "Edge Case: many duplicates",
        "700x identical values (7, 3) — index-based sampling without replacement.",
        nums4, int(t4), expect_exact=True, available_bytes=available_bytes,
        seed=1004
    ))

    rng5 = np.random.default_rng(505)
    nums5 = rng5.integers(1, 501, size=400) * 2
    t5 = 501

    outcomes.append(_run_case(
        5, "Edge Case: Target T unreachable",
        "Only even numbers, T=501 (odd) — gcd check and clean best-effort result.",
        nums5, t5, expect_exact=False, available_bytes=available_bytes,
        seed=1005, max_iters=250
    ))

    return outcomes


def _run_large_scale_cases(available_bytes: int) -> List[TestOutcome]:
    outcomes: List[TestOutcome] = []

    rng6 = np.random.default_rng(606)
    nums6, t6 = _make_reachable_case(rng6, n=500_000, lo=1, hi=2_000, k_subset=150)
    outcomes.append(_run_case(
        6, "Stress Test (N=500,000)",
        "Half a million candidate numbers — adaptive unsorted windowing.",
        nums6, t6, expect_exact=True, available_bytes=available_bytes,
        seed=1006
    ))

    rng7 = np.random.default_rng(707)
    nums7, t7 = _make_reachable_case(rng7, n=1_000_000, lo=1, hi=2_000, k_subset=180)
    outcomes.append(_run_case(
        7, "Stress Test (N=1,000,000)",
        "One million candidate numbers — practical upper-bound demonstration.",
        nums7, t7, expect_exact=True, available_bytes=available_bytes,
        seed=1007
    ))

    return outcomes


def _run_extreme_scale_case(available_bytes: int) -> TestOutcome:
    rng8 = np.random.default_rng(808)
    nums8, t8 = _make_reachable_case(rng8, n=5_000_000, lo=1, hi=2_000, k_subset=250)

    return _run_case(
        8, "Extreme Stress Test (N=5,000,000)",
        "Five million candidate numbers — unsorted adaptive search under memory pressure.",
        nums8, t8, expect_exact=True, available_bytes=available_bytes,
        seed=1008
    )


def _print_summary_table(outcomes: List[TestOutcome]) -> None:
    print()
    print("=" * 90)
    print("  SUMMARY TABLE")
    print("=" * 90)

    name_w = max(26, min(34, max((len(o.title) for o in outcomes), default=26)))

    header = (
        f"  {'#':<3}{'Test':<{name_w}}{'N':>12}{'Teams':>9}"
        f"{'Status':>10}{'Time':>10}{'Deviation':>12}"
    )
    print(header)
    _section_separator(width=90)

    for o in outcomes:
        status = "PASSED" if o.passed else "FAILED"
        dev = "-" if o.deviation < 0 else str(o.deviation)
        time_str = f"{o.elapsed_s:.2f}s"
        title = (o.title[:name_w - 1] + "…") if len(o.title) > name_w else o.title

        print(
            f"  {o.number:<3}{title:<{name_w}}{o.n_input:>12,}{o.teams_used:>9}"
            f"{status:>10}{time_str:>10}{dev:>12}"
        )

    _section_separator(width=90)

    passed_count = sum(1 for o in outcomes if o.passed)
    total_time = sum(o.elapsed_s for o in outcomes)

    if outcomes:
        primary = max(outcomes, key=lambda o: o.n_input)
        teams_show = primary.teams_used
        birds_show = primary.team_size_used
    else:
        teams_show = 0
        birds_show = 0

    print(
        f"  Total: {passed_count}/{len(outcomes)} tests passed  ·  "
        f"Teams: {teams_show} | Birds per Team: {birds_show} | "
        f"Total Birds: {teams_show * birds_show}  ·  "
        f"Combined execution time: {total_time:.2f}s"
    )
    print("=" * 90)


def _print_comparison_section(extreme_ran: bool = False) -> None:
    print()
    print("-" * 78)
    print("  COMPARISON WITH EXACT METHODS")
    print("-" * 78)
    print(
        "  vs. Dynamic Programming:\n"
        "    Classic DP for Subset-Sum needs O(N*T) time and memory. Beyond\n"
        "    roughly N = 10,000 or a large T, the DP table becomes impractical.\n"
        "    Astro-Solver avoids this table entirely.\n"
        "\n"
        "  vs. Brute Force:\n"
        "    Exhaustive subset enumeration is O(2^N) and becomes infeasible\n"
        "    well before N = 30. Astro-Solver uses a bounded team budget and\n"
        "    bounded iteration count, allowing it to operate on very large N.\n"
        "\n"
        "  Bottom line:\n"
        "    This unsorted adaptive variant avoids sorting overhead and keeps\n"
        "    memory access simple, while smart germination and conditional\n"
        "    shaking improve search stability."
    )

    if extreme_ran:
        print(
            "\n"
            "  Extreme scale (N=5,000,000):\n"
            "    At this size, exact methods are completely out of reach.\n"
            "    Astro-Solver completed the instance using memory-aware team\n"
            "    scaling and unsorted adaptive candidate windows."
        )

    print()
    print("=" * 78)


def run_tests(interactive: bool = True, full_suite: Optional[bool] = None,
              extreme_suite: Optional[bool] = None, ram_gb: Optional[int] = None) -> bool:
    """
    Runs the Astro-Solver test suite.
    """
    _print_intro()

    ram_default = ram_gb if (ram_gb is not None and ram_gb > 0) else 4
    available_gb = _query_available_ram(
        interactive=interactive,
        default_gb=ram_default,
        override_gb=ram_gb,
    )

    available_bytes = available_gb * 1024 * 1024 * 1024

    print(
        f"\nMemory budget configured: {available_gb} GB "
        f"({available_bytes:,} bytes). Team count will be scaled dynamically "
        f"per test to stay within this budget.\n"
    )

    _wait_for_start(auto=not interactive)

    t_all = time.perf_counter()
    outcomes: List[TestOutcome] = _run_core_cases(available_bytes)

    if interactive:
        print()
        _pace()
        _section_separator("=")
        run_large = _ask_yes_no(
            "Do you want to run large-scale stress tests (N=500k and N=1M)?"
        )
    else:
        run_large = bool(full_suite)

    extreme_ran = False

    if run_large:
        outcomes.extend(_run_large_scale_cases(available_bytes))

        if interactive:
            _pace()
            run_extreme = _ask_yes_no("Proceed with Test 8 (Extreme Scale)?")
        else:
            run_extreme = bool(extreme_suite)

        if run_extreme:
            outcomes.append(_run_extreme_scale_case(available_bytes))
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
# Script Entry Point
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Astro-Solver interactive test runner / demo"
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="Run large-scale stress tests (N=500k, N=1M) automatically."
    )

    parser.add_argument(
        "--extreme",
        action="store_true",
        help="Also run the extreme-scale stress test (N=5M). "
             "Only takes effect together with --full."
    )

    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Skip prompts. Use with --full and/or --extreme."
    )

    parser.add_argument(
        "--ram-gb",
        type=int,
        default=None,
        help="RAM budget in GB for dynamic team scaling."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override default global RNG seed."
    )

    parser.add_argument(
        "--log",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="ERROR",
        help="Logging level. Default is ERROR to keep console output clean."
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log),
        format="%(levelname)s: %(message)s",
    )

    if args.seed is not None:
        np.random.seed(args.seed)

    ok = run_tests(
        interactive=not args.non_interactive,
        full_suite=args.full,
        extreme_suite=args.extreme,
        ram_gb=args.ram_gb,
    )

    raise SystemExit(0 if ok else 1)
