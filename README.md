# Astro-Solver

A vectorized heuristic solver for the Subset-Sum problem, designed to handle large input sizes (N > 10,000) where exact dynamic programming becomes computationally prohibitive due to memory constraints.

## Overview

The Subset-Sum problem is NP-complete. While exact algorithms (e.g., dynamic programming or meet-in-the-middle) guarantee optimal solutions, their time and space complexity scale poorly with large, sparse datasets. 

Astro-Solver addresses this by employing a hybrid metaheuristic approach. It combines heuristic search space reduction with a vectorized swarm intelligence algorithm, utilizing the Gumbel-Max trick for efficient categorical sampling and memetic local search to escape local optima.

## Algorithm Architecture

The solver operates in four distinct phases:

1. **Pruning & Heuristic Ranking**: Filters elements strictly greater than the target `T`. The remaining pool is sorted by a "germination score" `K(x) = x * frequency(x mod 7)`, prioritizing large values from frequently occurring modulo classes.
2. **Vectorized Gumbel-Max Swarm**: Maintains a population of `B` agents building subsets concurrently. At each iteration, agents sample the next element using the Gumbel-Max trick: `argmax(ln(W) - ln(-ln(U)))`, where `W` is inversely proportional to the distance to the target. This allows for O(1) categorical sampling per agent without cumulative distribution functions. For large pools, this is optimized via a sliding window around the residual value `T - S`.
3. **Memetic Local Search**: Agents that reach a dead end (no valid moves remaining) trigger a local repair mechanism. They attempt to swap an internal path element with an unused external element to reduce the distance to `T`, occasionally accepting non-improving swaps to maintain diversity.
4. **Pheromone Decay**: Periodically drops the last added element from active agent paths to prevent premature convergence and encourage exploration of alternative branches.

## Performance Benchmarks

Tests were executed on a standard consumer CPU using NumPy vectorization.

| Input Size (N) | Target Type | Status | Execution Time | Iterations |
| :--- | :--- | :--- | :--- | :--- |
| 50 | Exact reachable | Success | 0.018 s | 9 |
| 5,000 | Exact reachable | Success | 5.689 s | 55 |
| 100,000 | Exact reachable | Success | 3.463 s | 61 |
| 400 (Even nums) | Unreachable (Odd T) | Best-Effort | 2.123 s | 200 |

*Note: The 100,000-item benchmark utilizes the sliding window optimization, reducing the effective search space per agent and preventing memory exhaustion.*

## Installation

Requires Python 3.8+ and NumPy.

```bash
pip install numpy
```

## Usage

```python
from astro_solver import astro_solve

# Example: Cargo weights and truck capacity
weights = [3, 7, 12, 9, 4, 15, 22]
capacity = 28

result = astro_solve(weights, capacity)

if result.success:
    print(f"Exact solution found: {result.values} (Sum: {result.best_sum})")
else:
    print(f"Best effort: {result.values} (Sum: {result.best_sum}, Deviation: {result.deviation})")
    print(f"Reason: {result.message}")
```

## Testing

The repository includes a comprehensive built-in test suite validating correctness, performance, and edge cases (e.g., duplicates, mathematically unreachable targets via GCD checks).

To run the test suite:

```bash
python astro_solver.py
```
