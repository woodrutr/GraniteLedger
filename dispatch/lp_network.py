"""Linear programming dispatch with a multi-region transmission network."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple, cast

try:  # pragma: no cover - optional dependency guard
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = cast(Any, None)

from io_loader import Frames

from .interface import DispatchResult
from .lp_single import HOURS_PER_YEAR

_TOL = 1e-9


def _ensure_pandas() -> None:
    """Ensure :mod:`pandas` is available before executing pandas-dependent logic."""

    if pd is None:  # pragma: no cover - helper exercised indirectly
        raise ImportError(
            "pandas is required for dispatch.lp_network; install it with `pip install pandas`."
        )


@dataclass(frozen=True)
class GeneratorSpec:
    """Specification of an individual generator in the dispatch problem."""

    name: str
    region: str
    fuel: str
    variable_cost: float
    capacity: float
    emission_rate: float
    covered: bool = True

    def marginal_cost(self, allowance_cost: float) -> float:
        """Return the effective marginal cost including any allowance payments."""

        carbon_cost = self.emission_rate * allowance_cost if self.covered else 0.0
        return self.variable_cost + carbon_cost


def _normalize_interfaces(
    interfaces: Mapping[Tuple[str, str], float] | Iterable[Tuple[Tuple[str, str], float]],
) -> Dict[Tuple[str, str], float]:
    """Normalize an interface mapping to use sorted region pairs with symmetric limits."""

    normalized: Dict[Tuple[str, str], float] = {}
    items = interfaces.items() if isinstance(interfaces, Mapping) else interfaces

    for regions, limit in items:
        if len(regions) != 2:
            raise ValueError('Interface keys must contain exactly two regions.')

        region_a, region_b = regions
        if region_a == region_b:
            raise ValueError('Interfaces must connect two distinct regions.')

        limit = float(limit)
        if limit < 0.0:
            raise ValueError('Transfer capability must be non-negative.')

        key = tuple(sorted((region_a, region_b)))
        if key in normalized:
            if abs(normalized[key] - limit) > _TOL:
                raise ValueError(
                    f'Conflicting limits specified for interface between {region_a} and {region_b}.',
                )
        else:
            normalized[key] = limit

    return normalized


def _matrix_rank(matrix: List[List[float]], tol: float = _TOL) -> int:
    """Return the rank of a matrix using Gaussian elimination."""

    if not matrix or not matrix[0]:
        return 0

    working = [row[:] for row in matrix]
    rows = len(working)
    cols = len(working[0])
    rank = 0

    for col in range(cols):
        pivot = None
        for row in range(rank, rows):
            if abs(working[row][col]) > tol:
                pivot = row
                break
        if pivot is None:
            continue

        working[rank], working[pivot] = working[pivot], working[rank]
        pivot_val = working[rank][col]
        working[rank] = [value / pivot_val for value in working[rank]]

        for row in range(rows):
            if row != rank and abs(working[row][col]) > tol:
                factor = working[row][col]
                working[row] = [
                    value - factor * working[rank][idx] for idx, value in enumerate(working[row])
                ]

        rank += 1
        if rank == rows:
            break

    return rank


def _solve_linear_system(
    matrix: List[List[float]],
    rhs: List[float],
    tol: float = _TOL,
) -> List[float] | None:
    """Solve ``matrix @ x = rhs`` using Gaussian elimination."""

    if not matrix:
        return [] if all(abs(value) <= tol for value in rhs) else None

    rows = len(matrix)
    cols = len(matrix[0])
    augmented = [row[:] + [rhs[idx]] for idx, row in enumerate(matrix)]

    pivot_row = 0
    pivot_cols: List[int] = []
    for col in range(cols):
        pivot = None
        for row in range(pivot_row, rows):
            if abs(augmented[row][col]) > tol:
                pivot = row
                break
        if pivot is None:
            return None
        if pivot != pivot_row:
            augmented[pivot], augmented[pivot_row] = augmented[pivot_row], augmented[pivot]

        pivot_val = augmented[pivot_row][col]
        augmented[pivot_row] = [value / pivot_val for value in augmented[pivot_row]]

        for row in range(rows):
            if row != pivot_row and abs(augmented[row][col]) > tol:
                factor = augmented[row][col]
                augmented[row] = [
                    value - factor * augmented[pivot_row][idx]
                    for idx, value in enumerate(augmented[row])
                ]

        pivot_cols.append(col)
        pivot_row += 1
        if pivot_row == cols:
            break

    for row in range(pivot_row, rows):
        if abs(augmented[row][-1]) > tol:
            return None

    solution = [0.0] * cols
    for idx, col in enumerate(pivot_cols):
        solution[col] = augmented[idx][-1]

    return solution


def _bound_assignments(
    indices: Sequence[int],
    lower_bounds: Sequence[float],
    upper_bounds: Sequence[float],
    tol: float = _TOL,
):
    """Yield assignments for non-basic variables to either bound."""

    if not indices:
        yield {}
        return

    options = []
    for idx in indices:
        lower = lower_bounds[idx]
        upper = upper_bounds[idx]
        if abs(lower - upper) <= tol:
            options.append((lower,))
        else:
            options.append((lower, upper))

    for values in product(*options):
        yield dict(zip(indices, values))


def _solve_dispatch_problem(
    matrix: List[List[float]],
    rhs: List[float],
    lower_bounds: List[float],
    upper_bounds: List[float],
    costs: List[float],
    tol: float = _TOL,
) -> Tuple[List[float], float]:
    """Solve the linear program using enumeration of basic feasible solutions."""

    num_vars = len(costs)
    if num_vars == 0:
        if any(abs(value) > tol for value in rhs):
            raise RuntimeError('No decision variables available to satisfy the load balance.')
        return [], 0.0

    rank = _matrix_rank(matrix, tol=tol)
    if rank == 0:
        raise RuntimeError('Dispatch problem must include at least one balance constraint.')

    indices = list(range(num_vars))
    best_solution: List[float] | None = None
    best_objective: float | None = None

    for basis_indices in combinations(indices, rank):
        basis_set = set(basis_indices)
        submatrix = [[matrix[row][col] for col in basis_indices] for row in range(len(matrix))]
        if _matrix_rank(submatrix, tol=tol) < rank:
            continue

        non_basis = [idx for idx in indices if idx not in basis_set]
        for assignment in _bound_assignments(non_basis, lower_bounds, upper_bounds, tol):
            rhs_adjusted = []
            for row in range(len(matrix)):
                total = sum(matrix[row][idx] * value for idx, value in assignment.items())
                rhs_adjusted.append(rhs[row] - total)

            solution_segment = _solve_linear_system(submatrix, rhs_adjusted, tol)
            if solution_segment is None:
                continue

            candidate = [0.0] * num_vars
            feasible = True
            for idx, value in assignment.items():
                lower = lower_bounds[idx]
                upper = upper_bounds[idx]
                if value < lower - tol or value > upper + tol:
                    feasible = False
                    break
                candidate[idx] = min(max(value, lower), upper)

            if not feasible:
                continue

            for pos, idx in enumerate(basis_indices):
                value = solution_segment[pos]
                lower = lower_bounds[idx]
                upper = upper_bounds[idx]
                if value < lower - tol or value > upper + tol:
                    feasible = False
                    break
                candidate[idx] = min(max(value, lower), upper)

            if not feasible:
                continue

            for row in range(len(matrix)):
                total = sum(matrix[row][col] * candidate[col] for col in range(num_vars))
                if abs(total - rhs[row]) > 1e-6:
                    feasible = False
                    break

            if not feasible:
                continue

            objective = sum(costs[idx] * candidate[idx] for idx in range(num_vars))
            if best_objective is None or objective < best_objective - 1e-9:
                best_objective = objective
                best_solution = candidate

    if best_solution is None or best_objective is None:
        raise RuntimeError('Unable to find feasible dispatch solution.')

    return best_solution, best_objective


def solve(
    load_by_region: Mapping[str, float],
    generators: Sequence[GeneratorSpec],
    interfaces: Mapping[Tuple[str, str], float] | Iterable[Tuple[Tuple[str, str], float]],
    allowance_cost: float,
    region_coverage: Mapping[str, bool] | None = None,
) -> DispatchResult:
    """Solve the economic dispatch problem with transmission interfaces."""

    if not generators and not load_by_region:
        raise ValueError('At least one generator or load is required to solve the dispatch problem.')

    normalized_interfaces = _normalize_interfaces(interfaces)

    region_coverage_map = {
        str(region): bool(flag)
        for region, flag in (region_coverage.items() if region_coverage else [])
    }

    regions = set(load_by_region)
    for generator in generators:
        regions.add(generator.region)
    for region_a, region_b in normalized_interfaces:
        regions.add(region_a)
        regions.add(region_b)

    if not regions:
        raise ValueError('No regions supplied for dispatch problem.')

    region_list = sorted(regions)
    region_index = {region: idx for idx, region in enumerate(region_list)}

    matrix_columns: List[List[float]] = []
    lower_bounds: List[float] = []
    upper_bounds: List[float] = []
    costs: List[float] = []
    generator_refs: List[GeneratorSpec | None] = []
    generator_indices: List[int] = []
    flow_indices: Dict[int, Tuple[str, str]] = {}

    inferred_region_coverage: Dict[str, bool] = {}

    for generator in generators:
        if generator.region not in region_index:
            raise ValueError(f'Region {generator.region} is not defined in the load data.')
        column = [0.0] * len(region_list)
        column[region_index[generator.region]] = 1.0
        matrix_columns.append(column)
        lower_bounds.append(0.0)
        upper_bounds.append(max(0.0, generator.capacity))
        costs.append(generator.marginal_cost(allowance_cost))
        generator_refs.append(generator)
        generator_indices.append(len(matrix_columns) - 1)

        current_flag = inferred_region_coverage.get(generator.region)
        inferred_region_coverage[generator.region] = (
            generator.covered if current_flag is None else current_flag and generator.covered
        )

    for region_pair, limit in sorted(normalized_interfaces.items()):
        region_a, region_b = region_pair
        column = [0.0] * len(region_list)
        column[region_index[region_a]] = -1.0
        column[region_index[region_b]] = 1.0
        matrix_columns.append(column)
        lower_bounds.append(-limit)
        upper_bounds.append(limit)
        costs.append(0.0)
        generator_refs.append(None)
        flow_indices[len(matrix_columns) - 1] = (region_a, region_b)

    matrix = [[column[idx] for column in matrix_columns] for idx in range(len(region_list))]
    rhs = [float(load_by_region.get(region, 0.0)) for region in region_list]

    solution, objective = _solve_dispatch_problem(matrix, rhs, lower_bounds, upper_bounds, costs)

    gen_by_fuel: Dict[str, float] = {}
    emissions_tons = 0.0
    # initialize accumulators
    emissions_by_region_totals: Dict[str, float] = {region: 0.0 for region in region_list}
    generation_by_region: Dict[str, float] = {region: 0.0 for region in region_list}
    generation_by_coverage: Dict[str, float] = {"covered": 0.0, "non_covered": 0.0}

    # loop over generators
    for idx in generator_indices:
        generator = generator_refs[idx]
        assert generator is not None
        output = float(solution[idx])
        gen_by_fuel.setdefault(generator.fuel, 0.0)
        gen_by_fuel[generator.fuel] += output
        emissions_tons += generator.emission_rate * output
        emissions_by_region_totals[generator.region] += generator.emission_rate * output
        generation_by_region[generator.region] += output
        coverage_key = "covered" if generator.covered else "non_covered"
        generation_by_coverage[coverage_key] += output

    # compute coverage flags and trade flows
    region_coverage_result: Dict[str, bool] = {}
    imports_to_covered = 0.0
    exports_from_covered = 0.0
    for region in region_list:
        if region in region_coverage_map:
            covered = bool(region_coverage_map[region])
        else:
            covered = bool(inferred_region_coverage.get(region, True))
        region_coverage_result[region] = covered

        load = float(load_by_region.get(region, 0.0))
        generation = generation_by_region.get(region, 0.0)
        net_import = load - generation
        if covered:
            if net_import > _TOL:
                imports_to_covered += net_import
            elif net_import < -_TOL:
                exports_from_covered += -net_import

    # region marginal prices
    delta = 1e-4
    region_prices: Dict[str, float] = {}
    for region, row_idx in region_index.items():
        rhs_up = rhs[:]
        rhs_up[row_idx] += delta
        try:
            _, objective_up = _solve_dispatch_problem(
                matrix, rhs_up, lower_bounds, upper_bounds, costs
            )
            price = (objective_up - objective) / delta
        except RuntimeError:
            rhs_down = rhs[:]
            rhs_down[row_idx] -= delta
            _, objective_down = _solve_dispatch_problem(
                matrix, rhs_down, lower_bounds, upper_bounds, costs
            )
            price = (objective - objective_down) / delta
        region_prices[region] = price

    emissions_by_region = {
        region: float(total) for region, total in emissions_by_region_totals.items()
    }

    flows: Dict[Tuple[str, str], float] = {
        pair: float(solution[idx]) for idx, pair in flow_indices.items()
    }

    return DispatchResult(
        gen_by_fuel=gen_by_fuel,
        region_prices=region_prices,
        emissions_tons=emissions_tons,
        emissions_by_region=emissions_by_region,
        flows=flows,
        generation_by_region=generation_by_region,
        generation_by_coverage=generation_by_coverage,
        imports_to_covered=imports_to_covered,
        exports_from_covered=exports_from_covered,
        region_coverage=region_coverage_result,
    )


def solve_from_frames(
    frames: Frames | Mapping[str, pd.DataFrame],
    year: int,
    allowance_cost: float,
) -> DispatchResult:
    """Solve the dispatch problem using frame-based inputs."""

    _ensure_pandas()

    frames_obj = Frames.coerce(frames)

    load_mapping = {
        str(region): float(value)
        for region, value in frames_obj.demand_for_year(year).items()
    }

    units = frames_obj.units()
    fuels = frames_obj.fuels()
    coverage_by_fuel = {
        str(row.fuel): bool(row.covered) for row in fuels.itertuples(index=False)
    }
    coverage_by_region = frames_obj.coverage_for_year(year)

    generators: List[GeneratorSpec] = []
    for row in units.itertuples(index=False):
        region_raw = row.region
        region = (
            str(region_raw) if region_raw is not None and not pd.isna(region_raw) else 'default'
        )
        fuel = str(row.fuel)
        if coverage_by_region:
            covered = bool(coverage_by_region.get(region, True))
        else:
            covered = coverage_by_fuel.get(fuel, True)
        capacity = float(row.cap_mw) * float(row.availability) * HOURS_PER_YEAR
        variable_cost = float(row.vom_per_mwh) + float(row.hr_mmbtu_per_mwh) * float(
            row.fuel_price_per_mmbtu
        )
        emission_rate = float(row.ef_ton_per_mwh)

        generators.append(
            GeneratorSpec(
                name=str(row.unit_id),
                region=region,
                fuel=fuel,
                variable_cost=variable_cost,
                capacity=capacity,
                emission_rate=emission_rate,
                covered=bool(covered),
            )
        )

    interfaces: Dict[Tuple[str, str], float] = {}
    for row in frames_obj.transmission().itertuples(index=False):
        limit = float(row.limit_mw) * HOURS_PER_YEAR
        interfaces[(str(row.from_region), str(row.to_region))] = limit

    return solve(
        load_mapping,
        generators,
        interfaces,
        allowance_cost,
        region_coverage=coverage_by_region,
    )


__all__ = ["GeneratorSpec", "solve", "solve_from_frames"]
