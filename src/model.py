import numpy as np
from scipy.optimize import linprog

def solve_transport(times, supply, demand, arc_capacity, closed_arcs=None):
    """Minimize total team-minutes; satisfy every incident demand."""
    times = np.asarray(times, dtype=float)
    supply = np.asarray(supply, dtype=float)
    demand = np.asarray(demand, dtype=float)
    cap = np.asarray(arc_capacity, dtype=float).copy()
    nb, ni = times.shape
    if closed_arcs:
        for i, j in closed_arcs:
            cap[i, j] = 0.0

    # x[i,j] = number of teams sent from base i to incident point j.
    c = times.reshape(-1)
    A_ub, b_ub = [], []
    for i in range(nb):
        row = np.zeros(nb * ni)
        row[i * ni:(i + 1) * ni] = 1
        A_ub.append(row)
        b_ub.append(supply[i])

    A_eq, b_eq = [], []
    for j in range(ni):
        row = np.zeros(nb * ni)
        row[j::ni] = 1
        A_eq.append(row)
        b_eq.append(demand[j])

    bounds = [(0, float(cap[i, j])) for i in range(nb) for j in range(ni)]
    result = linprog(
        c, A_ub=np.asarray(A_ub), b_ub=np.asarray(b_ub),
        A_eq=np.asarray(A_eq), b_eq=np.asarray(b_eq),
        bounds=bounds, method="highs-ds", options={"presolve": True}
    )
    if result.x is not None:
        result.x = result.x.reshape(nb, ni)
    return result, cap

def greedy_transport(times, supply, demand, arc_capacity):
    """Simple baseline: allocate in ascending travel-time order."""
    remaining_supply = np.asarray(supply, dtype=float).copy()
    remaining_demand = np.asarray(demand, dtype=float).copy()
    remaining_cap = np.asarray(arc_capacity, dtype=float).copy()
    x = np.zeros_like(times, dtype=float)
    arcs = sorted(
        ((float(times[i, j]), i, j) for i in range(len(supply)) for j in range(len(demand))),
        key=lambda item: item[0]
    )
    for _, i, j in arcs:
        amount = min(remaining_supply[i], remaining_demand[j], remaining_cap[i, j])
        if amount > 0:
            x[i, j] += amount
            remaining_supply[i] -= amount
            remaining_demand[j] -= amount
            remaining_cap[i, j] -= amount
    feasible = np.allclose(remaining_demand, 0, atol=1e-9)
    return x, feasible, float(np.sum(x * times))
