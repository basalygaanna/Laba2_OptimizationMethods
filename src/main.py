import sys
from pathlib import Path
import numpy as np
import pandas as pd

from data import load_data, ROOT
from model import solve_transport, greedy_transport
from validation import validate_solution
from reporting import save_solution, save_network, write_json

def status_text(status):
    return {
        0: "optimal — найдено оптимальное решение",
        1: "iteration/time limit — достигнут лимит",
        2: "infeasible — задача несовместна",
        3: "unbounded — целевая функция не ограничена",
        4: "solver error — ошибка решателя"
    }.get(status, f"неизвестный статус: {status}")

def run():
    cfg, d = load_data()
    bases, points = cfg["bases"], cfg["incident_points"]
    times = np.asarray(d["travel_time_minutes"], dtype=float)
    supply = np.asarray(d["base_supply_teams"], dtype=float)
    demand = np.asarray(d["incident_demand_teams"], dtype=float)
    capacities = np.asarray(d["arc_capacity_teams"], dtype=float)

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)

    # Базовый сценарий
    result, effective_cap = solve_transport(times, supply, demand, capacities)
    with (ROOT / "logs" / "solver.log").open("w", encoding="utf-8") as log:
        log.write("ЛР №2, вариант 18 — базовый сценарий\n")
        log.write(f"Статус HiGHS: {status_text(result.status)}\n")
        log.write(f"Сообщение: {result.message}\n")
        log.write(f"Целевая функция: {result.fun}\n")
        log.write("Алгоритм: dual simplex (HiGHS-DS)\n")
        log.write("Presolve: включён\n")
        log.write(f"Итерации: {getattr(result, 'nit', 'нет данных')}\n")

    if not result.success:
        print(status_text(result.status))
        print(result.message)
        return

    x = result.x
    validation = validate_solution(x, times, supply, demand, capacities)
    save_solution(x, bases, points, times, capacities, "base_solution.csv")
    save_network(x, bases, points, times, capacities, "network.png")
    write_json(ROOT / "results" / "validation.json", validation)

    # Сценарий отказа наиболее загруженной дуги
    i, j = np.unravel_index(np.argmax(x), x.shape)
    failed_result, failed_cap = solve_transport(
        times, supply, demand, capacities, closed_arcs=[(i, j)]
    )
    failure = {
        "closed_arc": [bases[i], points[j]],
        "original_flow_teams": float(x[i, j]),
        "status": status_text(failed_result.status),
        "objective_team_minutes": float(failed_result.fun) if failed_result.success else None,
        "change_in_objective": float(failed_result.fun - result.fun) if failed_result.success else None,
        "message": failed_result.message
    }
    if failed_result.success:
        save_solution(failed_result.x, bases, points, times, failed_cap, "failure_solution.csv")
        save_network(failed_result.x, bases, points, times, failed_cap, "failure_network.png")
        failure["validation"] = validate_solution(
            failed_result.x, times, supply, demand, failed_cap
        )
    write_json(ROOT / "results" / "scenario_arc_failure.json", failure)

    # Сценарий роста спроса на 20%
    increased_demand = demand * 1.2
    growth_result, growth_cap = solve_transport(times, supply, increased_demand, capacities)
    growth = {
        "demand_multiplier": 1.2,
        "new_demand_teams": increased_demand.tolist(),
        "status": status_text(growth_result.status),
        "objective_team_minutes": float(growth_result.fun) if growth_result.success else None,
        "change_in_objective": float(growth_result.fun - result.fun) if growth_result.success else None,
        "message": growth_result.message
    }
    if growth_result.success:
        save_solution(growth_result.x, bases, points, times, growth_cap, "growth_solution.csv")
        growth["validation"] = validate_solution(
            growth_result.x, times, supply, increased_demand, growth_cap
        )
    write_json(ROOT / "results" / "scenario_demand_growth.json", growth)

    # Узкие места: дуги, полностью использующие пропускную способность.
    saturated = []
    for a in range(len(bases)):
        for b in range(len(points)):
            if abs(x[a, b] - capacities[a, b]) <= 1e-7 and x[a, b] > 0:
                saturated.append({
                    "base": bases[a], "incident_point": points[b],
                    "flow_teams": float(x[a, b]),
                    "capacity_teams": float(capacities[a, b])
                })
    write_json(ROOT / "results" / "bottlenecks.json", saturated)

    # Простая жадная эвристика для сравнения
    gx, gfeasible, gobj = greedy_transport(times, supply, demand, capacities)
    greedy = {
        "feasible": bool(gfeasible),
        "objective_team_minutes": gobj if gfeasible else None,
        "optimal_objective_team_minutes": float(result.fun),
        "gap_percent": ((gobj / result.fun - 1) * 100) if gfeasible and result.fun != 0 else None
    }
    if gfeasible:
        save_solution(gx, bases, points, times, capacities, "greedy_solution.csv")
    write_json(ROOT / "results" / "greedy_comparison.json", greedy)

    summary = pd.DataFrame([
        {"scenario": "Базовый", "status": status_text(result.status),
         "objective_team_minutes": float(result.fun)},
        {"scenario": "Отказ наиболее загруженной дуги", "status": failure["status"],
         "objective_team_minutes": failure["objective_team_minutes"]},
        {"scenario": "Рост спроса на 20%", "status": growth["status"],
         "objective_team_minutes": growth["objective_team_minutes"]}
    ])
    summary.to_csv(ROOT / "results" / "scenarios.csv", index=False, encoding="utf-8-sig")

    print("ЛР №2, вариант 18")
    print(status_text(result.status))
    print(f"Минимальное суммарное время: {result.fun:.2f} бригадо-минут")
    print(f"Независимая проверка: {'пройдена' if validation['valid'] else 'НЕ ПРОЙДЕНА'}")
    print(f"Отказ дуги: {failure['status']}")
    print(f"Рост спроса на 20%: {growth['status']}")
    print("Результаты сохранены в папках results/ и logs/.")

if __name__ == "__main__":
    run()
