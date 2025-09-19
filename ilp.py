from typing import override
from typing_extensions import Hashable
from ortools.linear_solver import pywraplp
from dataclasses import dataclass
from itertools import combinations, chain


@dataclass
class Skill:
    id: Hashable
    threshold: int


@dataclass
class User:
    id_number: int
    skill_levels: dict[Hashable, int]

    @override
    def __hash__(self):
        return self.id_number


def f_k(team: list[User], skill: Skill) -> int:
    """
    Returns the team evaluation heuristic for a skill.
    """
    return sum(user.skill_levels.get(skill.id, 0) > skill.threshold for user in team)


def f(team: list[User], skills: list[Skill]):
    """
    Returns a teams evaluation heuristic across all skills.
    """
    return sum(f_k(team, skill) for skill in skills) / len(skills)


def solve(users: list[User], skills: list[Skill], num_teams: int):
    solver = pywraplp.Solver.CreateSolver("SAT")
    if not solver:
        return

    react = Skill("react", 5)
    python = Skill("python", 3)

    # Example data - replace with your actual data
    users = [
        User(1, {"react": 8, "python": 6}),
        User(2, {"react": 4, "python": 7}),
        User(3, {"react": 9, "python": 2}),
        User(4, {"react": 6, "python": 8}),
        User(5, {"react": 1, "python": 5}),
        User(6, {"react": 3, "python": 4}),
        User(7, {"react": 2, "python": 8}),
        User(8, {"react": 8, "python": 7}),
        User(9, {"react": 7, "python": 9}),
        User(10, {"react": 3, "python": 2}),
    ]
    skills = [react, python]

    # TODO: This might not work for team #teams that divide weird.
    base = len(users) // num_teams
    min_team_size = max(1, base - 1)
    max_team_size = base + 2

    feasible_teams = list(
        chain.from_iterable(
            combinations(users, r) for r in range(min_team_size, max_team_size + 1)
        )
    )
    num_feasible_teams = len(feasible_teams)

    x = {}
    for u in range(num_feasible_teams):
        x[u] = solver.IntVar(0, 1, f"x[{u}]")

    print("Number of variables =", solver.NumVariables())

    # Constraint: Each user is assigned to exactly one CHOSEN team
    for user in users:
        solver.Add(sum(x[i] for i, t in enumerate(feasible_teams) if user in t) == 1)

    print("Number of constraints =", solver.NumConstraints())

    # Objective: Maximize total team performance across all teams
    objective = sum(f(list(t), skills) * x[i] for i, t in enumerate(feasible_teams))

    solver.Maximize(objective)

    print(f"Solving with {solver.SolverVersion()}")
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        print("Solution:")
        print("Objective value =", solver.Objective().Value())

        # Print team assignments
        teams = []
        for i, t in enumerate(feasible_teams):
            if x[i].solution_value() > 0.5:  # Binary variable is 1
                teams.append(t)

        return teams
    else:
        print("The problem does not have an optimal solution.")

    print("\nAdvanced usage:")
    print(f"Problem solved in {solver.wall_time():d} milliseconds")
    print(f"Problem solved in {solver.iterations():d} iterations")
    print(f"Problem solved in {solver.nodes():d} branch-and-bound nodes")
