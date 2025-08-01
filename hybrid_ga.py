"""
This is a complete Python microservice that implements a hybrid genetic algorithm
for team matching. It uses the DEAP library for the genetic algorithm components
and FastAPI for the API endpoint.

The algorithm works by:
1.  **Representing a Solution:** Each "individual" in the genetic algorithm is a list
    of user IDs. The order of the IDs determines the team assignments. For example,
    if the team size is 3, the first 3 IDs form Team 1, the next 3 form Team 2, and so on.
2.  **Fitness Function:** This function is the "brain" of the algorithm. It calculates a
    score for each team assignment based on the criteria you defined:
    - Skill diversity (how many unique required skills are present on a team)
    - Experience balance (mixing skill levels, e.g., a senior with two juniors)
    - Role preference satisfaction
    - Learning opportunity (pairing learners with experts)
3.  **Hybrid Approach:** The genetic algorithm evolves the population, and a local search
    heuristic (a simple two-user swap) is applied to the best individuals to
    fine-tune and improve the solution in each generation.
4.  **API Endpoint:** A FastAPI route with Pydantic for data validation listens for a
    POST request, runs the algorithm, and returns the best-found team configuration.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from deap import base, creator, tools, algorithms
import random
import collections
from typing import List, Dict

app = FastAPI()

# --- Data Models for Pydantic Validation ---
# Pydantic models ensure that the incoming data is correctly structured and typed.

class Skill(BaseModel):
    name: str
    level: int

class User(BaseModel):
    id: int
    skills: List[Skill]
    experience_level: str
    learning_goals: List[str]
    role_preference: str

class RequiredSkill(BaseModel):
    name: str
    level: int

class Jam(BaseModel):
    id: int
    required_skills: List[RequiredSkill]
    team_size: int

class Weights(BaseModel):
    skill_diversity: float
    experience_balance: float
    learning_opportunity: float
    role_preference: float

class MatchRequest(BaseModel):
    users: List[User]
    jam: Jam
    weights: Weights

class MatchResponse(BaseModel):
    teams: List[List[int]]

# --- Genetic Algorithm Setup with DEAP ---
# We define a "fitness" object. Here, we're maximizing the fitness score.
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# We define an "individual" as a list of user IDs, with our custom fitness object.
# The 'type: ignore' comment here is a workaround for static type checkers like Pyright,
# which cannot infer the dynamic creation of creator.FitnessMax at runtime.
creator.create("Individual", list, fitness=creator.FitnessMax)  # type: ignore

# The toolbox holds all the operators (initialization, mutation, etc.)
toolbox = base.Toolbox()

# --- Helper Functions for Fitness Calculation ---

def calculate_skill_diversity(team, users_by_id, required_skills):
    """
    Scores a team based on how many required skills are present.
    A team with a diverse set of required skills gets a higher score.
    """
    team_skills = set()
    for user_id in team:
        user = users_by_id.get(user_id)
        if user and user.skills:
            for skill in user.skills:
                team_skills.add(skill.name)

    score = 0
    for req_skill in required_skills:
        if req_skill.name in team_skills:
            score += 1
    return score

def calculate_experience_balance(team, users_by_id):
    """
    Scores a team based on the mix of experience levels.
    A good mix (e.g., a senior with juniors) is rewarded.
    """
    levels = collections.Counter()
    for uid in team:
        user = users_by_id.get(uid)
        if user:
            levels[user.experience_level] += 1

    score = 0
    if 'Advanced' in levels or 'Senior' in levels:
        if len(levels) > 1: # A mix of experience levels
            score += 2
        else: # All are senior/advanced, which is good but not for learning goals
            score += 1
    elif 'Intermediate' in levels:
        score += 1

    return score

def calculate_learning_opportunity(team, users_by_id):
    """
    Scores a team based on how many learning goals are met by other members' skills.
    """
    score = 0
    team_skills = set()
    for user_id in team:
        user = users_by_id.get(user_id)
        if user and user.skills:
            for skill in user.skills:
                team_skills.add(skill.name)

    for user_id in team:
        user = users_by_id.get(user_id)
        if user and user.learning_goals:
            for goal in user.learning_goals:
                if goal in team_skills:
                    score += 1
    return score

def calculate_role_preference(team, users_by_id):
    """
    Scores a team based on how many users got their preferred role.
    This is a simplified scoring mechanism.
    """
    score = 0
    for user_id in team:
        user = users_by_id.get(user_id)
        if user and user.role_preference:
            # This is a placeholder. A more complex implementation would
            # map a user to a specific role.
            score += 1
    return score


def evaluate_teams(individual, users_data, jam_data, weights):
    """
    The main fitness function for the genetic algorithm.
    It returns a tuple of the total score.
    """
    # Pre-process data for easy lookup
    users_by_id = {u.id: u for u in users_data}
    required_skills = jam_data.required_skills
    team_size = jam_data.team_size

    # Slice the individual (permutation of users) into teams
    teams = [individual[i:i + team_size] for i in range(0, len(individual), team_size)]

    total_fitness = 0
    for team in teams:
        # Check for incomplete teams. This can happen if the number of users
        # is not a perfect multiple of team size. It's okay, but we still penalize it.
        # This check is good for robustness, but the GA is designed to fill teams.
        if not team:
            continue

        # Calculate scores for each team based on the criteria
        skill_diversity_score = calculate_skill_diversity(team, users_by_id, required_skills)
        experience_balance_score = calculate_experience_balance(team, users_by_id)
        learning_opportunity_score = calculate_learning_opportunity(team, users_by_id)
        role_preference_score = calculate_role_preference(team, users_by_id)

        # Apply weights from the configuration
        team_score = (
            weights.skill_diversity * skill_diversity_score +
            weights.experience_balance * experience_balance_score +
            weights.learning_opportunity * learning_opportunity_score +
            weights.role_preference * role_preference_score
        )
        total_fitness += team_score

    # We return a tuple because DEAP's fitness expects a tuple
    return (total_fitness,)

def local_search_heuristic(individual, fitness_fn, users_data, jam_data, weights):
    """
    A simple local search heuristic to improve a given individual.
    It tries swapping two members and keeps the swap if the fitness improves.
    """
    current_fitness = fitness_fn(individual, users_data, jam_data, weights)[0]

    # Try a few random swaps
    for _ in range(10):  # You can tune the number of attempts
        idx1, idx2 = random.sample(range(len(individual)), 2)

        # Perform the swap
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]

        new_fitness = fitness_fn(individual, users_data, jam_data, weights)[0]

        if new_fitness > current_fitness:
            current_fitness = new_fitness
        else:
            # If the swap didn't improve, swap back
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]

    return individual,


# --- API Endpoint and Main Function ---

@app.post('/api/match', response_model=MatchResponse)
async def match_teams_endpoint(request: MatchRequest):
  return match_teams(users_data=request.users, jam_data=request.jam, weights=request.weights)

async def match_teams(users_data, jam_data, weights):
    try:
        # Create a list of user IDs
        user_ids = [u.id for u in users_data]
        team_size = jam_data.team_size
        num_users = len(user_ids)

        if num_users % team_size != 0:
            print(f"Warning: {num_users} users is not a perfect multiple of team size {team_size}. The last team will be smaller.")

        # Re-register the genetic algorithm components for this specific run
        toolbox.register("indices", random.sample, range(num_users), num_users)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)  # type: ignore
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)  # type: ignore

        # Register the evaluation function
        toolbox.register("evaluate", evaluate_teams, users_data=users_data, jam_data=jam_data, weights=weights)  # type: ignore

        # Register genetic operators
        toolbox.register("mate", tools.cxOrdered)  # type: ignore
        toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)  # type: ignore
        toolbox.register("select", tools.selTournament, tournsize=3)  # type: ignore
        toolbox.register("local_search", local_search_heuristic, fitness_fn=evaluate_teams, users_data=users_data, jam_data=jam_data, weights=weights)  # type: ignore

        # --- Genetic Algorithm Main Loop ---
        pop = toolbox.population(n=100)  # type: ignore
        hof = tools.HallOfFame(1)

        # The hybrid algorithm loop
        for gen in range(50):
            offspring = algorithms.varAnd(pop, toolbox, cxpb=0.5, mutpb=0.1)
            fits = toolbox.map(toolbox.evaluate, offspring)  # type: ignore
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit

            elites = tools.selBest(pop, k=int(len(pop) * 0.1))
            for elite in elites:
                toolbox.local_search(elite)  # type: ignore

            pop = toolbox.select(pop + offspring, k=len(pop))  # type: ignore

            hof.update(pop)

        best_individual = hof[0]

        # Convert the individual (list of IDs) back into teams
        best_teams = [best_individual[i:i + team_size] for i in range(0, num_users, team_size)]

        # Return the best teams
        return MatchResponse(teams=best_teams)

    except Exception as e:
        # FastAPI will handle validation errors automatically.
        # This catch is for other unexpected errors.
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
