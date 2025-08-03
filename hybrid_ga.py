"""
This is a complete Python microservice that implements a hybrid genetic algorithm
for team matching. It uses the DEAP library for the genetic algorithm components
and FastAPI for the API endpoint.

The algorithm works by:
1.  **Representing a Solution:** Each "individual" in the genetic algorithm is a list
    of user indices. These indices are mapped to string user IDs. The order of the indices determines the team assignments.
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

from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from deap import base, creator, tools, algorithms
import random
import collections
from typing import List, Dict
import logging
import datetime
import json

app = FastAPI()

logger = logging.getLogger(__name__)

class Skill(BaseModel):
    name: str
    level: int

class User(BaseModel):
    id: str
    skills: List[Skill]
    experience_level: str
    learning_goals: List[str]
    role_preference: str

class Jam(BaseModel):
    id: str
    required_skills: List[str]
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
    teams: List[List[str]]

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)  # type: ignore

toolbox = base.Toolbox()

def calculate_skill_diversity(team, users_by_id, required_skills: List[str]):
    team_skills = set()
    for user_id in team:
        user = users_by_id.get(user_id)
        if user and user.skills:
            for skill in user.skills:
                team_skills.add(skill.name)

    score = 0
    for req_skill in required_skills:
        if req_skill in team_skills:
            score += 1
    return score

def calculate_experience_balance(team, users_by_id):
    levels = collections.Counter()
    for uid in team:
        user = users_by_id.get(uid)
        if user:
            levels[user.experience_level] += 1

    score = 0
    if 'Advanced' in levels or 'Senior' in levels:
        if len(levels) > 1:
            score += 2
        else:
            score += 1
    elif 'Intermediate' in levels:
        score += 1

    return score

def calculate_learning_opportunity(team, users_by_id):
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
    score = 0
    for user_id in team:
        user = users_by_id.get(user_id)
        if user and user.role_preference:
            score += 1
    return score

def evaluate_teams(individual, users_data, jam_data, weights):
    users_by_id = {u.id: u for u in users_data}
    id_lookup = {i: u.id for i, u in enumerate(users_data)}
    required_skills = jam_data.required_skills
    team_size = jam_data.team_size

    ids = [id_lookup[i] for i in individual]
    teams = [ids[i:i + team_size] for i in range(0, len(ids), team_size)]

    total_fitness = 0
    for team in teams:
        if not team:
            continue

        skill_diversity_score = calculate_skill_diversity(team, users_by_id, required_skills)
        experience_balance_score = calculate_experience_balance(team, users_by_id)
        learning_opportunity_score = calculate_learning_opportunity(team, users_by_id)
        role_preference_score = calculate_role_preference(team, users_by_id)

        team_score = (
            weights.skill_diversity * skill_diversity_score +
            weights.experience_balance * experience_balance_score +
            weights.learning_opportunity * learning_opportunity_score +
            weights.role_preference * role_preference_score
        )
        total_fitness += team_score

    return (total_fitness,)

def local_search_heuristic(individual, fitness_fn, users_data, jam_data, weights):
    current_fitness = fitness_fn(individual, users_data, jam_data, weights)[0]

    for _ in range(10):
        idx1, idx2 = random.sample(range(len(individual)), 2)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        new_fitness = fitness_fn(individual, users_data, jam_data, weights)[0]

        if new_fitness > current_fitness:
            current_fitness = new_fitness
        else:
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]

    return individual,

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    error_details = {
        "method": request.method,
        "url": str(request.url),
        "client_ip": request.client.host if request.client else "unknown",
        "user_agent": request.headers.get("user-agent", "unknown"),
        "content_type": request.headers.get("content-type", "unknown"),
        "validation_errors": exc.errors(),
        "request_body": exc.body
    }

    logger.error(f"Validation Error: {json.dumps(error_details, indent=2)}")

    return JSONResponse(
        status_code=422,
        content={"detail": "Validation error", "errors": exc.errors()}
    )

@app.post('/api/match', response_model=MatchResponse)
async def match_teams_endpoint(request: MatchRequest):
    return await match_teams(users_data=request.users, jam_data=request.jam, weights=request.weights)

async def match_teams(users_data, jam_data, weights):
    logging.basicConfig(level=logging.INFO)
    try:
        user_ids = [u.id for u in users_data]
        num_users = len(user_ids)
        logger.info(f"Number of users: {num_users}")
        team_size = jam_data.team_size

        if num_users % team_size != 0:
            print(f"Warning: {num_users} users is not a perfect multiple of team size {team_size}. The last team will be smaller.")

        id_lookup = {i: uid for i, uid in enumerate(user_ids)}

        if (num_users == team_size):
            logger.warning("Only one team available")
            return MatchResponse(teams=[[user_ids[0]]])

        toolbox.register("indices", random.sample, range(num_users), num_users)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)  # type: ignore
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)  # type: ignore
        toolbox.register("evaluate", evaluate_teams, users_data=users_data, jam_data=jam_data, weights=weights)  # type: ignore
        toolbox.register("mate", tools.cxOrdered)  # type: ignore
        toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)  # type: ignore
        toolbox.register("select", tools.selTournament, tournsize=3)  # type: ignore
        toolbox.register("local_search", local_search_heuristic, fitness_fn=evaluate_teams, users_data=users_data, jam_data=jam_data, weights=weights)  # type: ignore

        pop = toolbox.population(n=1000)  # type: ignore
        hof = tools.HallOfFame(1)

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
        best_teams = [[id_lookup[i] for i in best_individual[j:j + team_size]] for j in range(0, num_users, team_size)]

        return MatchResponse(teams=best_teams)

    except Exception as e:
        logger.error(f"Error in match_teams: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True, use_colors=True)
