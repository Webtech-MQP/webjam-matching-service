from fastapi import FastAPI
from ilp import Skill, User, solve
from pydantic import BaseModel

app = FastAPI()


class MatchRequest(BaseModel):
    num_teams: int
    users: list[User]
    skills: list[Skill]


@app.post("/match")
def match_teams(request: MatchRequest):
    solution = solve(request.users, request.num_teams)
    return solution
