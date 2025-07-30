from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

class Participant(BaseModel):
    name: str
    skills: List[str]
    experience: int
    goals: List[str]

@app.post("/match")
def match_participants(participants: List[Participant]):
    teams = [participants] 
    return {"teams": teams}
