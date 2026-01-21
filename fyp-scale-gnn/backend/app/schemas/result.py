from pydantic import BaseModel
from typing import List

class RankedTest(BaseModel):
    test_id: str
    score: float

class RankingResponse(BaseModel):
    experiment_id: int
    results: List[RankedTest]
