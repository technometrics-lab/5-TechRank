import dataclasses
from typing import Dict


@dataclasses.dataclass
class Company:
    id: str
    name: str
    location: Dict[str, str]
    degree: float =  0
    rank_CB: float = 0
    rank_algo: float =  0


@dataclasses.dataclass
class Investor:
    id: str
    name: str

@dataclasses.dataclass
class Technology:
    name: str
    # score: float = 0
    degree: float =  0
    rank_algo: float =  0

    def update_score(self, a: int):
        self.score = a
