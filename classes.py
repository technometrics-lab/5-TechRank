import dataclasses


@dataclasses.dataclass
class Company:
    id: str
    name: str
    centrality: float =  0
    rank_algo: float =  0
    location_country: str
    location_city: str


@dataclasses.dataclass
class Investor:
    id: str
    name: str

@dataclasses.dataclass
class Technology:
    id: str
    name: str
    score: float = 0
    centrality: float =  0
    rank_algo: float =  0

    def update_score(self, a: int):
        self.score = a
