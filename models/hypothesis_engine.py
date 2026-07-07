from dataclasses import dataclass
from itertools import combinations

@dataclass(frozen=True)
class Condition:
    feature:str
    operator:str
    value:object

@dataclass
class Hypothesis:
    id:str
    direction:str
    conditions:list

class HypothesisEngine:
    def generate(self, feature_rules:dict, max_features:int=3):
        idx=1
        keys=list(feature_rules.keys())
        for r in range(1,max_features+1):
            for combo in combinations(keys,r):
                cond=[Condition(f,*feature_rules[f]) for f in combo]
                yield Hypothesis(f'H{idx:06d}','AUTO',cond)
                idx+=1
