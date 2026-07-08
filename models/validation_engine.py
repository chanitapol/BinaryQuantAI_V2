from dataclasses import dataclass
from math import sqrt

@dataclass
class ValidationResult:
    passed: bool
    reason: str
    confidence: float
    winrate: float
    occurrence: int

class ValidationEngine:
    def __init__(self,min_winrate=0.53,min_occurrence=1000):
        self.min_winrate=min_winrate
        self.min_occurrence=min_occurrence
    def confidence(self,winrate,occurrence):
        if occurrence<=0:return 0.0
        return max(0.0,min(1.0,(winrate-0.5)*sqrt(occurrence)*2))
    def evaluate(self,winrate:float,occurrence:int):
        c=self.confidence(winrate,occurrence)
        if occurrence<self.min_occurrence:
            return ValidationResult(False,'insufficient_occurrence',c,winrate,occurrence)
        if winrate<self.min_winrate:
            return ValidationResult(False,'winrate_below_threshold',c,winrate,occurrence)
        return ValidationResult(True,'accepted',c,winrate,occurrence)