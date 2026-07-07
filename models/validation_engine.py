from dataclasses import dataclass

@dataclass
class ValidationResult:
    passed: bool
    reason: str

class ValidationEngine:
    def __init__(self,min_winrate=0.53,min_occurrence=1000):
        self.min_winrate=min_winrate
        self.min_occurrence=min_occurrence
    def evaluate(self,winrate:float,occurrence:int)->ValidationResult:
        if occurrence<self.min_occurrence:
            return ValidationResult(False,'insufficient_occurrence')
        if winrate<self.min_winrate:
            return ValidationResult(False,'winrate_below_threshold')
        return ValidationResult(True,'accepted')
