from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import json
import uuid

@dataclass
class Experiment:
    hypothesis: str
    parameters: dict
    train_win: float|None=None
    validation_win: float|None=None
    test_win: float|None=None
    status: str='PENDING'
    id: str=''

class ExperimentManager:
    def __init__(self, root='experiments'):
        self.root=Path(root)
        self.root.mkdir(exist_ok=True)
    def create(self,hypothesis,parameters):
        e=Experiment(hypothesis,parameters,id=uuid.uuid4().hex[:12])
        return e
    def save(self,e:Experiment):
        data=asdict(e)
        data['saved_at']=datetime.utcnow().isoformat()
        (self.root/f'{e.id}.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
