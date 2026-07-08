from dataclasses import dataclass,asdict
from pathlib import Path
from datetime import datetime
import json,uuid

@dataclass
class Experiment:
    hypothesis:str
    parameters:dict
    train_win:float|None=None
    validation_win:float|None=None
    test_win:float|None=None
    status:str='PENDING'
    dataset:str=''
    version:str='v1'
    id:str=''

class ExperimentManager:
    def __init__(self,root='experiments'):
        self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True)
    def create(self,hypothesis,parameters,dataset=''):
        return Experiment(hypothesis,parameters,dataset=dataset,id=uuid.uuid4().hex[:12])
    def save(self,e):
        d=asdict(e); d['saved_at']=datetime.utcnow().isoformat(); (self.root/f'{e.id}.json').write_text(json.dumps(d,indent=2),encoding='utf-8')
    def load(self,eid):
        return json.loads((self.root/f'{eid}.json').read_text(encoding='utf-8'))
    def list(self):
        return sorted(self.root.glob('*.json'))