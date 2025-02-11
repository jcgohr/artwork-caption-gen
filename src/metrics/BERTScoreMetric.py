from bert_score import BERTScorer
import torch

class BERTScoreMetric():
    def __init__(self,model:str):
        self.device="cuda" if torch.cuda.is_available() else "cpu"
        self.scorer=BERTScorer(model_type=model,lang="en",device=self.device)   
    
    def __call__(self,candidate,reference):
        P, R, F1 = self.scorer.score(candidate, reference)
        return [p.item() for p in P],[r.item() for r in R],[f1.item() for f1 in F1]
    
if __name__=="__main__":
    scr=BERTScoreMetric("distilbert-base-uncased")
    print(scr(["test test test test","test test test test test12345"],["test test test test","test test test test test"]))