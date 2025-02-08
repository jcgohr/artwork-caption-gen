from bert_score import BERTScorer

class BERTScoreMetric():
    def __init__(self,model:str):
        self.scorer=BERTScorer(model_type=model,lang="en")   
    
    def __call__(self,candidate,reference):
        P, R, F1 = self.scorer.score(candidate, reference)
        return P,R,F1