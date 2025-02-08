from bert_score import BERTScore

class BERTScoreMetric():
    def __init__(self,model:str):
        self.scorer=BERTScore(model_type=model,lang="en")   
    
    def __call__(self,candidate,reference):
        P, R, F1 = self.scorer.score(candidate, reference)
        return P,R,F1