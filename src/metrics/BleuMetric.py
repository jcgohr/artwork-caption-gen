from bleuscore import evaluate

def BleuMetric():    
    def __call__(self,candidate,reference):
        results = evaluate(predictions=candidate, references=reference)
        return results