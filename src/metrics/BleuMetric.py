from bleuscore import compute

def BleuMetric():    
    def __call__(self,candidate,reference):
        results = compute(predictions=candidate, references=reference)
        return results