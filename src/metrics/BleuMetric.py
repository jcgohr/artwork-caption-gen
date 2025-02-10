from bleuscore import compute

class BleuMetric():    
    def __call__(self,candidate,reference):
        results = compute(predictions=candidate, references=reference)
        return results
    
if __name__=="__main__":
    bleu=BleuMetric()
    bleu(["test test test","test test zee"],["test test test","test test test"])