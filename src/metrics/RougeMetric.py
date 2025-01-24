from rouge import Rouge 

class RougeMetric():
    def __init__(self):
        self.rouge=Rouge()
        
    def __call__(self,candidate,reference):
        return self.rouge.get_scores(candidate,reference)