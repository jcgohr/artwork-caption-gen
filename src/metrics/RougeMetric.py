from rouge import Rouge 

class RougeMetric():
    def __init__(self):
        self.rouge=Rouge()
        
    def __call__(self,candidate,reference):
        return [list([list(dict(r).values()) for r in rouge_dict.values()]) for rouge_dict in self.rouge.get_scores(candidate,reference)]
    
if __name__=="__main__":
    rge=RougeMetric()
    # print(rge("test test test","test test test"))
    print(rge(["test test test","test test zee"],["test test test","test test test"]))
    