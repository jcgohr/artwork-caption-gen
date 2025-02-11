from torcheval.metrics import BLEUScore
import torch

class BleuMetric():
    def __init__(self):
        self.bleu=BLEUScore(n_gram=4)

    def __call__(self,candidate,reference):
        if type(candidate)==str and type(reference)==str:
            self.bleu.update(candidate, reference)
            return self.bleu.compute()
        bleus=[]
        for cand,ref in zip(candidate,reference):
            self.bleu.reset()
            self.bleu.update([cand],[ref])
            bleus.append(self.bleu.compute().item())
        return bleus
    
if __name__=="__main__":
    bleu=BleuMetric()
    print(bleu(["test test test test2","test test test zzee test3 zzee","test test test test4"],["test test test test2","test test test zzee test55 zzee","test test test test test44"]))