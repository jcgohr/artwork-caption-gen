from pymeteor.pymeteor import meteor as pymeteor

class MeteorMetric():
    def __call__(self,candidate,reference):
        if type(candidate)==str and type(reference)==str:
            return pymeteor(reference,candidate)
        return [pymeteor(ref,cand) for ref,cand in zip(reference,candidate)]
    
if __name__=="__main__":
    mtr=MeteorMetric()
    print(mtr("test test test","test test test"))
    print(mtr(["test test test","test test test"],["test test test","test test test"]))