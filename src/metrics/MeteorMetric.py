from pymeteor.pymeteor import meteor as pymeteor

class MeteorMetric():
    def __init__(self):
        pass
    
    def __call__(self,candidate,reference):
        return pymeteor(reference,candidate)

