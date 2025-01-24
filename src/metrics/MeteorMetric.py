from pymeteor.pymeteor import meteor as pymeteor

class MeteorMetric():
    def __init__(self):
        pass
    
    def __call__(reference,candidate):
        return pymeteor(reference,candidate)

