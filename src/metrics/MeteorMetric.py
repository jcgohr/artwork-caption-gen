from evaluate import load

class MeteorMetric():
    def __init__(self):
        self.meteor = load('meteor')
        
    def __call__(self, candidate, reference):
        if isinstance(candidate, str) and isinstance(reference, str):
            # Single prediction/reference case
            result = self.meteor.compute(predictions=[candidate], references=[reference])
            return float(result['meteor'])
        
        # Multiple predictions/references case - compute individual scores
        return [float(self.meteor.compute(predictions=[cand], references=[ref])['meteor']) 
                for cand, ref in zip(candidate, reference)]

if __name__ == "__main__":
    mtr = MeteorMetric()
    # Test single string case
    print(mtr("test test test", "test test test"))
    # Test list case
    print(mtr(["test test test", "test test test"], 
              ["test test test", "test test test"]))