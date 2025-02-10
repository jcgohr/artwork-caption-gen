from metrics import BleuMetric,MeteorMetric,RougeMetric,BERTScoreMetric 
from parsers import CaptionMetricParser

parser=CaptionMetricParser()
args=parser.parse_args()

scores={}
metrics=[
    # Returns p (precision)
    (BleuMetric(),["bleu_precision"]),
    # Return a METEOR scores
    (MeteorMetric(),["meteor_score"]),
    # Returns averages of averages (p, r, f1)   
    (RougeMetric(),[["rouge-1-p","rouge-1-r","rouge-1-f1"],["rouge-2-p","rouge-2-r","rouge-2-f1"],["rouge-l-p","rouge-l-r","rouge-l-f1"]]),
    # Returns p's, r's, f1's
    (BERTScoreMetric("distilbert-base-uncased"),["distilbert-base-uncased-bertscore"]),
]