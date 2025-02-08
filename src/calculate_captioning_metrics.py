from metrics import BleuMetric,MeteorMetric,RougeMetric,BERTScoreMetric 
from parsers import CaptionMetricParser

parser=CaptionMetricParser()
args=parser.parse_args()

metrics=[
    # Returns p (precision)
    BleuMetric(),
    # Return a METEOR scores
    MeteorMetric(),
    # Returns averages of averages (p, r, f1)   
    RougeMetric(),
    # Returns r's,r's,f1's
    BERTScoreMetric("distilbert-base-uncased")
]