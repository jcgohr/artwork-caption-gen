from metrics import BleuMetric,MeteorMetric,RougeMetric,BERTScoreMetric 
from parsers import CaptionMetricParser

parser=CaptionMetricParser()
args=parser.parse_args()

metrics=[
    BleuMetric(),
    MeteorMetric(),
    RougeMetric(),
    BERTScoreMetric("distilbert-base-uncased")
]