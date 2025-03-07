from metrics import BleuMetric,MeteorMetric,RougeMetric,BERTScoreMetric 
from parsers import CaptionMetricParser
from utils import parse_metric_results,write_scores_to_tsv

parser=CaptionMetricParser()
parser.parse_args()


metrics=[
    # Returns averages of averages (p, r, f1)   
    (RougeMetric(),[["rouge-1-p","rouge-1-r","rouge-1-f1"],["rouge-2-p","rouge-2-r","rouge-2-f1"],["rouge-l-p","rouge-l-r","rouge-l-f1"]]),
    # Returns p's, r's, f1's
    (BERTScoreMetric("distilbert-base-uncased"),["distilbert-base-uncased-bertscore-p","distilbert-base-uncased-bertscore-r","distilbert-base-uncased-bertscore-f1"]),
    # Returns p (precision)
    (BleuMetric(),["bleu_precision"]),
    # Return a METEOR scores
    (MeteorMetric(),["meteor_score"]),
]

caption_keys=list(parser.data.keys())
scores={k:{} for k in caption_keys}
for metric,metric_keys in metrics:
    results=metric([captions[parser.key] for captions in list(parser.data.values())],[captions["True"] for captions in list(parser.data.values())]) 
    scores = parse_metric_results(scores, caption_keys, results, metric_keys)
    
write_scores_to_tsv(scores,parser.output_path,parser.data,parser.key)