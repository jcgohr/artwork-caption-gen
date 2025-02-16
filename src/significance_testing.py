from ranx import compare,Run,Qrels
from parsers import SignificanceTestingParser

parser=SignificanceTestingParser()

report = compare(
    qrels=Qrels.from_dict(parser.qrel),
    runs=[Run.from_dict(run) for run in parser.runs],
    metrics=["precision@1", "mrr"],
    max_p=0.05  # P-value threshold
)

print(report)