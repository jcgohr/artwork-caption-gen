from parsers import FusionParser
from ranx import fuse,Run

parser = FusionParser()

combined_run = fuse(
runs=[Run.from_dict(run) for run in parser.runs],
norm="min-max", # Default normalization strategy
method="rrf",
)

combined_run.save(parser.output_path)