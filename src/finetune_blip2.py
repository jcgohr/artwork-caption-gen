
from src.blip import train_blip
from src.parsers import BlipArgumentParser

parser = BlipArgumentParser()
args = parser.parse_args()

# Now you can use the validated arguments in your train_blip function
train_blip(
    train_file=args.train_file,
    val_file=args.val_file,
    train_captions_file=args.train_captions_file,
    val_captions_file=args.val_captions_file,
    output_dir=args.output_dir
)