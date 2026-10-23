"""Named benchmark workloads, independent of kernel/search configuration."""

from dataclasses import asdict

from weftmoe.config import ModelSpec


MODELS = {
    "qwen35b": ModelSpec(hidden=2048, intermediate=512, experts=256, top_k=8),
    "qwen122b": ModelSpec(hidden=3072, intermediate=1024, experts=256, top_k=8),
    "flash": ModelSpec(hidden=4096, intermediate=2048, experts=256, top_k=6),
}


def add_model_arguments(parser, *, default="qwen35b"):
    parser.add_argument("--model", choices=MODELS, default=default)
    for name in ("hidden", "intermediate", "experts", "top-k"):
        parser.add_argument("--" + name, type=int,
                            help="Override the corresponding model preset dimension")


def resolve_model(args):
    values = asdict(MODELS[args.model])
    values.update({key: getattr(args, key) for key in values
                   if getattr(args, key, None) is not None})
    model = ModelSpec(**values)
    for key, value in asdict(model).items():
        setattr(args, key, value)
    return model
