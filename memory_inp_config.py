import argparse
import toml


class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)

    @classmethod
    def from_toml(cls, file_path):
        with open(file_path) as f:
            config_dict = toml.load(f)
        return cls(**config_dict)

    @classmethod
    def from_args(cls, args):
        return cls(**vars(args))

    def write_config(self, file_path):
        with open(file_path, "w") as f:
            toml.dump(self.__dict__, f)

    def get(self, item):
        return self.__dict__.get(item, None)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-name", type=str, help="Project name", default="meta-regression"
    )
    # training
    parser.add_argument("--seed", type=int, help="Random seed", default=0)
    parser.add_argument("--load-dir", type=str, help="Load directory")
    parser.add_argument("--load-it", type=str, help="Load iteration", default="best")
    parser.add_argument("--batch-size", type=int, help="Batch size", default=32)
    parser.add_argument("--num-epochs", type=int, help="Number of epochs", default=10)
    parser.add_argument(
        "--sort-context",
        type=str2bool,
        const=True,
        nargs="?",
        help="Sort context",
        default=False,
    )
    parser.add_argument("--lr", type=float, help="Learning rate", default=1e-3)
    parser.add_argument("--decay-lr", type=int, help="Decay learning rate", default=10)
    parser.add_argument("--train-split", type=str, help="Train split", default="train")
    parser.add_argument("--val-split", type=str, help="Validation split", default="val")
    parser.add_argument(
        "--n-trials", type=int, help="Number of optuna trials", default=1
    )
    parser.add_argument("--beta", type=float, help="Beta VAE", default=1)
    # general
    parser.add_argument("--device", type=str, help="Device", default="cuda")
    parser.add_argument("--input-dim", type=int, help="Input dimension", default=1)
    parser.add_argument("--output-dim", type=int, help="Output dimension", default=1)
    # dataloader
    parser.add_argument(
        "--dataset", type=str, help="Dataset", default="custom-regression"
    )
    parser.add_argument("--split-file", type=str, help="Split file", default=None)
    parser.add_argument(
        "--knowledge-type", type=str, help="Knowledge type", default="none"
    )
    parser.add_argument(
        "--min-num-context", type=int, help="Minimum number of context", default=0
    )
    parser.add_argument(
        "--max-num-context", type=int, help="Maximum number of context", default=100
    )
    parser.add_argument(
        "--num-targets", type=int, help="Number of targets", default=100
    )
    parser.add_argument("--noise", type=float, help="Observation noise std", default=0)
    parser.add_argument("--x-sampler", type=str, help="X sampler", default="uniform")
    # dataset encoder
    parser.add_argument(
        "--dataset-encoder-type",
        type=str,
        help="Dataset encoder type",
        default="set_transformer",
    )
    parser.add_argument(
        "--dataset-representation-dim",
        type=int,
        help="Dataset representation dimension",
        default=128,
    )
    parser.add_argument(
        "--set-transformer-num-heads",
        type=int,
        help="Number of heads in set transformer",
        default=4,
    )
    parser.add_argument(
        "--set-transformer-num-inds",
        type=int,
        help="Number of indices in set transformer",
        default=6,
    )
    parser.add_argument(
        "--set-transformer-ln",
        type=str2bool,
        help="Use layer normalization in set transformer",
        default=True,
    )
    parser.add_argument(
        "--set-transformer-hidden-dim",
        type=int,
        help="Hidden dimension in set transformer",
        default=128,
    )
    parser.add_argument(
        "--set-transformer-num-seeds",
        type=int,
        help="Number of seeds in set transformer",
        default=1,
    )
    parser.add_argument(
        "--x-transf-dim", type=int, help="X transformation dimension", default=128
    )
    parser.add_argument(
        "--xy-encoder-hidden-dim",
        type=int,
        help="Hidden dimension in xy encoder",
        default=128,
    )
    parser.add_argument(
        "--xy-encoder-num-hidden",
        type=int,
        help="Number of hidden layers in xy encoder",
        default=2,
    )
    # knowledge encoder
    parser.add_argument(
        "--knowledge-representation-dim",
        type=int,
        help="Knowledge representation dimension",
        default=128,
    )
    parser.add_argument(
        "--text-encoder",
        type=str,
        help="Text encoder",
        default="none",
        choices=["simple", "none", "roberta", "set", "set2", "mlp"],
    )
    parser.add_argument(
        "--roberta-freeze-llm",
        type=str2bool,
        const=True,
        nargs="?",
        help="Freeze LLM",
        default=True,
    )
    parser.add_argument(
        "--roberta-tune-llm-layer-norms",
        type=str2bool,
        const=True,
        nargs="?",
        help="Tune LLM layer norms",
        default=False,
    )
    parser.add_argument(
        "--set-embedding-num-hidden",
        type=int,
        help="Number of hidden layers in set embedding",
        default=2,
    )
    parser.add_argument(
        "--knowledge-encoder-num-hidden",
        type=int,
        help="Number of hidden layers in knowledge encoder",
        default=2,
    )
    parser.add_argument(
        "--knowledge-dropout", type=float, help="Knowledge dropout", default=0.0
    )
    # understanding encoder
    parser.add_argument(
        "--understanding-representation-dim",
        type=int,
        help="Understanding representation dimension",
        default=128,
    )
    parser.add_argument(
        "--knowledge-dataset-merge",
        type=str,
        help="Knowledge dataset merge",
        default="sum",
        choices=["sum", "concat", "mlp"],
    )
    parser.add_argument(
        "--knowledge-dataset-merger-hidden-dim",
        type=int,
        help="Hidden dimension in knowledge dataset merger",
        default=128,
    )
    parser.add_argument(
        "--knowledge-dataset-merger-num-hidden",
        type=int,
        help="Number of hidden layers in knowledge dataset merger",
        default=2,
    )
    parser.add_argument(
        "--understanding-encoder-num-hidden",
        type=int,
        help="Number of hidden layers in understanding encoder",
        default=2,
    )
    # data interaction encoder
    parser.add_argument(
        "--data-interaction-mlp-num-hidden",
        type=int,
        help="Number of hidden layers in data interaction MLP",
        default=2,
    )
    parser.add_argument(
        "--data-interaction-self-attention-hidden-dim",
        type=int,
        help="Hidden dimension in data interaction self attention",
        default=128,
    )
    parser.add_argument(
        "--data-interaction-self-attention-num-heads",
        type=int,
        help="Number of heads in data interaction self attention",
        default=2,
    )
    parser.add_argument(
        "--data-interaction-cross-attention-hidden-dim",
        type=int,
        help="Hidden dimension in data interaction cross attention",
        default=128,
    )
    parser.add_argument(
        "--data-interaction-cross-attention-num-heads",
        type=int,
        help="Number of heads in data interaction cross attention",
        default=2,
    )
    parser.add_argument(
        "--data-interaction-dim",
        type=int,
        help="Dimension in data interaction",
        default=128,
    )
    # memory module
    parser.add_argument(
        "--use-memory",
        type=str2bool,
        const=True,
        nargs="?",
        help="Use memory",
        default=True,
    )
    parser.add_argument(
        "--memory-slots", type=int, help="Number of memory slots", default=64
    )
    parser.add_argument(
        "--memory-gamma", type=float, help="Gamma in memory", default=0.7
    )
    # latent encoder module
    parser.add_argument(
        "--data-interaction-understanding-merge",
        type=str,
        help="Data interaction understanding merge",
        default="sum",
        choices=["sum", "concat", "mlp"],
    )
    parser.add_argument(
        "--data-interaction-understanding-merger-hidden-dim",
        type=int,
        help="Hidden dimension in data interaction understanding merger",
        default=128,
    )
    parser.add_argument(
        "--data-interaction-understanding-merger-num-hidden",
        type=int,
        help="Number of hidden layers in data interaction understanding merger",
        default=2,
    )
    parser.add_argument(
        "--latent-encoder-hidden-dim",
        type=int,
        help="Hidden dimension in latent encoder",
        default=128,
    )
    parser.add_argument(
        "--latent-encoder-num-hidden",
        type=int,
        help="Number of hidden layers in latent encoder",
        default=1,
    )
    # decoder module
    parser.add_argument(
        "--decoder-activation", type=str, help="Decoder activation", default="gelu"
    )
    parser.add_argument(
        "--decoder-hidden-dim", type=int, help="Hidden dimension in decoder", default=64
    )
    parser.add_argument(
        "--decoder-num-hidden",
        type=int,
        help="Number of hidden layers in decoder",
        default=3,
    )
    parser.add_argument(
        "--test-num-z-samples", type=int, help="Number of test z samples", default=16
    )
    parser.add_argument(
        "--train-num-z-samples", type=int, help="Number of train z samples", default=1
    )
    # saving args
    parser.add_argument(
        "--run-name-prefix", type=str, help="Run name prefix", default="run"
    )
    parser.add_argument(
        "--run-name-suffix", type=str, help="Run name suffix", default="tuned"
    )

    args = parser.parse_args()

    print("Setting memory_inp_config.toml")
    config = Config.from_args(args)

    config.write_config("memory_inp_config.toml")

    return config


if __name__ == "__main__":
    config = main()
