import copy
import numpy as np
import torch
import argparse

from tqdm import tqdm
from src.train_diffusion import main as train_diffusion_main
from experiments.params import params as EXPERIMENT_PARAMS

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Seed for randomness. -1 for random seed.")
    parser.add_argument("--dataname", type=str, default="adult", help="Name of dataset.")
    parser.add_argument("--timesteps", type=int, default=500, help="Number of diffusion steps.")
    parser.add_argument("--beta_1", type=float, default=0.0001, help="initial variance param")
    parser.add_argument("--beta_T", type=float, default=0.02, help="final variance param")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_sz", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument(
        "--attack",
        type=bool,
        default=False,
        help="global flag to perform attack",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument("--attack_lr", type=float, default=0.06)
    parser.add_argument(
        "--attack-epoch-lr", type=float, default=0.5, help="learning rate for epoch parameter in attack"
    )
    parser.add_argument(
        "--attack-epoch-lr-schedule",
        type=str,
        default="linear",
        help="learning rate schedule for epoch parameter in attack",
    )
    parser.add_argument("--optim_iters", type=int, default=2000, help="How many iterations per inversion attack step")
    parser.add_argument(
        "--skip_eval",
        type=bool,
        default=False,
        help="Skip computing DCR, NNDR, TableakScores",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--simple-attack",
        type=bool,
        default=False,
        help="whether to perform simplified attack",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument("--pool-size", type=int, default=1, help="number of parallel runs for inversion")
    parser.add_argument(
        "--save-model",
        type=bool,
        default=False,
        help="whether to save the trained model or not, essential for GIDM",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument("--save-dir", type=str, default="results/", help="directory to save results")
    parser.add_argument("--known-t-amount", type=int, default=0, help="number of known timesteps in attack")
    parser.add_argument(
        "--known-noise-amount", type=int, default=0, help="percentage of known values of starting noise in attack"
    )
    parser.add_argument(
        "--known-mask-amount",
        type=int,
        default=0,
        help="percentage of known rows of mask latent in attack (masked diffusion only)",
    )
    parser.add_argument(
        "--subset-size", type=int, default=None, help="If set, use only a subset of training data of this size"
    )

    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="what experiments to run",
        choices=sorted(EXPERIMENT_PARAMS.keys()),
    )
    parser.add_argument("--no-t", help="whether to not use timestep information in attack", action="store_true")
    parser.add_argument(
        "--t-init",
        type=str,
        default="midpoint",
        choices=["random", "midpoint"],
        help="dummy_t init strategy: 'random' = uniform over [0,T), 'midpoint' = T//2",
    )
    parser.add_argument(
        "--t-lr-mult",
        type=float,
        default=None,
        help="Timestep optimizer LR multiplier; defaults to 10.0",
    )
    parser.add_argument(
        "--noise-lr-mult",
        type=float,
        default=None,
        help="Noise optimizer LR multiplier; defaults to 1.0 for FT-Transformer and masked attacks, 0.1 otherwise",
    )
    parser.add_argument(
        "--no-noise", help="whether to not use starting noise information in attack", action="store_true"
    )
    parser.add_argument(
        "--no-mask",
        help="whether to freeze the mask latent in the attack (masked diffusion only)",
        action="store_true",
    )
    parser.add_argument("--no-fast-align", help="whether to not use fast align in ensembling", action="store_true")
    parser.add_argument("--no-align", help="whether to not use align in ensembling", action="store_true")
    parser.add_argument(
        "--reshuffle",
        help="reshuffle the client's minibatch composition every epoch (breaks the fixed batch order the cross-epoch ensemble assumes)",
        action="store_true",
    )
    parser.add_argument(
        "--global-ensemble",
        help="order-invariant cross-epoch ensembling: blend each batch into the globally-matched pool entries instead of a fixed batch slot (robust to --reshuffle)",
        action="store_true",
    )
    parser.add_argument(
        "--fedavg",
        help="attack a per-client FedAvg weight update (K local SGD steps) instead of a single-batch gradient",
        action="store_true",
    )
    parser.add_argument("--fedavg-local-steps", type=int, default=2, help="K, number of local SGD steps per FedAvg update")
    parser.add_argument("--fedavg-local-lr", type=float, default=None, help="local SGD learning rate for the FedAvg update (defaults to --lr)")
    parser.add_argument(
        "--align-diagnostic",
        help="also align each reconstruction with greedy fast-align AND Hungarian at eval time, logging both scores and their assignment disagreement",
        action="store_true",
    )
    parser.add_argument(
        "--anchor",
        type=str,
        default="first",
        help="alignment anchor strategy for multi-epoch ensembling: 'first' keeps the first epoch's anchor, 'latest' updates to the latest ensembled reconstruction",
        choices=["first", "latest"],
    )
    parser.add_argument(
        "--verbose",
        help="whether to print performance manager rows",
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--mask-verbose",
        help="print dummy latents (t, mask) during masked diffusion attack",
        action="store_true",
        default=False,
    )

    parser.add_argument("--perfect-pooling", help="whether to use perfect pooling in attack", action="store_true")

    parser.add_argument(
        "--defense",
        type=str,
        default=None,
        help="defense mechanism to use",
        choices=["clipping", "compression", "noise", "censor", "dp"],
    )
    # Certified DP-SGD defense: per-example clip + calibrated Gaussian noise.
    parser.add_argument("--dp-clip", type=float, default=1.0, help="DP-SGD per-example L2 clip norm C")
    parser.add_argument("--dp-epsilon", type=float, default=None, help="DP-SGD target epsilon (over --dp-steps); z is calibrated to it")
    parser.add_argument("--dp-delta", type=float, default=1e-5, help="DP-SGD target delta")
    parser.add_argument("--dp-steps", type=int, default=1, help="DP-SGD number of composed gradient releases for accounting")
    parser.add_argument("--dp-noise-multiplier", type=float, default=None, help="DP-SGD noise multiplier z (overrides --dp-epsilon if set)")
    parser.add_argument(
        "--defended-training",
        action="store_true",
        help="client trains on its own defended gradients (pays the defense's utility cost); "
        "logs per-epoch train_loss. Intended for utility runs with --no-attack",
    )
    parser.add_argument(
        "--backbone", type=str, default="mlp", choices=["mlp", "ft-transformer"],
        help="diffusion model backbone: mlp (default) or ft-transformer",
    )
    parser.add_argument(
        "--ft-init-std", type=float, default=1.0,
        help="FT-Transformer feature-tokenizer init std (balances input tokens vs timestep embedding)",
    )
    parser.add_argument(
        "--ft-head-init-std",
        type=float,
        default=0.01,
        help="FT-Transformer output-head initialization std; a small value starts near zero-noise prediction",
    )
    parser.add_argument(
        "--ft-lr",
        type=float,
        default=0.0001,
        help="FT-Transformer training learning rate",
    )
    parser.add_argument(
        "--ft-analytic-weight", type=float, default=0.0,
        help="Weight of the FT-Transformer closed-form tokenizer constraint (batch 1 only, where "
             "x_i = <dW_i,db_i>/||db_i||^2 recovers the noised input exactly). 0 disables.",
    )
    parser.add_argument(
        "--ft-analytic-init", action="store_true",
        help="Seed the attack's dummy data at the FT-Transformer closed-form tokenizer solution "
             "(batch 1 only) instead of random init",
    )
    # FT-Transformer capacity (d_model=256, 4 layers): chosen empirically on a small adult/batch-32
    # screen; a larger backbone is not assumed to always help the attack.
    parser.add_argument("--ft-d-model", type=int, default=256, help="FT-Transformer token width")
    parser.add_argument("--ft-nhead", type=int, default=8, help="FT-Transformer attention heads")
    parser.add_argument("--ft-nlayers", type=int, default=4, help="FT-Transformer encoder layers")
    parser.add_argument(
        "--ft-dim-feedforward", type=int, default=512, help="FT-Transformer feedforward width",
    )
    parser.add_argument("--baseline", type=str, default=None, help="baseline to use", choices=["gidm", "tableak"])
    parser.add_argument(
        "--random-latents",
        type=str,
        default="",
        help="Comma-separated subset of {t,noise,mask} (or 'all') to randomize once and freeze (no optimization). Used for random baselines.",
    )
    parser.add_argument(
        "--init-strategy",
        type=str,
        default=None,
        help="initialization strategy for dummy data in attack. None = uniform every epoch (data warm-start disabled by default)",
        choices=["uniform", "previous", "normal"],
    )
    parser.add_argument(
        "--warm-data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="warm-start dummy data from previous epoch's blended reconstruction (equivalent to --init-strategy=previous). On by default for multi-epoch attacks.",
    )
    parser.add_argument(
        "--warm-latents",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="warm-start dummy latents (t, noise, mask) from previous epoch. On by default.",
    )
    parser.add_argument(
        "--loss-fn",
        type=str,
        default="auto",
        help="loss function for gradient matching: mse, cossim, both, double-cossim, tokenwise (per-feature cosine on the FT-Transformer tokenizer), or auto (schema-aware)",
        choices=["mse", "cossim", "both", "double-cossim", "tokenwise", "auto"],
    )
    parser.add_argument(
        "--double-cossim-alpha",
        type=float,
        default=0.5,
        help="Weight of cont-block cosine term in --loss-fn=double-cossim (rest block gets 1-alpha)",
    )
    parser.add_argument(
        "--full-dataset-attack",
        type=bool,
        default=False,
        help="Collect all batch gradients then jointly optimize over full dataset",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--diffusion-type",
        type=str,
        default="gaussian",
        choices=["gaussian", "masked"],
        help="Type of diffusion process to use",
    )
    parser.add_argument(
        "--debug-prior",
        help="Log per-iter prior / grad-match trace CSVs under save_dir to debug the mask prior",
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--disable-prior",
        dest="no_prior",
        help="Disable the mask<->t coupling prior in masked diffusion attacks (diagnostic)",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--prior-peak",
        type=float,
        default=None,
        help="Override masked-diffusion KL prior peak weight (class default if unset)",
    )
    parser.add_argument(
        "--prior-cooldown-start",
        type=float,
        default=None,
        help="Override masked-diffusion KL prior cooldown-start fraction in [0,1] (class default if unset)",
    )

    args = parser.parse_args()

    if args.experiment is not None:
        args.attack = args.experiment not in ["save-model"]
        args.seed = -1  # when running experiments, we always randomize seed
        starting_args = copy.deepcopy(args)
        from experiments.params import params

        values = params[args.experiment]
        counter = 0
        to_skip = 0  # number of leading configs to skip, e.g. to resume an interrupted experiment
        for dataname, timesteps, batch_sz, epochs, run, (t, noise), extra_args in values:
            if counter < to_skip:
                counter += 1
                continue
            args = copy.deepcopy(starting_args)
            args.simple_attack = t == "simple"
            if extra_args is not None:
                for key, val in extra_args.items():
                    setattr(args, key, val)
                    if args.verbose:
                        tqdm.write(f"Setting extra arg {key} = {val}")
            if args.verbose:
                tqdm.write(
                    f"Running experiment {args.experiment} on dataset {dataname} with {timesteps=}, batch size={batch_sz}, optimizing t: {t}, optimizing noise: {noise}, for {epochs = }"
                )
            args.dataname = dataname
            args.timesteps = timesteps
            args.batch_sz = batch_sz
            args.epochs = epochs
            args.save_dir = f"results/{args.experiment}/{dataname}"
            args.no_t = not t
            args.no_noise = not noise

            if args.subset_size is None:
                args.subset_size = batch_sz * 100
            if getattr(args, "fixed_subset", False):
                # --experiment runs draw a fresh random data subset each time, while manual runs use
                # the fixed random_state=42 subset (src/train_diffusion.py keys this on
                # args.experiment). Clearing it makes the run identical to the manual command.
                args.experiment = None

            if args.save_model:
                args.batch_sz = 1024
                args.attack = False
                args.baseline = None
            train_diffusion_main(args)
    else:
        if args.save_model:
            args.batch_sz = 1024
            args.attack = False
            args.baseline = None
        if args.save_dir is None:
            args.save_dir = f"results/tests"
        train_diffusion_main(args)


if __name__ == "__main__":
    exit(main())
