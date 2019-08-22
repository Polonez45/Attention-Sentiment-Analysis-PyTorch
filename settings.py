import os
import argparse
import torch

def parse_arguments():
    """
    Prepares arguments from the command line indicated by the user,
    otherwise using default values.
    """
    argparser = argparse.ArgumentParser()
    default_logdir = "runs/imdb_lr0.001_weightd0.0001_batch32_epochs5/01.pt"

    argparser.add_argument("--train", default=True, action='store_true')
    argparser.add_argument("--no-train", dest='train', action='store_false')
    argparser.add_argument("--dataset", type=str, default="imdb")
    argparser.add_argument("--logdir", type=str, default=default_logdir)
    argparser.add_argument("--n_epochs", type=int, default=5)
    argparser.add_argument("--batch", type=int, default=32)
    argparser.add_argument("--lr", type=float, default=0.001)
    argparser.add_argument("--weight_decay", type=float, default=0.0001)
    argparser.add_argument("--dropout", type=float, default=0.5)
    argparser.add_argument("--seed", type=int, default=77)
    args = argparser.parse_args()

    return args


class Settings(object):
    """
    Class to share hyperparameters and various settings globally across
    modules.
    """
    PAD = "<PAD>"
    SOS = "<SOS>"
    EOS = "<EOS>"
    UNK = "<UNK>"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = parse_arguments()

    logdir = os.path.join("runs",
                          "{}_lr{}_weightd{}_batch{}_epochs{}".format(
                              args.dataset, args.lr, args.weight_decay,
                              args.batch, args.n_epochs))
