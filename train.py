import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import math
import pandas
import GPUtil
import argparse

import warnings

# class imports
from wcst import WCST
from model import Transformer
# from utils import adapt_batch_for_encoder
# we do the encoder model later for now the decoder is most important

warnings.filterwarnings("ignore")

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wcst_generator = WCST(args.batch_size)

    # TODO: transformer model
