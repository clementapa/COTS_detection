'''
    Author: Clément APAVOU
'''
from agents.trainer import Trainer
from utils.logger import init_logger
import argparse

parser = argparse.ArgumentParser(
    description='Script to launch the training',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--config_file',
                    default="config/config.yaml",
                    type=str,
                    help='Configuration train file')
parser.add_argument('--log', type=str, default="DEBUG", help='Level of log')
parser.add_argument('--fast_dev_run',
                    type=bool,
                    default=True,
                    help='fast dev run launch only one batch')
parser.add_argument('--no_test',
                    type=bool,
                    default=False,
                    help='Use the right wandb project')
parser.add_argument('--num_workers',
                    type=int,
                    default=0,
                    help='Number of workers for dataloaders')
parser.add_argument('--notebook',
                    type=bool,
                    default=False,
                    help='For tqdm bar progress in notebook')
parser.add_argument('--batch_size',
                    type=int,
                    help='batch size')
parser.add_argument(
    '--checkpoint',
    type=bool,
    default=False,
    help='If the training crashed and you want to relaunch with checkpoint'
)  # to implement
parser.add_argument(
    '--relaunch',
    type=int,
    help='If you want to relaunch the training to do more epoch'
)  # to implement
parser.add_argument('--csv_file',
                    type=str,
                    help='csv file')
parser.add_argument('--root_path',
                    type=str,
                    help='root_path train_image')
parser.add_argument('--it',
                    type=int,
                    help='number of iterations')
parser.add_argument('--epoch',
                    type=int,
                    help='number of epoch')                    
parser.add_argument('--load_checkpoint',
                    type=str,
                    help='checkpoint path')

args = parser.parse_args()

logger = init_logger("Trainer", args.log)

Trainer(args.config_file, logger, args)