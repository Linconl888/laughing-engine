import os
import sys
import time
import torch
import random
import logging
import argparse
import numpy as np
from datetime import datetime

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_config():
    parser = argparse.ArgumentParser()
    num_classes = {'fraud': 2, 'fraud_after_TextFooler': 2, 'fraud_3.5_R1': 2, 'fraud_3.5_R2': 2, 'fraud_3.5_R3': 2, 'fraud_3.7_R1': 2, 'fraud_3.7_R2': 2, 'fraud_3.7_R3': 2}
    ''' Base '''
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--dataset', type=str, default='sst2', choices=num_classes.keys())
    parser.add_argument('--model_name', type=str, default='bert', choices=['bert', 'roberta', 'bert_chinese'])
    parser.add_argument('--method', type=str, default='dualcl', choices=['ce', 'scl', 'dualcl'])
    ''' Seed '''
    parser.add_argument('--seed', type=int, default=42, help='random seed for reproducibility')
    ''' Optimization '''
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=128)
    parser.add_argument('--num_epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--decay', type=float, default=0.01)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--temp', type=float, default=0.1)
    ''' Environment '''
    parser.add_argument('--backend', default=False, action='store_true')
    parser.add_argument('--timestamp', type=int, default='{:.0f}{:03}'.format(time.time(), random.randint(0, 999)))
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    args.num_classes = num_classes[args.dataset]
    args.device = torch.device(args.device)
    set_seed(args.seed)
    args.log_name = '{}_{}_{}_{}.log'.format(args.dataset, args.model_name, args.method, datetime.now().strftime('%Y-%m-%d_%H-%M-%S')[2:])
    if not os.path.exists('logs'):
        os.mkdir('logs')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.addHandler(logging.FileHandler(os.path.join('logs', args.log_name)))
    return args, logger
