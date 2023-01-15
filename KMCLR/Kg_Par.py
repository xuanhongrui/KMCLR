import os
import torch
import argparse

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def parse_args():
    parser = argparse.ArgumentParser(description="'Model Params'")
    parser.add_argument('--bpr_batch', type=int,default=2048)
    parser.add_argument('--recdim', type=int,default=32)
    parser.add_argument('--layer', type=int,default=3)
    parser.add_argument('--lr', type=float,default=1e-3)
    parser.add_argument('--decay', type=float,default=1e-4)
    parser.add_argument('--dropout', type=int,default=1)
    parser.add_argument('--keepprob', type=float,default=0.7)
    parser.add_argument('--a_fold', type=int,default=100)
    parser.add_argument('--testbatch', type=int,default=4096)
    parser.add_argument('--dataset', type=str,default='Tmall')
    parser.add_argument('--topks', nargs='?',default="[20]")
    parser.add_argument('--tensorboard', type=int, default=0)
    parser.add_argument('--comment', type=str,default="lgn")
    parser.add_argument('--multicore', type=int, default=0)
    parser.add_argument('--pretrain', type=int, default=0)
    parser.add_argument('--test_file', type=str, default='test.txt')

    return parser.parse_args()
args = parse_args()

config = {}
config['bpr_batch_size'] = args.bpr_batch
config['latent_dim_rec'] = args.recdim
config['lightGCN_n_layers'] = args.layer
config['dropout'] = args.dropout
config['keep_prob'] = args.keepprob
config['A_n_fold'] = args.a_fold
config['test_u_batch_size'] = args.testbatch
config['multicore'] = args.multicore
config['lr'] = args.lr
config['decay'] = args.decay
config['pretrain'] = args.pretrain
config['A_split'] = False


GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")
entity_num_per_item = 10
kgc_temp = 0.2
kg_p_drop = 0.5
dataset = args.dataset
test_file = "/" + args.test_file



