import argparse
import ast
import time
import os
import numpy as np
from data_process import *
from train import train
import json


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")




def main():
    train_dkt = 1
    arg_parser = argparse.ArgumentParser(description="train dkt model")
    arg_parser.add_argument('--data_dir', type=str, default='data')
    arg_parser.add_argument("--log_dir", type=str, default='logs')
    arg_parser.add_argument('--train', type=str2bool, default='t')
    arg_parser.add_argument('--hidden_neurons', type=str, default='[200,100]')
    arg_parser.add_argument("--lr", type=float, default=0.001)
    arg_parser.add_argument("--lr_decay", type=float, default=0.92)
    arg_parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')
    arg_parser.add_argument('--dropout_keep_probs', type=str, default='[0.6,0.8,1]')
    arg_parser.add_argument('--aggregator', type=str, default='sum')
    arg_parser.add_argument('--model', type=str, default='gikt')
    arg_parser.add_argument('--l2_weight', type=float, default=1e-8)
    arg_parser.add_argument('--limit_max_len',type=int,default=200)
    arg_parser.add_argument('--limit_min_len',type=int,default=3)


    arg_parser.add_argument('--dataset', type=str, default='assist2009')

    arg_parser.add_argument("--field_size", type=int, default=3)
    arg_parser.add_argument("--embedding_size", type=int, default=100)
    arg_parser.add_argument("--max_step", type=int, default=200)
    arg_parser.add_argument("--input_trans_size", type=int, default=100)
    arg_parser.add_argument("--batch_size", type=int, default=32)
    arg_parser.add_argument("--select_index", type=str, default='[0,1,2]')
    arg_parser.add_argument('--num_epochs', type=int, default=150)
    arg_parser.add_argument('--patience', type=int, default=10)
    arg_parser.add_argument('--n_hop', type=int, default=3)
    arg_parser.add_argument('--skill_neighbor_num', type=int, default=10)
    arg_parser.add_argument('--question_neighbor_num', type=int, default=4)
    arg_parser.add_argument('--hist_neighbor_num', type=int, default=0)  # history neighbor num
    arg_parser.add_argument('--next_neighbor_num', type=int, default=4)  # next neighbor num

    arg_parser.add_argument('--att_bound', type=float, default=0.5)#filtring irralate emb in topk selection
    arg_parser.add_argument('--sim_emb', type=str, default='skill_emb')#filtring irralate emb in topk selection
    arg_parser.add_argument('--seq_attn_window', type=int, default=20)
    arg_parser.add_argument('--hgkt_exer_layers', type=int, default=2)
    arg_parser.add_argument('--hgkt_schema_layers', type=int, default=1)



    args = arg_parser.parse_args()
    args.hidden_neurons = ast.literal_eval(args.hidden_neurons)
    args.dropout_keep_probs = ast.literal_eval(args.dropout_keep_probs)
    args.select_index = ast.literal_eval(args.select_index)
    train_dkt = args.train
    print(args.model)
    tag_path = os.path.join("%s_tag.txt"%args.dataset)
    tag = time.time()
    args.tag = tag

    config_name = 'logs/%f_config.json' % tag
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    config = {}
    for k,v in vars(args).items():
        config[k] = vars(args)[k]

    jsObj = json.dumps(config)

    fileObject = open(config_name, 'w')
    fileObject.write(jsObj)
    fileObject.close()
    print(config)
    args = data_process(args)

    train(args,train_dkt)

    log_file = open(tag_path, 'w')
    log_file.write(str(tag))

if __name__ == "__main__":
    main()
