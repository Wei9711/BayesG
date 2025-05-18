

import argparse
import configparser
import logging
import threading
from torch.utils.tensorboard.writer import SummaryWriter
from envs.large_grid_env import LargeGridEnv
from envs.real_net_env import RealNetEnv
from envs.Large_city import Large_city_Env
from agents.models import IA2C, IA2C_FP, MA2C_NC, IA2C_CU, MA2C_CNET, MA2C_DIAL, BayesianGraph, IA2C_LToS
from utils import (Counter, Trainer, Tester, Evaluator,
                   check_dir, copy_file, find_file,
                   init_dir, init_log, init_test_flag)

import torch
import os
import datetime
import random
import numpy as np


def parse_args():
    default_base_dir = '/BayesG'
    default_config_dir = './config/config_BayesG_grid.ini'
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str, required=False,
                        default=default_base_dir, help="experiment base dir")
    
    # Create subparsers
    subparsers = parser.add_subparsers(dest='option', help="train or evaluate")
    
    # Train subparser
    sp_train = subparsers.add_parser('train', help='train a single agent under base dir')
    sp_train.add_argument('--config-dir', type=str, required=False,
                    default=default_config_dir, help="experiment config path")
    sp_train.add_argument('--n-heads', type=int, default=4,
                    help='Number of attention heads for GAT')
    sp_train.add_argument('--gnn-type', type=str, default='gcn',
                    choices=['gat', 'gcn', 'sage'],
                    help='Type of GNN to use (gat, gcn, or sage)')
    sp_train.add_argument('--is-mlp-gnn', type=bool, default=False,
                    help='Flag to use MLP for GNN (true or false)')
    sp_train.add_argument('--n-neighbor-hops', type=int, default=1,
                    help='Number of neighbor hops for GNN')
    # Evaluate subparser
    sp_eval = subparsers.add_parser('evaluate', help="evaluate and compare agents under base dir")
    sp_eval.add_argument('--evaluation-seeds', type=str, required=False,
                    default=','.join([str(i) for i in range(2000, 2500, 10)]),
                    help="random seeds for evaluation, split by ,")
    sp_eval.add_argument('--demo', action='store_true', help="shows SUMO gui")
    sp_eval.add_argument('--checkpoint', type=str, required=False, default=None,
                    help="Path to a specific model checkpoint to evaluate")
    
    args = parser.parse_args()
    if not args.option:
        parser.print_help()
        exit(1)
    return args


def init_env(config, port=None):
    env_name = config.get('env_name')
    # Generate random port between 0 and 50000 if not specified
    if port is None:
        port = random.randint(0, 50000)
    if env_name == "Grid_ATSC":# Adaptive traffic signal control
        return LargeGridEnv(config, port=port) # 5*5 synthetic traffic grid 
    elif env_name == "Monaco":
        return RealNetEnv(config, port=port) # Real-world Monaco City
    elif env_name == "Large_city":
        net_path = config.get('net_path')
        sim_path = config.get('sim_path')
        reward_scale = config.get('reward_scale')
        return Large_city_Env(net_path, sim_path,None,reward_scale) 
    else:
        raise ValueError(f"Invalid environment name: {env_name}")


def init_agent(env, config, total_step, seed,use_gpu):
    print("config:",config)
    # Read algo from MODEL_CONFIG section
    algo = config.get('MODEL_CONFIG', 'algo')
    env_name = config.get('ENV_CONFIG', 'env_name')
    if algo == 'IA2C':
        return IA2C(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                    total_step, config['MODEL_CONFIG'], seed=seed, use_gpu=use_gpu)
    elif algo == 'IA2C_FP':
        return IA2C_FP(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                     total_step, config['MODEL_CONFIG'], seed=seed, use_gpu=use_gpu)
    elif algo == 'NeurComm': # env.agent == 'ma2c_nc'
        if env_name == "Large_city":
            coop_gamma = -1
            adj_order = 30
            # neighbor_mask = env.cal_n_order_matrix(env.n_agent,adj_order,env.neighbor_mask) 
            # print("neighbor_mask:",env.neighbor_mask.sum(axis=1))
            # print("high order matrix:",neighbor_mask.sum(axis=1))
            gnn_type = config.get('MODEL_CONFIG', 'gnn_type', fallback='gat')
            return MA2C_NC(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, coop_gamma,
                       total_step, config['MODEL_CONFIG'], seed=seed, use_gpu=use_gpu, gnn_type=gnn_type)
        else:
            return MA2C_NC(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                       total_step, config['MODEL_CONFIG'], seed=seed, use_gpu=use_gpu)
    elif algo == 'BayesianGraph':
        if env_name == "Large_city":
            coop_gamma = -1
            adj_order = 30
            # neighbor_mask = env.cal_n_order_matrix(env.n_agent,adj_order,env.neighbor_mask) 
            # print("neighbor_mask:",env.neighbor_mask.sum(axis=1))
            # print("high order matrix:",neighbor_mask.sum(axis=1))
            gnn_type = config.get('MODEL_CONFIG', 'gnn_type', fallback='gat')
            return BayesianGraph(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, coop_gamma,
                       total_step, config['MODEL_CONFIG'], seed=seed, use_gpu=use_gpu)
        else:
            return BayesianGraph(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                         total_step, config['MODEL_CONFIG'], seed=seed, use_gpu=use_gpu)
    elif algo == 'CommNet':
        if env_name == "Large_city":
            coop_gamma = -1
            adj_order = 30
            neighbor_mask = env.cal_n_order_matrix(env.n_agent,adj_order,env.neighbor_mask) 
            return MA2C_CNET(env.n_s_ls, env.n_a_ls, neighbor_mask, env.distance_mask, coop_gamma,
                            total_step, config['MODEL_CONFIG'], seed=seed, use_gpu=use_gpu)
        return MA2C_CNET(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                         total_step, config['MODEL_CONFIG'], seed=seed, use_gpu=use_gpu)
    elif algo == 'Consensus':
        return IA2C_CU(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                       total_step, config['MODEL_CONFIG'], seed=seed, use_gpu=use_gpu)
    elif algo == 'ma2c_dial':
        return MA2C_DIAL(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                         total_step, config['MODEL_CONFIG'], seed=seed, use_gpu=use_gpu)
    elif algo == 'IA2C_LToS':
        return IA2C_LToS(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                         total_step, config['MODEL_CONFIG'], seed=seed, use_gpu=use_gpu)
    else:
        return None


def train(args):
    base_dir = args.base_dir
    config_dir = args.config_dir
    
    # Load config first
    config = configparser.ConfigParser()
    config.read(config_dir)
    
    # Update config with command line arguments
    if not 'MODEL_CONFIG' in config:
        config.add_section('MODEL_CONFIG')
    
    # Extract config name from config path
    config_name = os.path.splitext(os.path.basename(args.config_dir))[0]
    # Get environment name from config
    env_name = config['ENV_CONFIG']['env_name']
    # Modify log directory to include env_name and config name
    log_dir = os.path.join(base_dir, 'log', env_name, config_name)
    
    # Now we can pass the config to init_dir
    dirs = init_dir(base_dir, custom_log_dir=log_dir, config=config)
    
    init_log(dirs['log'])
    copy_file(config_dir, dirs['data'])

    # Log all configurations
    for section in config.sections():
        config_str = f"[{section}]\n"
        for key, value in config.items(section):
            config_str += f"{key} = {value}\n"
        logging.info(config_str.strip())  # Log the entire section at once

    # init env
    env = init_env(config['ENV_CONFIG'])
    print("Adjacency matrix:",env.neighbor_mask)
    print("Node degree:",env.neighbor_mask.sum(axis=1))
    # logging.info('Adjacency matrix:',env.neighbor_mask)
    # logging.info('Node degree:',env.neighbor_mask.sum(axis=1))
    logging.info('Training: a dim %r, agent dim: %d' % (env.n_a_ls, env.n_agent))
    logging.info('Adjacency matrix Degree: %s', str(env.neighbor_mask.sum(axis=1)))

    # init step counter
    total_step = int(config.getfloat('TRAIN_CONFIG', 'total_step'))
    test_step = int(config.getfloat('TRAIN_CONFIG', 'test_interval'))
    log_step = int(config.getfloat('TRAIN_CONFIG', 'log_interval'))
    global_counter = Counter(total_step, test_step, log_step)

    # init centralized or multi agent
    use_gpu = True
    seed = config.getint('ENV_CONFIG', 'seed')
    # print("MODEL_CONFIG contents:")
    # for key, value in config['MODEL_CONFIG'].items():
    #     print(f"    {key} = {value}")
    algo = config.get('MODEL_CONFIG', 'algo')
    env_name = config.get('ENV_CONFIG', 'env_name')
    model = init_agent(env, config, total_step, seed, use_gpu)
    
    load_model = config.getboolean('TRAIN_CONFIG', 'load_model')
    logging.info(f'Loading pre-trained model: {load_model}')
    if load_model:
        logging.info(f'Loading model from {dirs["model"]}')
        model.load(dirs['model'], train_mode=True)
        logging.info('Model loaded successfully')
    else:
        logging.info('No pre-trained model loaded')

    # disable multi-threading for safe SUMO implementation
    summary_writer = SummaryWriter(dirs['log'], flush_secs=10000)
    trainer = Trainer(env, env_name, algo, model, global_counter, summary_writer, output_path=dirs['data'], logger=logging)
    trainer.run()

    # save model
    final_step = global_counter.cur_step
    model.save(dirs['model'], final_step)
    summary_writer.close()


def evaluate_fn(agent_dir, output_dir, seeds, port, demo, checkpoint=None):
    agent = agent_dir.split('/')[-1]
    if not check_dir(agent_dir):
        logging.error('Evaluation: %s does not exist!' % agent)
        return
    # Find the config directory
    if checkpoint is not None:
        # Extract the timestamp from the checkpoint filename
        import re
        match = re.search(r'(\d{12})checkpoint', checkpoint)
        if match:
            timestamp = match.group(1)
            config_dir = find_file(os.path.join(agent_dir, 'data', timestamp))
            model_dir = os.path.join(agent_dir, 'model', timestamp)
        else:
            raise ValueError("Could not extract timestamp from checkpoint filename.")
    else:
        config_dir = find_file(agent_dir + '/data/')
        model_dir = agent_dir + '/model/'

    if not config_dir:
        return
    config = configparser.ConfigParser()
    config.read(config_dir)

        # Debug: Print config sections and values
    logging.info(f"Config file: {config_dir}")
    logging.info(f"Config sections: {config.sections()}")
    for section in config.sections():
        logging.info(f"Section {section}:")
        for key, value in config.items(section):
            logging.info(f"  {key} = {value}")
    # init env
    env = init_env(config['ENV_CONFIG'], port=port)
    env_name = config.get('ENV_CONFIG', 'env_name')
    env.init_test_seeds(seeds)

    # load model for agent
    model = init_agent(env, config, 0, 0,use_gpu = False)
    if model is None:
        return

    if checkpoint is not None:
        checkpoint_path = checkpoint if os.path.isabs(checkpoint) else os.path.join(model_dir, os.path.basename(checkpoint))
        if not model.load(checkpoint_path):
            return
    else:
        if not model.load(model_dir):
            return

    # collect evaluation data
    evaluator = Evaluator(env, env_name, model, output_dir, gui=demo)
    evaluator.run()


def evaluate(args):
    base_dir = args.base_dir
    if not args.demo:
        dirs = init_dir(base_dir, pathes=['eva_data', 'eva_log'])
        init_log(dirs['eva_log'])
        output_dir = dirs['eva_data']
    else:
        output_dir = None
    seeds = args.evaluation_seeds
    logging.info('Evaluation: random seeds: %s' % seeds)
    if not seeds:
        seeds = []
    else:
        seeds = [int(s) for s in seeds.split(',')]
    evaluate_fn(base_dir, output_dir, seeds, 1, args.demo, args.checkpoint)


if __name__ == '__main__':
    args = parse_args()
    if args.option == 'train':
        train(args)
    else:
        evaluate(args)
