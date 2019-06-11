# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from agent import Agent
from brain import Brain
from env import Environment
#from log import Log
from pool import Pool
import eval_
from consts import *
import json, random, torch
import utils
import argparse
#SEED = 112233
#POOL_SIZE  =   2000000
#==============================
#%%
def is_time(epoch, trigger):
    return (trigger > 0) and (epoch % trigger == 0)

def main_(Load=True):
    
    np.set_printoptions(threshold=np.inf)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    #self.DATA_FILE = DATA_FILE
    DATA_FILE = './data/mb-train'
    DATA_VAL_FILE = './data/mb-val'
    META_FILE = './data/mb-meta'
    data = pd.read_pickle(DATA_FILE)
    
    data_val = pd.read_pickle(DATA_VAL_FILE)
    meta = pd.read_pickle(META_FILE)
    feats = meta.index
    costs = meta[META_COSTS]
    
    for col in COLUMN_DROP:
        if col in data.columns:
            data.drop(col, axis=1, inplace=True)    
            data_val.drop(col, axis=1, inplace=True)    
    
    #data[feats] = (data[feats] - meta[META_AVG]) / meta[META_STD]            # normalize
    #data_val[feats] = (data_val[feats] - meta[META_AVG]) / meta[META_STD]    # normalize
    #np.save('chck_train_data',data)
    print("Using", DATA_FILE, "with", len(data), "samples.")
    pool  = Pool(POOL_SIZE)
    env   = Environment(data, costs, FEATURE_FACTOR)
    brain = Brain(pool)
    print( " brain : " )
    agent = Agent(env, pool, brain)
    #log   = Log(data_val, costs, FEATURE_FACTOR, brain)
    epoch_start = 0
    #epoch_start = 0
    
    if not BLANK_INIT:
        print("Loading progress..")
        brain._load()
    
        with open('run.state', 'r') as file:
            save_data = json.load(file)
    
        epoch_start = save_data['epoch']
    
    brain.update_lr(epoch_start)
    agent.update_epsilon(epoch_start)
    
    #==============================
    print("Initializing pool..")
    for i in range(POOL_SIZE // AGENTS):
        utils.print_progress(i, POOL_SIZE // AGENTS)
        agent.step()
    
    pool.cuda()
    #%%    
    print("Starting..")
    #info = list()
    for epoch in range(epoch_start + 1, TRAINING_EPOCHS + 1):
        # SAVE
        if is_time(epoch, SAVE_EPOCHS):
            brain._save()
    
            save_data = {}
            save_data['epoch'] = epoch
            #info.append(test.test_action())
    
            with open('run.state', 'w') as file:
                json.dump(save_data, file)
            eval_.test_action()     
    
        # SET VALUES
        if is_time(epoch, EPSILON_UPDATE_EPOCHS):
            agent.update_epsilon(epoch)
    
        if is_time(epoch, LR_SC_EPOCHS):
            brain.update_lr(epoch)
    
        # LOG
        if is_time(epoch, LOG_EPOCHS):
            print("Epoch: {}/{}".format(epoch, TRAINING_EPOCHS))
               
            
            #log.log()
            #log.print_speed()
    
        if is_time(epoch, LOG_PERF_EPOCHS): pass
            #slog.log_perf()
    
        # TRAIN
        brain.train()
        
        for i in range(EPOCH_STEPS):
            agent.step()
        
    
            # sys.stdout.write('.'); sys.stdout.flush()        
#==============================
#main_()
if __name__ =="__main__":
    main_()
