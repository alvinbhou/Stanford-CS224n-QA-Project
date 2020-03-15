import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from collections import defaultdict
import pickle
import torch

# Root directory of output files
root_dir = './save/output'
dirs = sorted([f for f in os.listdir(root_dir)])
print(dirs)
# You could overwrite it yourself
# e.g. dirs = ['albert-large-v2-dev', 'roberta-large-dev']

dirs = ['bert-large-dev',
        'bert-large-train',
        'bert-large-batch6-dev',
        'bert-large-batch6-train',
        'bert-large-cls-weight0.5-without-force-pred-eval-dev',
        'bert-large-cls-weight0.5-without-force-pred-eval-train']


def generate(only_dev=True):
    features_train = []
    features_dev = []

    all_results_train = []
    all_results_dev = []

    tokenizers_train = []
    tokenizers_dev = []

    for model_dir in dirs:
        model_name = ''
        tokenizer = None
        all_results = None
        features = None

        with open(os.path.join(root_dir, model_dir, 'config.json')) as f:
            config = json.load(f)
            model_name = config['model_name'].replace('-dev', '').replace('-train', '').replace('-01', '')
        # We only want dev files for vote
        if only_dev and config['type'] == 'train':
            continue
        print(model_name, config['type'])

        with open(os.path.join(root_dir, model_dir, 'features.pkl'), 'rb') as f:
            features = pickle.load(f)

        with open(os.path.join(root_dir, model_dir, 'all_results.pkl'), 'rb') as f:
            all_results = pickle.load(f)

        with open(os.path.join(root_dir, model_dir, 'tokenizer.pkl'), 'rb') as f:
            tokenizer = pickle.load(f)

        if config['type'] == 'train':
            features_train.append(features)
            all_results_train.append(all_results)
            tokenizers_train.append(tokenizer)
        elif config['type'] == 'dev':
            features_dev.append(features)
            all_results_dev.append(all_results)
            tokenizers_dev.append(tokenizer)

    dev_to_save = [features_dev, all_results_dev, tokenizers_dev]
    with open(os.path.join(root_dir, 'saved_data_dev.pkl'), 'wb') as f:
        pickle.dump(dev_to_save, f)

    if not only_dev:
        with open(os.path.join(root_dir, 'saved_data_train.pkl'), 'wb') as f:
            pickle.dump([features_train, all_results_train, tokenizers_train], f)


if __name__ == "__main__":
    generate(only_dev=False)
