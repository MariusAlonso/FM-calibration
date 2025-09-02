
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

from argparse import Namespace
from calib_utils import *
from sklearn.linear_model import LogisticRegression

import sys
sys.path.append('group_calibration')

from group_calibration.methods.group_calibration import calibrate_combine
from group_calibration.methods.temp_scaling import calibrate as ts_calibrate
from group_calibration.methods.mix_calibration import calibrate as ets_calibrate

from sklearn.model_selection import train_test_split
import tqdm
import json

datasets = []
train_data, val_data, test_data = [], [], []



import json
with open('MOMENT_train_sets.json', 'r') as f:
    train_sets = json.load(f)
    train_sets = {k: (torch.tensor(v[0]).float(), torch.tensor(v[1]).long()) for k, v in train_sets.items()}
with open('MOMENT_test_sets.json', 'r') as f:
    test_sets = json.load(f)
    test_sets = {k: (torch.tensor(v[0]).float(), torch.tensor(v[1]).long()) for k, v in test_sets.items()}

datasets = []
train_data, val_data, test_data = [], [], []
for dataset in tqdm.tqdm(train_sets.keys()):
    x_train = np.concatenate([train_sets[dataset][0], test_sets[dataset][0]], axis=0)
    y_train = np.concatenate([train_sets[dataset][1], test_sets[dataset][1]], axis=0)

    try:
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.5, random_state=42, stratify=y_train)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    except ValueError as e:
        print(f"Error splitting dataset {dataset}: {e}")
        continue

    train_data.append((x_train, y_train))
    val_data.append((x_val, y_val))
    test_data.append((x_test, y_test))
    datasets.append(dataset + '-MOMENT')

import json
with open('MANTIS_train_sets.json', 'r') as f:
    train_sets = json.load(f)
    train_sets = {k: (torch.tensor(v[0]).float(), torch.tensor(v[1]).long()) for k, v in train_sets.items()}
with open('MANTIS_test_sets.json', 'r') as f:
    test_sets = json.load(f)
    test_sets = {k: (torch.tensor(v[0]).float(), torch.tensor(v[1]).long()) for k, v in test_sets.items()}

for dataset in tqdm.tqdm(train_sets.keys()):
    x_train = np.concatenate([train_sets[dataset][0], test_sets[dataset][0]], axis=0)
    y_train = np.concatenate([train_sets[dataset][1], test_sets[dataset][1]], axis=0)

    try:
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.5, random_state=42, stratify=y_train)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    except ValueError as e:
        print(f"Error splitting dataset {dataset}: {e}")
        continue

    train_data.append((x_train, y_train))
    val_data.append((x_val, y_val))
    test_data.append((x_test, y_test))
    datasets.append(dataset + '-MANTIS')

with open('GPT2_train_sets.json', 'r') as f:
    train_sets = json.load(f)
    train_sets = {k: (torch.tensor(v[0]).float(), torch.tensor(v[1]).long()) for k, v in train_sets.items()}
with open('GPT2_test_sets.json', 'r') as f:
    test_sets = json.load(f)
    test_sets = {k: (torch.tensor(v[0]).float(), torch.tensor(v[1]).long()) for k, v in test_sets.items()}


for dataset in tqdm.tqdm(train_sets.keys()):
    x_train = np.concatenate([train_sets[dataset][0], test_sets[dataset][0]], axis=0)
    y_train = np.concatenate([train_sets[dataset][1], test_sets[dataset][1]], axis=0)

    try:
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.5, random_state=42, stratify=y_train)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    except ValueError as e:
        print(f"Error splitting dataset {dataset}: {e}")
        continue

    train_data.append((x_train, y_train))
    val_data.append((x_val, y_val))
    test_data.append((x_test, y_test))
    datasets.append(dataset + '-GPT2')


with open('BERT_train_sets.json', 'r') as f:
    train_sets = json.load(f)
    train_sets = {k: (torch.tensor(v[0]).float(), torch.tensor(v[1]).long()) for k, v in train_sets.items()}
with open('BERT_test_sets.json', 'r') as f:
    test_sets = json.load(f)
    test_sets = {k: (torch.tensor(v[0]).float(), torch.tensor(v[1]).long()) for k, v in test_sets.items()}


for dataset in tqdm.tqdm(train_sets.keys()):
    x_train = np.concatenate([train_sets[dataset][0], test_sets[dataset][0]], axis=0)
    y_train = np.concatenate([train_sets[dataset][1], test_sets[dataset][1]], axis=0)

    try:
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.5, random_state=42, stratify=y_train)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    except ValueError as e:
        print(f"Error splitting dataset {dataset}: {e}")
        continue

    train_data.append((x_train, y_train))
    val_data.append((x_val, y_val))
    test_data.append((x_test, y_test))
    datasets.append(dataset + '-BERT')


import json
with open('tabPFN/train_embeddings.json', 'r') as f:
    train_dict = json.load(f)
    train_dict = {k: torch.tensor(v).float().numpy() for k, v in train_dict.items()}
with open('tabPFN/val_embeddings.json', 'r') as f:
    val_dict = json.load(f)
    val_dict = {k: torch.tensor(v).float().numpy() for k, v in val_dict.items()}
with open('tabPFN/test_embeddings.json', 'r') as f:
    test_dict = json.load(f)
    test_dict = {k: torch.tensor(v).float().numpy() for k, v in test_dict.items()}
with open('tabPFN/train_labels.json', 'r') as f:
    train_labels = json.load(f)
    train_labels = {k: torch.tensor(v).long().numpy() for k, v in train_labels.items()}
with open('tabPFN/val_labels.json', 'r') as f:
    val_labels = json.load(f)
    val_labels = {k: torch.tensor(v).long().numpy() for k, v in val_labels.items()}
with open('tabPFN/test_labels.json', 'r') as f:
    test_labels = json.load(f)
    test_labels = {k: torch.tensor(v).long().numpy() for k, v in test_labels.items()}

train_data += list(zip(train_dict.values(), train_labels.values()))
val_data += list(zip(val_dict.values(), val_labels.values()))
test_data += list(zip(test_dict.values(), test_labels.values()))
datasets += [k + '-TAB' for k in train_dict.keys()]

import glob
FOLDER = 'clip'
files_embeddings = glob.glob(f'{FOLDER}/*_embeddings.pt')
clip_datasets = [f.split('\\')[-1].split('_')[0] for f in files_embeddings]

print(clip_datasets)

for dataset in tqdm.tqdm(clip_datasets):
    x_train = torch.load(f'{FOLDER}/{dataset}_embeddings.pt', map_location=torch.device('cpu')).numpy()
    y_train = torch.load(f'{FOLDER}/{dataset}_labels.pt', map_location=torch.device('cpu')).numpy()

    print(x_train.shape, y_train.shape, FOLDER)
    if len(x_train) > 5000:
        np.random.seed(0)
        perm = np.random.permutation(len(x_train))
        x_train, y_train = x_train[perm][:5000], y_train[perm][:5000]

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.5, random_state=42, stratify=y_train)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42, stratify=y_train)


    train_data.append((x_train, y_train))
    val_data.append((x_val, y_val))
    test_data.append((x_test, y_test))
    datasets.append(dataset + '-CLIP')


res_table = []

import tqdm

for I in tqdm.tqdm(range(len(datasets))):

    x_train, y_train = train_data[I]
    x_val, y_val = val_data[I]
    x_test, y_test = test_data[I]

    if len(x_val) < 40:
        continue

    # y_train = 1 * (y_train == 0)
    # y_val = 1 * (y_val == 0)
    # y_test = 1 * (y_test == 0)

    x = np.concatenate([x_train, x_val, x_test])
    y = np.concatenate([y_train, y_val, y_test])

    lreg = LogisticRegression(max_iter=1000, C=1e0, random_state=0).fit(x_train, y_train)
    print(datasets[I], lreg.score(x_test, y_test))


    y_pred_train = lreg.predict_proba(x_train)
    y_pred_val = lreg.predict_proba(x_val)
    y_pred_test = lreg.predict_proba(x_test)

    res_table.append([datasets[I], 'Accuracy', 'no cal', (y_pred_test.argmax(axis=1) == y_test).mean()])
    res_table.append([datasets[I], 'CE', 'no cal', safeCE(y_pred_test, y_test)])
    res_table.append([datasets[I], 'Brier', 'no cal', mc_brier_score(y_pred_test, y_test)])
    res_table.append([datasets[I], 'ECE', 'no cal', ece(y_pred_test, y_test)])
    res_table.append([datasets[I], 'smECE', 'no cal', smece(y_pred_test, y_test)])
    res_table.append([datasets[I], 'ACE', 'no cal', ace(y_pred_test, y_test)])      

    print('------------------', len(y_pred_val), len(y_val),x_train.shape, x_val.shape, y_val.shape)
    t_prior = temperature_scaling_grid(y_pred_val, y_val)
    y_cal_prior = F.softmax(torch.log(torch.tensor(y_pred_test)) * t_prior, dim=-1).numpy()
    y_cal_prior_val = F.softmax(torch.log(torch.tensor(y_pred_val)) * t_prior, dim=-1).numpy()

    res_table.append([datasets[I], 'CE', 'global', safeCE(y_cal_prior, y_test)])
    res_table.append([datasets[I], 'Brier', 'global', mc_brier_score(y_cal_prior, y_test)])
    res_table.append([datasets[I], 'ECE', 'global', ece(y_cal_prior, y_test)])  
    res_table.append([datasets[I], 'smECE', 'global', smece(y_cal_prior, y_test)])
    res_table.append([datasets[I], 'ACE', 'global', ace(y_cal_prior, y_test)])

    ct = ClusteredTemperature(nb_clusters=2, nb_ensembles=100, dim_frac=32)
    ct.fit(x_val, y_pred_val, y_val, x_train, seed=0, loss=0., loss_center=np.log(t_prior))
    temps, y_cal_test_2 = ct.predict(x_test, y_pred_test, return_t=True)

    res_table.append([datasets[I], 'CE', 'kmeans', safeCE(y_cal_test_2, y_test)])
    res_table.append([datasets[I], 'Brier', 'kmeans', mc_brier_score(y_cal_test_2, y_test)])
    res_table.append([datasets[I], 'ECE', 'kmeans', ece(y_cal_test_2, y_test)])
    res_table.append([datasets[I], 'smECE', 'kmeans', smece(y_cal_test_2, y_test)])
    res_table.append([datasets[I], 'ACE', 'kmeans', ace(y_cal_test_2, y_test)])

    # for gamma in [0.2,0.3,0.6,1.0,2.0]:
    #     for sigma in [1,2,3,5,8,12,20]: # [0.2,0.3,0.6,1.0,2.0]:
    #         pt = LocalCalib(gamma=gamma, n_bins=sigma, sigma=0.01, dim_reduce=32)
    #         pt.fit(x_val, y_cal_prior_val, y_val, x_train, seed=0)
    #         _, y_cal_test_4 = pt.predict(x_test, y_cal_prior, return_t=True)

    #         res_table.append([datasets[I], 'CE', f'g{gamma}s{sigma}', safeCE(y_cal_test_4, y_test)])
    #         res_table.append([datasets[I], 'Brier', f'g{gamma}s{sigma}', mc_brier_score(y_cal_test_4, y_test)])
    #         res_table.append([datasets[I], 'ECE', f'g{gamma}s{sigma}', ece(y_cal_test_4, y_test)]) 

    N_val = len(x_val)

    res = calibrate_combine(
        torch.tensor(x_val[:N_val//4]).float(), torch.tensor(np.log(y_pred_val[:N_val//4])).float(), torch.tensor(y_val[:N_val//4]),
        torch.tensor(x_val[N_val//4:]).float(), torch.tensor(np.log(y_pred_val[N_val//4:])).float(), torch.tensor(y_val[N_val//4:]),
        torch.tensor(x_test).float(), torch.tensor(np.log(y_pred_test)).float(), Namespace(
            w_net=Namespace(model='linear', weight_decay=0.1),
            optimizer=Namespace(name='lbfgs', steps=100, lr=1e-3),
            num_partitions=20,
            num_groups=2,
            base_calibrator=Namespace(name='temp_scaling')
        ),
        ts_calibrate, 0, None,
    )


    y_cal_test_5 = res['prob'].numpy()

    res_table.append([datasets[I], 'CE', 'kmeans', safeCE(y_cal_test_5, y_test)])
    res_table.append([datasets[I], 'Brier', 'kmeans', mc_brier_score(y_cal_test_5, y_test)])
    res_table.append([datasets[I], 'ECE', 'kmeans', ece(y_cal_test_5, y_test)])
    res_table.append([datasets[I], 'smECE', 'kmeans', smece(y_cal_test_5, y_test)])
    res_table.append([datasets[I], 'ACE', 'kmeans', ace(y_cal_test_5, y_test)])

import pandas as pd

pd.DataFrame(res_table).to_csv('calib_scores.csv')