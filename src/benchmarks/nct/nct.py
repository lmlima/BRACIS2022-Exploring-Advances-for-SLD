# -*- coding: utf-8 -*-
"""
Autor: André Pacheco
Email: pacheco.comp@gmail.com

"""

import sys
sys.path.insert(0,'../../') # including the path to deep-tasks folder
sys.path.insert(0,'../../my_models') # including the path to my_models folder
from constants import RAUG_PATH
sys.path.insert(0,RAUG_PATH)
from raug.loader import get_data_loader
from raug.train import fit_model
from raug.eval import test_model
from my_model import set_model
import pandas as pd
import os
import torch.optim as optim
import torch.nn as nn
import torch
from aug_nct import ImgTrainTransform, ImgEvalTransform
import datetime
from sacred import Experiment
from sacred.observers import FileStorageObserver
from raug.utils.loader import get_labels_frequency

# Dataset download: https://zenodo.org/record/1214456#.YYGN_7zMKV4


# Starting sacred experiment
ex = Experiment()

@ex.config
def cnfg():

    # Dataset variables
    _folder = 1
    _base_path = "/home/patcha/Datasets/CRC-VAL-HE-7K"
    _csv_path_train = os.path.join(_base_path, "NCT_train.csv")
    _imgs_folder_train = os.path.join(_base_path, "imgs")

    _batch_size = 30
    _epochs = 150

    # Training variables
    _best_metric = "loss"
    _pretrained = True
    _lr_init = 0.001
    _sched_factor = 0.1
    _sched_min_lr = 1e-6
    _sched_patience = 10
    _early_stop = 15
    _weights = "frequency"

    _model_name = 'mobilenet'
    _save_folder = "results/" + _model_name + "_fold_" + str(_folder) + "_" + str(datetime.datetime.now()).replace(' ', '')

    # This is used to configure the sacred storage observer. In brief, it says to sacred to save its stuffs in
    # _save_folder. You don't need to worry about that.
    SACRED_OBSERVER = FileStorageObserver(_save_folder)
    ex.observers.append(SACRED_OBSERVER)

@ex.automain
def main (_folder, _csv_path_train, _imgs_folder_train, _lr_init, _sched_factor, _sched_min_lr, _sched_patience,
          _batch_size, _epochs, _early_stop, _weights, _model_name, _pretrained, _save_folder,
          _best_metric):


    _metric_options = {
        'save_all_path': os.path.join(_save_folder, "best_metrics"),
        'pred_name_scores': 'predictions_best_test.csv',
        'normalize_conf_matrix': True}
    _checkpoint_best = os.path.join(_save_folder, 'best-checkpoint/best-checkpoint.pth')

    # Loading the csv file
    csv_all_folders = pd.read_csv(_csv_path_train)

    print("-" * 50)
    print("- Loading validation data...")
    val_csv_folder = csv_all_folders[ (csv_all_folders['folder'] == _folder) ]
    train_csv_folder = csv_all_folders[ csv_all_folders['folder'] != _folder ]

    # Loading validation data
    val_labels = val_csv_folder['target_number'].values
    val_labels_str = val_csv_folder['target'].values
    val_imgs_path_ = val_csv_folder['Image_id'].values
    val_imgs_path = ["{}/{}/{}".format(_imgs_folder_train, lab, img_id) for img_id, lab in zip(val_imgs_path_, val_labels_str)]
    val_meta_data = None
    val_data_loader = get_data_loader (val_imgs_path, val_labels, val_meta_data, transform=ImgEvalTransform(),
                                       batch_size=_batch_size, shuf=True, num_workers=16, pin_memory=True)
    print("-- Validation partition loaded with {} images".format(len(val_data_loader)*_batch_size))

    print("- Loading training data...")
    train_labels = train_csv_folder['target_number'].values
    train_labels_str = train_csv_folder['target'].values
    train_imgs_path_ = train_csv_folder['Image_id'].values
    train_imgs_path = ["{}/{}/{}".format(_imgs_folder_train, lab, img_id) for img_id, lab in zip(train_imgs_path_, train_labels_str)]
    train_meta_data = None
    train_data_loader = get_data_loader (train_imgs_path, train_labels, train_meta_data, transform=ImgTrainTransform(),
                                       batch_size=_batch_size, shuf=True, num_workers=16, pin_memory=True)
    print("-- Training partition loaded with {} images".format(len(train_data_loader)*_batch_size))

    print("-"*50)
    ####################################################################################################################

    ser_lab_freq = get_labels_frequency(train_csv_folder, "target", "Image_id")
    _labels_name = ser_lab_freq.index.values
    _freq = ser_lab_freq.values
    ####################################################################################################################
    print("- Loading", _model_name)

    model = set_model(_model_name, len(_labels_name), pretrained=_pretrained)
    ####################################################################################################################
    if _weights == 'frequency':
        _weights = (_freq.sum() / _freq).round(3)
    loss_fn = nn.CrossEntropyLoss(weight=torch.Tensor(_weights).cuda())
    # optimizer = optim.Adam(model.parameters(), lr=_lr_init)
    optimizer = optim.SGD(model.parameters(), lr=_lr_init, momentum=0.9, weight_decay=0.001)
    scheduler_lr = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=_sched_factor, min_lr=_sched_min_lr,
                                                                    patience=_sched_patience)
    ####################################################################################################################

    print("- Starting the training phase...")
    print("-" * 50)
    fit_model (model, train_data_loader, val_data_loader, optimizer=optimizer, loss_fn=loss_fn, epochs=_epochs,
               epochs_early_stop=_early_stop, save_folder=_save_folder, initial_model=None,
               device=None, schedule_lr=scheduler_lr, config_bot=None, model_name="CNN", resume_train=False,
               history_plot=True, val_metrics=["auc"], best_metric=_best_metric)
    ####################################################################################################################

    # Testing the validation partition
    print("- Evaluating the validation partition...")
    test_model (model, val_data_loader, checkpoint_path=_checkpoint_best, loss_fn=loss_fn, save_pred=True,
                partition_name='eval', metrics_to_comp='all', class_names=_labels_name, metrics_options=_metric_options,
                apply_softmax=True, verbose=False)
    ####################################################################################################################