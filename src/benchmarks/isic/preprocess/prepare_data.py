# -*- coding: utf-8 -*-
"""
Autor: André Pacheco
Email: pacheco.comp@gmail.com

"""

import sys
sys.path.insert(0,'../../..') # including the path to deep-tasks folder
from constants import RAUG_PATH
sys.path.insert(0,RAUG_PATH)
import os
from raug.utils.loader import parse_metadata
from raug.utils.loader import split_k_folder_csv, label_categorical_to_number
import pandas as pd

ISIC_BASE_PATH = "/home/leandro/Documentos/doutorado/dados/ISIC2019/ISIC2019"

# Load Metadata file
data_meta = parse_metadata (os.path.join(ISIC_BASE_PATH, "ISIC_2019_Training_Metadata.csv"), replace_nan="missing",
           cols_to_parse=['sex', 'anatom_site_general'], replace_rules={"age_approx": {"missing": 0}})

# Load groundtruth file
data_gt = parse_metadata (os.path.join(ISIC_BASE_PATH, "ISIC_2019_Training_GroundTruth.csv"))
data_gt["diagnostic"] = data_gt.drop(columns=["image"]).idxmax(axis=1)

# Merge files
data = pd.merge(data_meta, data_gt[["image", "diagnostic"]], on="image")

data = split_k_folder_csv(data, "diagnostic", save_path=None, k_folder=6, seed_number=8)
data = label_categorical_to_number (data, "diagnostic", col_target_number="diagnostic_number")
data_test = data[ data['folder'] == 6]
data_train = data[ data['folder'] != 6]

data_test.to_csv(os.path.join(ISIC_BASE_PATH, "ISIC2019_parsed_test.csv"), index=False)
data_train.to_csv(os.path.join(ISIC_BASE_PATH, "ISIC2019_parsed_train.csv"), index=False)








