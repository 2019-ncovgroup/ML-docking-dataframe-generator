#!/bin/bash

PROJ_DIR=/vol/ml/apartin/projects/covid-19/ML-docking-dataframe-generator
SCORES_DIR=$PROJ_DIR/data/raw/docking/V5.1/OZD
FEA_DIR=$PROJ_DIR/data/raw/features/fea-subsets-hpc
DRG_SET=OZD
FEA_TYPE=descriptors
JOBS=16

echo "Scores   $SCORES_PATH"
echo "Data dir $DATA_DIR"

echo "Generate dataframes ..."
python src/main_gen_dfs_v5dot1.py \
    --scores_dir $SCORES_DIR \
    --fea_dir $FEA_DIR \
    --drg_set $DRG_SET \
    --fea_type $FEA_TYPE \
    --par_jobs $JOBS

