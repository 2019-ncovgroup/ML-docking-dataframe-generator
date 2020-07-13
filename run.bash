#!/bin/bash

# Example:
# bash run.bash 

PROJ_DIR=/vol/ml/apartin/projects/covid-19/ML-docking-dataframe-generator
DRG_SET=OZD
SCR_DIR=$PROJ_DIR/data/raw/docking/V5.1

# FEA_DIR=$PROJ_DIR/data/raw/features/fea-subsets-hpc

# FEA_TYPE="descriptors fps images"
FEA_TYPE="descriptors"

OUTDIR="$PROJ_DIR/out/test_hpc/"
# JOBS=16
JOBS=1

echo "Drug set: $DRG_SET"
echo "Scores:   $SCR_DIR"
echo "Features: $FEA_TYPE"

echo "Generate dataframes ..."
python src/main_gen_dfs_v5dot1.py \
    --drg_set $DRG_SET \
    --scr_dir $SCR_DIR \
    --fea_type $FEA_TYPE \
    --par_jobs $JOBS \
    --baseline \
    --outdir $OUTDIR

    # --n_samples 20000 \
    # --n_top 3000 \
    # --flatten \

    # --fea_dir $FEA_DIR \

