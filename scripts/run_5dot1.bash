#!/bin/bash

# Example:
# bash scripts/run.bash 

PROJ_DIR=/vol/ml/apartin/projects/covid-19/ML-docking-dataframe-generator
DRG_SET=OZD
SCR_DIR=$PROJ_DIR/data/raw/docking/V5.1

# FEA_TYPE="descriptors fps images"
# FEA_TYPE="descriptors fps"
FEA_TYPE="descriptors"

# OUTDIR="$PROJ_DIR/out/test_hpc/"
OUTDIR="$PROJ_DIR/out/test_1M_flatten/"
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
    --n_samples 1000000 \
    --flatten \
    --outdir $OUTDIR

    # --n_samples 20000 \
    # --n_top 3000 \
    # --flatten \

    # --baseline \
    # --fea_dir $FEA_DIR \

