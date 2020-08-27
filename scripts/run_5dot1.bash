#!/bin/bash

# Example:
# bash scripts/run.bash 

PROJ_DIR=/vol/ml/apartin/projects/covid-19/ML-docking-dataframe-generator
DRG_SET=OZD
SCR_DIR=$PROJ_DIR/data/raw/docking/V5.1

# FEA_TYPE="descriptors fps images"
# FEA_TYPE="descriptors fps"
# FEA_TYPE="fps"
# FEA_TYPE="descriptors"

# sampling=random
sampling=flatten

# 100K
n_samples=100000
OUTDIR="$PROJ_DIR/out/V5.1-100K-$sampling-dd-fps/"

# 1M
# n_samples=1000000
# OUTDIR="$PROJ_DIR/out/V5.1-1M-$sampling-dd-fps/"

# 2M
# n_samples=2000000
# OUTDIR="$PROJ_DIR/out/V5.1-2M-$sampling-dd-fps/"

# JOBS=3
JOBS=1

echo "Drug set: $DRG_SET"
echo "Scores:   $SCR_DIR"
echo "Features: $FEA_TYPE"

DD_FPATH=$PROJ_DIR/data/raw/features/fea-agg-from-hpc/$DRG_SET/descriptors.mordred.parquet
FPS_FPATH=$PROJ_DIR/data/raw/features/fea-agg-from-hpc/$DRG_SET/fps.ecfp2.parquet

# ----------------------
#   Subset
# ----------------------
echo "Generate dataframes ..."
python src/main_gen_dfs_v5dot1.py \
    --drg_set $DRG_SET \
    --scr_dir $SCR_DIR \
    --dd_fpath $DD_FPATH \
    --fps_fpath $FPS_FPATH \
    --par_jobs $JOBS \
    --n_samples $n_samples \
    --baseline \
    --flatten \
    --outdir $OUTDIR

    # --fea_type $FEA_TYPE \
