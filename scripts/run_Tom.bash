#!/bin/bash

# Example:
# bash scripts/run.bash 

proj_dir=/vol/ml/apartin/projects/covid-19/ML-docking-dataframe-generator

# Data version
# ver="V5.1"
# ver="V7.0"
ver="Docking_Tom"

# Drug set
# drg_set=OZD
# drg_set=ORD
drg_set=MIX

# Path docking scores
# scr_dir=$proj_dir/data/raw/docking/$ver
scr_dir=$proj_dir/data/raw/raw_data/$ver

# fea_type="descriptors fps images"
# fea_type="descriptors fps"
# fea_type="fps"
# fea_type="descriptors"

# sampling=None
sampling=random
# sampling=flatten

# n_top=10000

# 100K dataset
# n_samples=100000
# outdir="$proj_dir/out/$ver-100K-$sampling"

# 1M dataset
# n_samples=1000000
# outdir="$proj_dir/out/$ver-1M-$sampling"

# 2M dataset
# n_samples=2000000
# outdir="$proj_dir/out/$ver-2M-$sampling"

# Full dataset
outdir="$proj_dir/out/$ver-all-$sampling"

# jobs=3
jobs=1

echo "Drug set: $drg_set"
echo "Scores:   $scr_dir"
echo "Features: $fea_type"

# dd_fpath=$proj_dir/data/raw/features/fea-agg-from-hpc/$drg_set/descriptors.mordred.parquet
# fps_fpath=$proj_dir/data/raw/features/fea-agg-from-hpc/$drg_set/fps.ecfp2.parquet
dd_fpath=$proj_dir/data/raw/features/receptors_Tom/dd.mordred.with.nans.parquet
fps_fpath=none

# ----------------------
#   Subset
# ----------------------
echo "Generate dataframes."
python src/main_gen_dfs_V7.py \
    --drg_set $drg_set \
    --scr_dir $scr_dir \
    --dd_fpath $dd_fpath \
    --fps_fpath $fps_fpath \
    --par_jobs $jobs \
    --sampling $sampling \
    --baseline \
    --outdir $outdir

    # --flatten \
    # --fea_type $fea_type \
