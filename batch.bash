#!/bin/bash
SCORES_PATH="data/raw/raw_data/V5_docking_data_april_24/pivot_TITLE.csv"
FEA_DIR="data/raw/features/BL2"
# BL_SET="BL2"
JOBS=4
echo $SCORES_PATH
echo $FEA_DIR
# echo $FEA_DIR/BL2.dsc.parquet 
# python src/main_gen_dfs.py --scores_path $SCORES_PATH --fea_path $FEA_DIR/BL2.dsc.parquet   --par_jobs $JOBS
python src/main_gen_dfs.py --scores_path $SCORES_PATH --fea_path $FEA_DIR/BL2.ecfp2.parquet --fea_list ecfp2 --outdir ./out/V5_batch/ecfp2 --par_jobs $JOBS
python src/main_gen_dfs.py --scores_path $SCORES_PATH --fea_path $FEA_DIR/BL2.ecfp4.parquet --fea_list ecfp4 --outdir ./out/V5_batch/ecfp4 --par_jobs $JOBS
python src/main_gen_dfs.py --scores_path $SCORES_PATH --fea_path $FEA_DIR/BL2.ecfp6.parquet --fea_list ecfp6 --outdir ./out/V5_batch/ecfp6 --par_jobs $JOBS


