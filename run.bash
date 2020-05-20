#!/bin/bash
MAIN_DIR=/vol/ml/apartin/projects/covid-19/ML-docking-dataframe-generator
SCORES_PATH=$MAIN_DIR/data/raw/raw_data/V5_docking_data_april_24/pivot_TITLE.csv
DATA_DIR=$MAIN_DIR/data/raw/features/BL2_test_images_and_others/
JOBS=64
# JOBS=1

echo "Scores   $SCORES_PATH"
echo "Data dir $DATA_DIR"

echo "Process dsc ..."
python src/main_gen_dfs.py --scores_path $SCORES_PATH \
    --fea_path $DATA_DIR/dsc.ids.0-30000.parquet --fea_list dsc \
    --img_path $DATA_DIR/images.ids.0-30000.pkl \
    --outdir ./out/test_images --par_jobs $JOBS

echo "Process ecfp2 ..."
python src/main_gen_dfs.py --scores_path $SCORES_PATH \
    --fea_path $DATA_DIR/ecfp2.ids.0-30000.parquet --fea_list ecfp2 \
    --outdir ./out/test_images --par_jobs $JOBS

echo "Process ecfp4 ..."
python src/main_gen_dfs.py --scores_path $SCORES_PATH \
    --fea_path $DATA_DIR/ecfp4.ids.0-30000.parquet --fea_list ecfp4 \
    --outdir ./out/test_images --par_jobs $JOBS

echo "Process ecfp6 ..."
python src/main_gen_dfs.py --scores_path $SCORES_PATH \
    --fea_path $DATA_DIR/ecfp6.ids.0-30000.parquet --fea_list ecfp6 \
    --outdir ./out/test_images --par_jobs $JOBS

