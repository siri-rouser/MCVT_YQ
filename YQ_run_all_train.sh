MCMT_CONFIG_FILE="aic_all_train.yml"

# cd detector
# python gen_images_aic_train.py ${MCMT_CONFIG_FILE} # Generate IMGS with right ROI
# python detectimgyolov9_train.py

# cd ../reid # This part is only for feature extracting
# python extract_image_feat_train.py "aic_reid1_train.yml"
# python extract_image_feat_train.py "aic_reid2_train.yml"
# python extract_image_feat_train.py "aic_reid3_train.yml"
# python merge_reid_feat_train.py ${MCMT_CONFIG_FILE}

# cd ../mot
# cd tool
# python pre_process_train.py # it generate the feature file
cd mot
python DeepsortTracking_train.py ${MCMT_CONFIG_FILE}
python Data_process_train.py ${MCMT_CONFIG_FILE}

# cd ../matching
# python Trackletmatching_train.py ${MCMT_CONFIG_FILE}
# # Basic programming flow is like this! Re-check the file saving location...
# # 28/03/2024 checked dataflow is working on! 