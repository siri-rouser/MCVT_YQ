MCMT_CONFIG_FILE="aic_all.yml"

# cd detector
# python gen_images_aic.py ${MCMT_CONFIG_FILE} # Generate IMGS with right ROI
# python detectimgyolov8.py

# cd ../reid # This part is only for feature extracting
# python extract_image_feat.py "aic_reid1.yml"
# python extract_image_feat.py "aic_reid2.yml"
# python extract_image_feat.py "aic_reid3.yml"
# python merge_reid_feat.py ${MCMT_CONFIG_FILE}

# cd ../mot
# cd tool
# python pre_process.py # it generate the feature file
cd mot
python DeepsortTracking.py ${MCMT_CONFIG_FILE}
python Data_process.py ${MCMT_CONFIG_FILE}

cd ../matching
python Trackletmatching.py
# Basic programming flow is like this! Re-check the file saving location...
# 28/03/2024 checked dataflow is working on! 