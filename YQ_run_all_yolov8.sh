#!/bin/bash

MCMT_CONFIG_FILE="aic_all.yml"
output_file="/home/yuqiang/yl4300/project/MCVT_YQ/time_logbook_yolov8_latest.txt"

Adding an initial echo to clear doubts about file writing
echo "The YOLOv8 test!!" >> $output_file

# # Record the start time and note for this run
# echo "----------------------------------------" >> $output_file
echo "Start time: $(date)" >> $output_file
# echo "Note: Running the script for batch process analysis." >> $output_file

start_time=$(date +%s) # Save the start time
echo "--------------------object detection--------------------" >> $output_file
Entering detector directory and running python script
cd detector
{ time python gen_images_aic.py ${MCMT_CONFIG_FILE}; } 2>> $output_file

# Moving to nested directory and executing a shell script
{ time python detectimgyolov8.py; } 2>> $output_file
echo "--------------------feature extraction--------------------" >> $output_file


# # Moving to the reid directory for feature extraction
cd reid
python extract_image_feat.py "aic_reid1.yml"
python extract_image_feat.py "aic_reid2.yml"
python extract_image_feat.py "aic_reid3.yml"
python merge_reid_feat.py ${MCMT_CONFIG_FILE}

echo "--------------------SCT--------------------" >> $output_file
# Processing motion tracking
cd ../mot
cd tool
{ time python pre_process.py; } 2>> $output_file

cd ..
{ time python DeepsortTracking.py; } 2>> $output_file
{ time python Data_process.py; } 2>> $output_file
# Optional or commented out scripts
# { time python auto_zone.py; } 2>> $output_file
# { time python camera_link.py; } 2>> $output_file
# { time python time_window.py; } 2>> $output_file
echo "--------------------MCT--------------------" >> $output_file
# Matching tracklets in the matching directory
cd ../matching
{ time python Trackletmatching_benchmark.py; } 2>> $output_file

# Calculating and recording the total execution time
end_time=$(date +%s)
echo "Total execution time: $((end_time - start_time)) seconds" >> $output_file
echo "----------------------------------------" >> $output_file
