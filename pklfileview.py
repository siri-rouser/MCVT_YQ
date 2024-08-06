import pickle
import argparse
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--cam", type=int)
parser.add_argument("-f", "--file", type=str)
# parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
#                     help='example to call this script: python pklfileview.py -c 41 -fmot_feat_new')


args = parser.parse_args()

cam = f'c0{args.cam}'

# Replace this with the path to your pickle file
mot_feat_new = f'/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/detect_merge/{cam}/{cam}_mot_feat_new.pkl'
new_tracklet = f'/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/detect_merge/{cam}/{cam}_new_tracklet.pkl'
tracklet = f'/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/detect_merge/{cam}/{cam}_tracklet.pkl'
dets_feat = f'/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/detect_reid1/{cam}/{cam}_dets_feat.pkl'
# Open the pickle file and load its contents

if args.file == 'mot_feat_new':
    path = mot_feat_new
elif args.file == 'new_tracklet':
    path = new_tracklet
elif args.file == 'dets_feat':
    path = dets_feat
elif args.file == 'tracklet':
    path = tracklet
else:
    print('pls input the right file arguments!')
    sys.exit()

with open(path, 'rb') as file:
    data = pickle.load(file)
i=0
id_list = []
if isinstance(data, (list, dict)):
    for key, value in data.items() if isinstance(data, dict) else enumerate(data):
        print(f"Key: {key}, Value: {value}")
  
        if 'bbox' not in value:
            i=i+1
    #     if i>300:
    #         break
        # if value['track_id'] not in id_list:
        #     id_list.append(value['track_id'])

    print(f'We got {i} wrong dets in cam')  