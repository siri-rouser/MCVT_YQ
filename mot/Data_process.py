import os 
import math
import pickle 
import json
import numpy as np
import sys
sys.path.append('../')
from config import cfg

def euclidean_dist(pointA, pointB):
    if(len(pointA) != len(pointB)):
        raise Exception("expected point dimensionality to match")
    total = float(0)
    for dimension in range(0, len(pointA)):
        total += (pointA[dimension] - pointB[dimension])**2
    return math.sqrt(total)


def post_process(seq):
    # cfg.merge_from_file(f'../config/{sys.argv[1]}')
    # cfg.freeze()
    # abs_path = cfg.DATA_DIR
    abs_path = '/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/detect_merge'
    track_path = os.path.join(abs_path,seq,f'{seq}_mot.txt')
    combined_track_path = os.path.join(abs_path,'tracklets.txt')
    feat_path = os.path.join(abs_path,seq,f'{seq}_mot_feat_new.pkl')
    tracklet_pkl_file = os.path.join(abs_path,seq,f'{seq}_tracklet.pkl')
    zone_data_path = os.path.join(abs_path,seq,f'{seq}_zone_data.json')
    track_id_lib = []
    track_bbox_xy = {}
    track_bbox = {}
    track_feature = {}
    frame_nums = {}
    # Make sure the result is updated
    if seq == 'c041':
        mot_result =  open(combined_track_path,'w')
    else:
        mot_result =  open(combined_track_path,'a')

    with open(feat_path,'rb') as f:
        feat_data = pickle.load(f)

    for key,value in feat_data.items():
      
        track_id = value['track_id']

        if track_id not in track_feature:
            track_feature[track_id] = {}
            track_feature[track_id]['feat'] = []
            track_feature[track_id]['conf'] = []
            track_id_lib.append(track_id)
        track_feature[track_id]['feat'].append(value['feat'])
        track_feature[track_id]['conf'].append(value['conf'])
      
        if track_id not in track_bbox_xy:
            track_bbox_xy[track_id] = []

        if track_id not in track_bbox:
            track_bbox[track_id] = []

        x_central = (value['bbox'][0]+value['bbox'][2])/2
        y_central = (value['bbox'][1]+value['bbox'][3])/2
        track_bbox_xy[track_id].append([x_central,y_central])
        track_bbox[track_id].append(value['bbox'])

        if track_id not in frame_nums:
            frame_nums[track_id]=[]
      
        frame_nums[track_id].append(value['frame'])

# Track_id_lib is from small to big

  # This part us designed for temporal attention
    track_feature_avg = {}
    for track_id, features in track_feature.items():
        feats = np.array(features['feat'])  
        confs = np.array(features['conf']) 

        confs_normalized = confs / np.sum(confs)
        weighted_avg_feat = np.dot(confs_normalized, feats) / np.sum(confs_normalized)
        track_feature_avg[track_id] = weighted_avg_feat
    # print(track_bbox_xy[2])

    new_data = {}
    zone_data = {}
    flag = 0
    for track_id in track_id_lib: # get how many tracklet we got(total number of track_id)
        feat = track_feature_avg[track_id]
        start_frame, end_frame = min(frame_nums[track_id]), max(frame_nums[track_id])
        travel_distance = np.linalg.norm(np.array(track_bbox_xy[track_id][-1]) - np.array(track_bbox_xy[track_id][0]))

        start_time, end_time =start_frame/10-0.1, end_frame/10-0.1

        track_time = end_time - start_time

        if travel_distance > 200 and track_time > 2:
            new_data[track_id] = {'cam': seq, 'track_id':track_id, 'start_time':start_time, 'end_time':end_time,'entry_pos':track_bbox_xy[track_id][0:1], 'exit_pos':track_bbox_xy[track_id][-1:], 'feat':feat}
            for i,frame_num in enumerate(frame_nums[track_id]):
                bbox_str = " ".join(map(str, map(int, track_bbox[track_id][i])))
                output_line = f'{str(seq)[2:]} {track_id} {frame_num} {bbox_str}\n'
                mot_result.write(output_line)

        if travel_distance > 100 and track_time > 1:
            zone_data[flag] = {'entry_pos':track_bbox_xy[track_id][0:1], 'exit_pos':track_bbox_xy[track_id][-1:],'entry_vector':[], 'exit_vector': []}
            inital_point = np.array(track_bbox_xy[track_id][0])
            end_point = np.array(track_bbox_xy[track_id][-1])
            for index,point in enumerate(track_bbox_xy[track_id]):
                if euclidean_dist(inital_point, point) > 80:
                    entry_vector = point - inital_point
                    zone_data[flag]['entry_vector'] = entry_vector.tolist()
                    break
            for index,point in enumerate(track_bbox_xy[track_id]):
                if euclidean_dist(end_point, point) < 80:
                    exit_vector = end_point - point
                    zone_data[flag]['exit_vector'] = exit_vector.tolist()
                    break

            flag = flag +1 # use flag to order the sequence of the dict(), because the result of meanshift is based on sequence


    pickle.dump(new_data, open(tracklet_pkl_file, 'wb'), pickle.HIGHEST_PROTOCOL)
    mot_result.close()
    with open(zone_data_path, "w") as outfile:
        json.dump(zone_data, outfile)

if __name__ == "__main__":
    seqs = ['c041','c042','c043','c044','c045','c046']
    # seqs = ['c041']
    for seq in seqs:
        print(f'start processing {seq} ---')
        post_process(seq)
        print(f'camera {seq} finished',seq)