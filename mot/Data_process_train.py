import os 
import pickle 
import numpy as np
import sys
sys.path.append('../')
from config import cfg


def post_process(seq):
    # cfg.merge_from_file(f'../config/{sys.argv[1]}')
    # cfg.freeze()
    # abs_path = cfg.DATA_DIR
    abs_path = '/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/detect_merge'
    abs_path = '/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results_yolov8_smiletrack/detect_merge'
    track_path = os.path.join(abs_path,seq,f'{seq}_mot.txt')
    combined_track_path = os.path.join(abs_path,'tracklets.txt')
    feat_path = os.path.join(abs_path,seq,f'{seq}_mot_feat_new.pkl')
    tracklet_pkl_file = os.path.join(abs_path,seq,f'{seq}_tracklet.pkl')
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
            track_feature[track_id] = []
            track_id_lib.append(track_id)
        if value['feat'].shape != (0,):
            track_feature[track_id].append(value['feat'])
            # print(value['feat'].shape)
      
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
  
    track_feature_avg = {}
    for track_id, features in track_feature.items():
        non_empty_features = [feat for feat in features if feat.shape != (0,)]
        if non_empty_features:
            track_feature_avg[track_id] = np.mean(np.stack(non_empty_features, axis=0), axis=0)
        else:
            track_feature_avg[track_id] = np.zeros((2048,))  # Handle cases where all features are empty


    new_data = {}
    for track_id in track_id_lib: # get how many tracklet we got(total number of track_id)
        feat = track_feature_avg[track_id]
        start_frame, end_frame = min(frame_nums[track_id]), max(frame_nums[track_id])
        travel_distance = np.linalg.norm(np.array(track_bbox_xy[track_id][-1]) - np.array(track_bbox_xy[track_id][0]))

        start_time, end_time =start_frame/10-0.1, end_frame/10-0.1

        track_time = end_time - start_time

        if travel_distance > 100 and track_time > 1:
            new_data[track_id] = {'cam': seq, 'track_id':track_id, 'start_time':start_time, 'end_time':end_time,'feat':feat}
            for i,frame_num in enumerate(frame_nums[track_id]):
                bbox_str = " ".join(map(str, map(int, track_bbox[track_id][i])))
                output_line = f'{str(seq)[2:]} {track_id} {frame_num} {bbox_str}\n'
                mot_result.write(output_line)


    pickle.dump(new_data, open(tracklet_pkl_file, 'wb'), pickle.HIGHEST_PROTOCOL)
    mot_result.close()

if __name__ == "__main__":
    seqs = ['c041','c042','c043','c044','c045','c046']
    for seq in seqs:
        print(f'start processing {seq} ---')
        post_process(seq)
        print(f'camera {seq} finished',seq)