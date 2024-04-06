import pickle
import time

# Replace this with the path to your pickle file
pickle_file_path = '/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/detect_merge/c041/c041_mot_feat_new.pkl'
pickle_test_file_path = '/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/detect_merge/c041/c041_tracklet.pkl'
reid_feature_path = '/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/detect_reid1/c041/c041_dets_feat.pkl'
# Open the pickle file and load its contents
with open(pickle_file_path, 'rb') as file:
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