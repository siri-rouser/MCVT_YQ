import pickle

cam_names = ['c041','c042','c043','c044','c045','c046']
cache_file = '/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/detect_merge/c041/c041_dets_feat.pkl'
cam_name = 'c041'
with open(cache_file, 'rb') as fid: # 'rb' means writing binary file
    data = pickle.load(fid)
    frame_data = {}
    last_image_name = ""
  #  print(data['img002000'])
    for index,key in enumerate(data):
        image_name = key.split("_")[0]
        print(key)
        # if index == 0:
        #     frame_data[key] = data[key]
        #     last_image_name = image_name
        #     continue
        # if last_image_name != image_name:
        #     save_file(frame_data,cam_name,last_image_name)
        #     frame_data = {}
        #     last_image_name = image_name
        # frame_data[key] = data[key]
        # if index == len(data) -1:
        #     save_file(frame_data,cam_name,last_image_name)