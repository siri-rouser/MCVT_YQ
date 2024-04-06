import os 
import numpy as np
import time
from CostMatrix import CostMatrix
from util_tools import calc_reid,update_output,reid_cat,reid_dict_filter,xytoxywh
# from hungarian_algorithm import algorithm


def main(previous_reid_dict,previous_rm_dict,query_cam,gallery_cam):
    H = {}
    abs_path = '/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/detect_merge'
    save_np_path = os.path.join(abs_path,query_cam,'dismat.npz') 

    query_track_path = os.path.join(abs_path,query_cam,f'{query_cam}_tracklet.pkl')
    gallery_track_path = os.path.join(abs_path,gallery_cam,f'{gallery_cam}_tracklet.pkl')
    cm = CostMatrix(query_track_path,gallery_track_path)
    dismat,q_track_ids,q_cam_ids, g_track_ids, g_cam_ids, q_times, g_times= cm.cost_matrix(metric = 'Cosine_Distance')
    # print(dismat.shape)
    # save temporal results in here
    np.savez(save_np_path,distmat=dismat,q_track_ids=q_track_ids,g_track_ids=g_track_ids,q_cam_ids=q_cam_ids,g_cam_ids=g_cam_ids,q_times=q_times,g_times=g_times)
    
    if previous_reid_dict:
        new_id = 0
       # print(previous_reid_dict)
        for cam_id in list(previous_reid_dict.keys()):
            for track_id in previous_reid_dict[cam_id]:
                track_info = previous_reid_dict[cam_id][track_id]
                if track_info['id'] > new_id:
                    new_id = track_info['id']
    else:
        new_id = 0

    reid_dict,rm_dict = calc_reid(dismat,q_track_ids,q_cam_ids, g_track_ids, g_cam_ids, q_times, g_times,new_id)
    # print(list(reid_dict.keys()))
    if previous_reid_dict:
        reid_dict = reid_cat(previous_reid_dict,reid_dict)

    # if not previous_reid_dict:
    #     print(reid_dict)

    if previous_rm_dict: # should update step by step!!!!
        if rm_dict:
            for cam_id in rm_dict:
                if cam_id in list(previous_reid_dict.keys()):
                    for track_id in rm_dict[cam_id]:
                        if track_id in list(previous_reid_dict[cam_id].keys()):
                            continue
                        else:
                            if cam_id not in previous_rm_dict:
                                previous_rm_dict[cam_id] = {}
                            if track_id not in previous_rm_dict[cam_id]:
                                previous_rm_dict[cam_id][track_id] = []
                            
                            previous_rm_dict[cam_id][track_id] = True
                            # new_rm_dict = previous_rm_dict
                else:
                    previous_reid_dict.update(rm_dict[cam_id])
            new_rm_dict = previous_rm_dict
        else:
            new_rm_dict = previous_rm_dict
    else:
        new_rm_dict = rm_dict

    return reid_dict, new_rm_dict



if __name__ == "__main__":
    seqs = ['c041','c042','c043','c044','c045','c046']
    abs_path = '/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/detect_merge'
    tracklets = os.path.join(abs_path,'tracklets.txt')
    final_res1 = os.path.join(abs_path,'final_track1.txt')
    final_res = os.path.join(abs_path,'final_track.txt')
    reid_dict = {}
    rm_dict = {}
  #  seqs = ['c041']
    for i in range(len(seqs)-1):
        print(f'start processing {seqs[i]} and {seqs[i+1]} pair ---')
        reid_dict, rm_dict = main(reid_dict,rm_dict,seqs[i],seqs[i+1])
        print(f'{seqs[i]} and {seqs[i+1]} pair finished -----')
    # test on
    # reid_dict, rm_dict = main(reid_dict,rm_dict,seqs[1],seqs[2])

    print(rm_dict)
    reid_dict = reid_dict_filter(reid_dict) # filter tracks only exist for one camera and once(in the first cam)

    with open(tracklets,"r") as f:
        or_tracks = f.readlines()
    g = open(final_res,"w")
    update_output(or_tracks,reid_dict,rm_dict,g)


    xytoxywh(final_res,final_res1)