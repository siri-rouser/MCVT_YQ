import pickle
import os
import sys
sys.path.append('../matching/')
from CostMatrix import CostMatrix_Zone
from util_tools import original_calc_reid

def data_loading(data_path):
    # this function will load tracklets pass through all entry_zone and exit_zone
    entry_zone_data = {}
    exit_zone_data = {}
    with open(data_path,'rb') as data:
        cam_data = pickle.load(data)
    
    for key,value in cam_data.items():
        # NOTE: those two might contains same track!
        if value['entry_zone_cls'] == 'entry_zone':
            entry_zone_id = value['entry_zone_id']
            if entry_zone_id not in entry_zone_data:
                entry_zone_data[entry_zone_id] = {}
            entry_zone_data[entry_zone_id][value['track_id']] = {'cam':value['cam'],'track_id':value['track_id'],'start_time':value['start_time'],'end_time':value['end_time'],'conf':value['conf'],'entry_zone_id':value['entry_zone_id'],'exit_zone_id':value['exit_zone_id'],'feat':value['feat']}

        if value['exit_zone_cls'] == 'exit_zone':
            exit_zone_id = value['exit_zone_id']
            if exit_zone_id not in exit_zone_data:
                exit_zone_data[exit_zone_id] = {}
            exit_zone_data[exit_zone_id][value['track_id']] = {'cam':value['cam'],'track_id':value['track_id'],'start_time':value['start_time'],'end_time':value['end_time'],'conf':value['conf'],'entry_zone_id':value['entry_zone_id'],'exit_zone_id':value['exit_zone_id'],'feat':value['feat']}

    print(f'we have {len(entry_zone_data.keys())} entry zone in cam {data_path[-21:-17]}')
    print(f'we have {len(exit_zone_data.keys())} exit zone in cam {data_path[-21:-17]}')
    return entry_zone_data,exit_zone_data    

def zone_pair(entry_zone_data,exit_zone_data):

    pair_res_temp = {}
    for entry_zone_id in entry_zone_data:
        for exit_zone_id in exit_zone_data:
            cmz = CostMatrix_Zone(entry_zone_data[entry_zone_id],exit_zone_data[exit_zone_id])
            distmat, q_track_ids, q_cam_ids, g_track_ids, g_cam_ids, q_times, g_times, q_entry_zone,q_exit_zone,g_entry_zone,g_exit_zone = cmz.cost_matrix_zone('Cosine_Distance')
            _, _,ave_distance, pair_num = original_calc_reid(distmat,q_track_ids,g_track_ids,q_cam_ids,g_cam_ids,dis_thre=0.5,dis_remove=0.6)
            if q_cam_ids[0] not in pair_res_temp:
                pair_res_temp[q_cam_ids[0]] = {}
                if g_cam_ids[0] not in pair_res_temp[q_cam_ids[0]]:
                    pair_res_temp[q_cam_ids[0]][g_cam_ids[0]]= {}
            
            if ave_distance<1:
                pair_res_temp[q_cam_ids[0]][g_cam_ids[0]] = {'dis':ave_distance,'entry_exit_pair':[entry_zone_id,exit_zone_id]}
                if ave_distance < pair_res_temp[q_cam_ids[0]][g_cam_ids[0]]['dis']:
                    pair_res_temp[q_cam_ids[0]][g_cam_ids[0]] = {'dis':ave_distance,'entry_exit_pair':[entry_zone_id,exit_zone_id]}

    print(q_cam_ids[0])
    return pair_res_temp


def main(seq1,seq2,pair_res):
    abs_path = '/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/detect_merge'
    cam1_path = os.path.join(abs_path,seq1,f'{seq1}_new_tracklet.pkl')
    cam2_path = os.path.join(abs_path,seq2,f'{seq2}_new_tracklet.pkl')
    cam1_entry_zone_data, cam1_exit_zone_data = data_loading(cam1_path)
    cam2_entry_zone_data, cam2_exit_zone_data = data_loading(cam2_path)
    temp_res1 = zone_pair(cam1_entry_zone_data,cam2_exit_zone_data)
    for key,value in temp_res1.items():
        if key in pair_res:
            pair_res[key].update(value)  # Update existing nested dictionary
        else:
            pair_res[key] = value  # Create new key-value pair

    temp_res2 = zone_pair(cam2_entry_zone_data,cam1_exit_zone_data)

    for key,value in temp_res2.items():
        if key in pair_res:
            pair_res[key].update(value)  # Update existing nested dictionary
        else:
            pair_res[key] = value  # Create new key-value pair

    return pair_res
    



if __name__ == "__main__":
    pair_res = {}
    seqs = ['c041', 'c042', 'c043', 'c044', 'c045', 'c046']
    #seqs = ['c042','c043'] 

    # abs_path = '/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/detect_merge'
    # for seq in seqs:
    #     cam_path = os.path.join(abs_path,seq,f'{seq}_new_tracklet.pkl')
    #     data_loading(cam_path)

    for i in range(len(seqs)-1):
        pair = f'{seqs[i]}&{seqs[i+1]}'
        print(f'start processing {seqs[i]} and {seqs[i+1]} pair ---')
        pair_res = main(seqs[i],seqs[i+1],pair_res)
        print(f'{seqs[i]} and {seqs[i+1]} pair finished -----')
    print(pair_res)