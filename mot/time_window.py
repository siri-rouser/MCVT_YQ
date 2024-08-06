from camera_link import data_loading
import os
import json
import sys
sys.path.append('../matching/')
from CostMatrix import CostMatrix_Zone
from util_tools import original_calc_reid

def data_filter(data):
    filtered_data = {}
    for track_id in data:
        total_time = data[track_id]['end_time'] - data[track_id]['start_time']
        if total_time > 1.5 and data[track_id]['conf'] > 0.7:
            filtered_data[track_id] = {'cam':data[track_id]['cam'],'track_id':data[track_id]['track_id'],'start_time':data[track_id]['start_time'],'end_time':data[track_id]['end_time'],'conf':data[track_id]['conf'],'entry_zone_id':data[track_id]['entry_zone_id'],'exit_zone_id':data[track_id]['exit_zone_id'],'feat':data[track_id]['feat']}

    return filtered_data

def _time_window(seq1,seq2,cam_pair):
  
    abs_path = '/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/detect_merge'
    time_transits = []
    cam1_path = os.path.join(abs_path,seq1,f'{seq1}_new_tracklet.pkl')
    cam2_path = os.path.join(abs_path,seq2,f'{seq2}_new_tracklet.pkl')
    cam1_entry_zone_data, cam1_exit_zone_data = data_loading(cam1_path)
    cam2_entry_zone_data, cam2_exit_zone_data = data_loading(cam2_path)
    # entry_zone_data[entry_zone_id][value['track_id']] = {'cam':value['cam'],'track_id':value['track_id'],'start_time':value['start_time'],'end_time':value['end_time'],'conf':value['conf'],'entry_zone_id':value['entry_zone_id'],'exit_zone_id':value['exit_zone_id'],'feat':value['feat']}
    # For the first pair!!! from c041 to c042
    entry_zone_id, exit_zone_id = cam_pair[seq1[-2:]][seq2[-2:]]['entry_exit_pair'][0], cam_pair[seq1[-2:]][seq2[-2:]]['entry_exit_pair'][1]
    processed_cam1_entry_zone_data = data_filter(cam1_entry_zone_data[entry_zone_id])

    processed_cam2_exit_zone_data = data_filter(cam2_exit_zone_data[exit_zone_id])
    cmz = CostMatrix_Zone(processed_cam1_entry_zone_data,processed_cam2_exit_zone_data)
    distmat, q_track_ids, q_cam_ids, g_track_ids, g_cam_ids, q_times, g_times, q_entry_zone,q_exit_zone,g_entry_zone,g_exit_zone = cmz.cost_matrix_zone('Cosine_Distance')

    reid_dict, _,ave_distance, pair_num = original_calc_reid(distmat,q_track_ids,g_track_ids,q_cam_ids,g_cam_ids,dis_thre=0.75,dis_remove=0.75)

    for track_id1 in reid_dict[int(seq1[-2:])]:
        if reid_dict[int(seq1[-2:])][track_id1]['dis'] < 0.3:
            id_temp = reid_dict[int(seq1[-2:])][track_id1]['id']
            for track_id2 in reid_dict[int(seq2[-2:])]:
                if reid_dict[int(seq2[-2:])][track_id2]['id'] == id_temp:
                    time_transit = processed_cam1_entry_zone_data[track_id1]['start_time']-processed_cam2_exit_zone_data[track_id2]['end_time']
                    if time_transit > 0:
                        time_transits.append(time_transit)

    # For the second pair!!! from c042 to c041
    entry_zone_id, exit_zone_id = cam_pair[seq2[-2:]][seq1[-2:]]['entry_exit_pair'][0], cam_pair[seq2[-2:]][seq1[-2:]]['entry_exit_pair'][1]
    processed_cam2_entry_zone_data = data_filter(cam2_entry_zone_data[entry_zone_id])
    processed_cam1_exit_zone_data = data_filter(cam1_exit_zone_data[exit_zone_id])
    cmz = CostMatrix_Zone(processed_cam2_entry_zone_data,processed_cam1_exit_zone_data)
    distmat, q_track_ids, q_cam_ids, g_track_ids, g_cam_ids, q_times, g_times, q_entry_zone,q_exit_zone,g_entry_zone,g_exit_zone = cmz.cost_matrix_zone('Cosine_Distance')
    reid_dict, _,ave_distance, pair_num = original_calc_reid(distmat,q_track_ids,g_track_ids,q_cam_ids,g_cam_ids,dis_thre=0.5,dis_remove=0.5)

    print(reid_dict)

    for track_id1 in reid_dict[int(seq2[-2:])]:
        if reid_dict[int(seq2[-2:])][track_id1]['dis'] < 0.3:
            id_temp = reid_dict[int(seq2[-2:])][track_id1]['id']
            for track_id2 in reid_dict[int(seq1[-2:])]:
                if reid_dict[int(seq1[-2:])][track_id2]['id'] == id_temp:
                    time_transit = processed_cam2_entry_zone_data[track_id1]['start_time']-processed_cam1_exit_zone_data[track_id2]['end_time']
                    if time_transit > 0:
                        time_transits.append(time_transit)

    # print(time_transits)
    return time_transits
    
    

if __name__ == "__main__":
    seqs = ['c041', 'c042', 'c043', 'c044', 'c045', 'c046']
    time_window = {}
    abs_path = '/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/detect_merge'
    pair_path = os.path.join(abs_path,'cam_pair.json')
    with open(pair_path,'r') as f:
        cam_pair = json.load(f)
    print(cam_pair)

    for i in range(len(seqs)-1):
        pair = f'{seqs[i]}&{seqs[i+1]}'
        print(f'start processing {seqs[i]} and {seqs[i+1]} pair ---')
        time_transits = _time_window(seqs[i],seqs[i+1],cam_pair)
        cam_pair[seqs[i][-2:]][seqs[i+1][-2:]]['time_pair'] = time_transits
        cam_pair[seqs[i+1][-2:]][seqs[i][-2:]]['time_pair'] = time_transits
        print(f'{seqs[i]} and {seqs[i+1]} pair finished -----')
    

    with open(pair_path,'w') as f:
        json.dump(cam_pair,f)