import pickle
import json
import os
import numpy as np
import sys
sys.path.append('../matching/')
from CostMatrix import CostMatrix_Zone
from util_tools import original_calc_reid

GroundTurth_pair = {41:{42:{'entry_exit_pair':[0,7]}},
                    42:{41:{'entry_exit_pair':[2,14]},43:{'entry_exit_pair':[8,12]}},
                    43:{42:{'entry_exit_pair':[6,3]},44:{'entry_exit_pair':[13,1]}},
                    44:{43:{'entry_exit_pair':[6,5]},45:{'entry_exit_pair':[2,3]}},
                    45:{44:{'entry_exit_pair':[[7,9],[7,10]]},46:{'entry_exit_pair':[4,8]}},
                    46:{45:{'entry_exit_pair':[[2,8],[2,12]]}}}

def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {convert_numpy(key): convert_numpy(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(elem) for elem in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy(elem) for elem in obj)
    else:
        return obj


def pair_res_filter(pair_res):
    for q_cam in pair_res:
        if len(pair_res[q_cam].keys()) > 1: 
            g_cams = list(pair_res[q_cam].keys())
            if pair_res[q_cam][g_cams[0]]['top']['pair'][0] == pair_res[q_cam][g_cams[1]]['top']['pair'][0]:
                if pair_res[q_cam][g_cams[0]]['top']['score'] > pair_res[q_cam][g_cams[1]]['top']['score']:
                    pair_res[q_cam][g_cams[1]]['top'] = pair_res[q_cam][g_cams[1]]['second_top']
                else:
                    pair_res[q_cam][g_cams[0]]['top'] = pair_res[q_cam][g_cams[0]]['second_top']

            
    for q_cam in pair_res:
        for g_cam in pair_res[q_cam]:
            pair_res[q_cam][g_cam] = {'score':pair_res[q_cam][g_cam]['top']['score'],'entry_exit_pair':pair_res[q_cam][g_cam]['top']['pair']}

    return pair_res



def compare_entry_exit_pairs(generated_dict, ground_truth_dict):
    total_pairs = 0
    correct_pairs = 0
    matched_pairs = []
    unmatched_pairs = []
    
    for key, nested_dict in ground_truth_dict.items():
        for nested_key, details in nested_dict.items():
            total_pairs += 1
            ground_truth_pair = details['entry_exit_pair']
            
            # Check if the pair exists in the generated dictionary
            if key in generated_dict and nested_key in generated_dict[key]:
                generated_pair = generated_dict[key][nested_key]['entry_exit_pair']
                
                # Check for direct match or match within a list of possible correct pairs
                if isinstance(ground_truth_pair[0], list):
                    # If ground truth pair is a list of lists, check if generated pair matches any
                    if generated_pair in ground_truth_pair:
                        correct_pairs += 1
                        matched_pairs.append(((key, nested_key), generated_pair))
                    else:
                        unmatched_pairs.append(((key, nested_key), generated_pair))
                else:
                    # Direct comparison for a single correct pair
                    if generated_pair == ground_truth_pair:
                        correct_pairs += 1
                        matched_pairs.append(((key, nested_key), generated_pair))
                    else:
                        unmatched_pairs.append(((key, nested_key), generated_pair))
            else:
                unmatched_pairs.append(((key, nested_key), 'Not found in generated data'))
    
    pair_rate = correct_pairs / total_pairs if total_pairs > 0 else 0
    return {
        'pair_rate': pair_rate,
        'total_pairs': total_pairs,
        'correct_pairs': correct_pairs,
        'matched_pairs': matched_pairs,
        'unmatched_pairs': unmatched_pairs
    }

def pair_score_cal(ave_distance,pair_num):
    pair_num = pair_num/100
    score = -ave_distance*0.7 + pair_num*0.3

    return score

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

    # print(f'we have {len(entry_zone_data.keys())} entry zone in cam {data_path[-21:-17]}')
    # print(f'we have {len(exit_zone_data.keys())} exit zone in cam {data_path[-21:-17]}')
    return entry_zone_data,exit_zone_data    

def zone_pair(entry_zone_data,exit_zone_data):

    pair_res_temp = {}
    for entry_zone_id in entry_zone_data:
        for exit_zone_id in exit_zone_data:
            cmz = CostMatrix_Zone(entry_zone_data[entry_zone_id],exit_zone_data[exit_zone_id])
            distmat, q_track_ids, q_cam_ids, g_track_ids, g_cam_ids, q_times, g_times, q_entry_zone,q_exit_zone,g_entry_zone,g_exit_zone = cmz.cost_matrix_zone('Cosine_Distance')
            _, _,ave_distance, pair_num = original_calc_reid(distmat,q_track_ids,g_track_ids,q_cam_ids,g_cam_ids,dis_thre=0.5,dis_remove=0.6)

            print('pair_num:',pair_num)
            print('ave_distance:',ave_distance)
            print(q_cam_ids[0],entry_zone_id,g_cam_ids[0],exit_zone_id)
            

            score = pair_score_cal(ave_distance,pair_num)

            if q_cam_ids[0] not in pair_res_temp:
                pair_res_temp[q_cam_ids[0]] = {}
            if g_cam_ids[0] not in pair_res_temp[q_cam_ids[0]]:
                pair_res_temp[q_cam_ids[0]][g_cam_ids[0]] = {'top': {'score': -float('inf'), 'pair': None}, 'second_top': {'score': -float('inf'), 'pair': None}}

            # Update the top and second top scores
            current_top = pair_res_temp[q_cam_ids[0]][g_cam_ids[0]]['top']
            current_second_top = pair_res_temp[q_cam_ids[0]][g_cam_ids[0]]['second_top']

            if score > current_top['score']:
                # Move current top to second top if new score is higher
                pair_res_temp[q_cam_ids[0]][g_cam_ids[0]]['second_top'] = current_top
                pair_res_temp[q_cam_ids[0]][g_cam_ids[0]]['top'] = {'score': score, 'pair': [entry_zone_id, exit_zone_id]}
            elif score > current_second_top['score']:
                # Update second top if new score is second highest
                pair_res_temp[q_cam_ids[0]][g_cam_ids[0]]['second_top'] = {'score': score, 'pair': [entry_zone_id, exit_zone_id]}

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

    abs_path = '/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/detect_merge'
    cam_pair_save_path = os.path.join(abs_path,'cam_pair.json')
    # for seq in seqs:
    #     cam_path = os.path.join(abs_path,seq,f'{seq}_new_tracklet.pkl')
    #     data_loading(cam_path)

    for i in range(len(seqs)-1):
        pair = f'{seqs[i]}&{seqs[i+1]}'
        print(f'start processing {seqs[i]} and {seqs[i+1]} pair ---')
        pair_res = main(seqs[i],seqs[i+1],pair_res)
        print(f'{seqs[i]} and {seqs[i+1]} pair finished -----')

    print(pair_res)
    pair_res = pair_res_filter(pair_res)
    print(pair_res)
    pair_res_converted = convert_numpy(pair_res)
    with open(cam_pair_save_path,'w') as f:
        json.dump(pair_res_converted,f)
    res = compare_entry_exit_pairs(pair_res, GroundTurth_pair)
    print(res)
