import numpy as np
'''
CAM_DIST = [[  0, 40, 55,100,120,145],
            [ 40,  0, 15, 60, 80,105],
            [ 55, 15,  0, 40, 65, 90],
            [100, 60, 40,  0, 20, 45],
            [120, 80, 65, 20,  0, 25],
            [145,105, 90, 45, 25,  0]]
'''
# The first element is entry_zone, second is exit_zone
# e.g. ZONE_PAIR[0][1] = [0,7] refers to camera c041 entry_zone_0 connect with c042 exit_zone_7
#      ZONE_PAIR[1][0] = [2,14] refers to camera c042 entry_zone_2 connect with c041 exit_zone_14

ZONE_PAIR = [[[],[0,7],[],[],[],[]],
             [[2,14],[],[8,12],[],[],[]],
             [[],[6,3],[],[13,1],[],[]],
             [[],[],[6,5],[],[2,3],[]],
             [[],[],[],[7,[9,10]],[],[4,8]],
             [[],[],[],[],[2,[8,12]],[]]]

'''
ZONE_PAIR = [[[],[0,7],[],[],[],[]],
             [[2,14],[],[8,12],[],[],[]],
             [[],[6,3],[],[13,1],[],[]],
             [[],[],[6,5],[],[2,3],[]],
             [[],[],[],[7,9],[],[4,8]],
             [[],[],[],[],[2,8],[]]]
'''

SPEED_LIMIT = [[(0,0), (400,1300), (550,2000), (1000,2000), (1200, 2000), (1450, 2000)],
               [(400,1300), (0,0), (100,900), (600,2000), (800,2000), (1050,2000)],
               [(550,2000), (100,900), (0,0), (350,1050), (650,2000), (900, 2000)],
               [(1000,2000), (600,2000), (350,1050), (0,0), (150,500), (450, 2000)],
               [(1200, 2000), (800,2000), (650,2000), (150,500), (0,0), (250,900)],
               [(1450, 2000), (1050,2000), (900, 2000), (450, 2000), (250,900), (0,0)]]  


def speed_limit_remove_gen(q_cam_id,g_cam_ids,q_entry_temp,q_exit_temp,q_entry_zone_id,q_exit_zone_id,q_time,g_times,order):
    # q_entry_temp is the entry_zone pair 
    speed_limit_remove = []
    flag_entry, flag_exit, high_confidence_track = False, False, False
    speed_limit_min = np.array([SPEED_LIMIT[q_cam_id-41][g_cam_id-41][0]/10 for g_cam_id in g_cam_ids[order]])
    speed_limit_max = np.array([SPEED_LIMIT[q_cam_id-41][g_cam_id-41][1]/10 for g_cam_id in g_cam_ids[order]])

    if q_entry_zone_id in q_entry_temp:
        # The entry_zone is the connected zone, which means this track might exit from c042 and then show up in c041
        speed_limit_remove =  (g_times[order][:,0] > (q_time[1] - speed_limit_min)) | (g_times[order][:,0] < (q_time[1] - speed_limit_max)) 
        flag_entry = True
    elif q_exit_zone_id in q_exit_temp:
        # the exit zone is the connected zone/
        speed_limit_remove =  (g_times[order][:,0] < (q_time[1] + speed_limit_min)) | (g_times[order][:,0] > (q_time[1] + speed_limit_max)) 
        flag_exit = True
    else:
        # the track are in undefined zone
        speed_limit_remove1 =  (g_times[order][:,0] > (q_time[1] - speed_limit_min)) | (g_times[order][:,0] < (q_time[1] - speed_limit_max)) 
        speed_limit_remove2 =  (g_times[order][:,0] < (q_time[1] + speed_limit_min)) | (g_times[order][:,0] > (q_time[1] + speed_limit_max)) 
        speed_limit_remove = [x and y for x, y in zip(speed_limit_remove1, speed_limit_remove2)] # element-wise compare


    if flag_entry and flag_exit:
        print('Turn happen!!!!!! set speed_limit to empty')
        speed_limit_remove = []
    if flag_entry or flag_exit:
        high_confidence_track = True

    return speed_limit_remove, high_confidence_track

def cam_remove_gen(q_cam_ids,g_cam_ids,q_entry_zone_id,q_entry_zone_cls,q_exit_zone_id,q_exit_zone_cls,g_entry_zones,g_exit_zones,order):
    cam_remove = []

    if type(ZONE_PAIR[q_cam_ids[0]-41][g_cam_ids[0]-41][0]) == int:
        q_entry_temp = [ZONE_PAIR[q_cam_ids[0]-41][g_cam_ids[0]-41][0]]
    else:
        q_entry_temp = ZONE_PAIR[q_cam_ids[0]-41][g_cam_ids[0]-41][0]
    
    if type(ZONE_PAIR[g_cam_ids[0]-41][q_cam_ids[0]-41][1]) == int:
        q_exit_temp = [ZONE_PAIR[g_cam_ids[0]-41][q_cam_ids[0]-41][1]]
    else:
        q_exit_temp = ZONE_PAIR[g_cam_ids[0]-41][q_cam_ids[0]-41][1]

    if (q_entry_zone_id in q_entry_temp) or\
            (q_exit_zone_id in q_exit_temp) or\
                q_entry_zone_cls =='undefined_zone' or q_exit_zone_cls =='undefined_zone':
        for ord in order:
            
            # For data_structure unify
            if type(ZONE_PAIR[g_cam_ids[0]-41][q_cam_ids[0]-41][0]) == int:
                g_entry_temp = [ZONE_PAIR[g_cam_ids[0]-41][q_cam_ids[0]-41][0]]
            else:
                g_entry_temp = ZONE_PAIR[g_cam_ids[0]-41][q_cam_ids[0]-41][0]

            if type(ZONE_PAIR[q_cam_ids[0]-41][g_cam_ids[0]-41][1]) == int:
                g_exit_temp = [ZONE_PAIR[q_cam_ids[0]-41][g_cam_ids[0]-41][1]]
            else:
                g_exit_temp = ZONE_PAIR[q_cam_ids[0]-41][g_cam_ids[0]-41][1]
            # For data_structure unify

            if (g_entry_zones[ord][1] in g_entry_temp) or\
                    (g_exit_zones[ord][1] in g_exit_temp) or\
                        (g_entry_zones[ord][0] == 'undefined_zone') or (g_exit_zones[ord][0] == 'undefined_zone'):
                cam_remove.append(False)
            else:
                cam_remove.append(True)
    else: 
        cam_remove = [True] * len(g_cam_ids)

    return cam_remove,q_entry_temp,q_exit_temp
    


def calc_reid(dismat,q_track_ids,q_cam_ids, g_track_ids, g_cam_ids, q_times, g_times, q_entry_zones, q_exit_zones, g_entry_zones, g_exit_zones, new_id, dis_thre=0.5,dis_remove=0.6):
    # dis_thre=0.47,dis_remove=0.57
    # For Euclidean Distance (0.29,0.34)
    # new_id = np.max(g_track_ids)
    rm_dict = {}
    reid_dict = {}
    indices = np.argsort(dismat, axis=1) 
    # print(indices)
    # num_q, num_g = dismat.shape    
    for index, q_track_id in enumerate(q_track_ids):

        q_cam_id = q_cam_ids[index]
        q_time = q_times[index]
        q_entry_zone_cls = q_entry_zones[index][0]
        q_entry_zone_id = q_entry_zones[index][1]
        q_exit_zone_cls = q_exit_zones[index][0]
        q_exit_zone_id = q_exit_zones[index][1]
        # q_entry_pair = []
        # q_exit_pair = []
        # g_times = np.array([time[0] for time in g_times])
        
        # check is that workable or not 
        order = indices[index] # the real order for the first query 

        # q_entry_pair.append(ZONE_PAIR[q_cam_id-41][g_cam_id-41] for g_cam_id in g_cam_ids[order])
        # q_exit_pair.append(ZONE_PAIR[g_cam_id-41][q_cam_id-41] for g_cam_id in g_cam_ids[order])
        # | is or True | False -> True

        order1=order.copy()

        cam_remove,q_entry_temp,q_exit_temp = cam_remove_gen(q_cam_ids,g_cam_ids,q_entry_zone_id,q_entry_zone_cls,q_exit_zone_id,q_exit_zone_cls,g_entry_zones,g_exit_zones,order1)


        speed_limit_remove,high_conf_track = speed_limit_remove_gen(q_cam_id,g_cam_ids,q_entry_temp,q_exit_temp,q_entry_zone_id,q_exit_zone_id,q_time,g_times,order1)

        if high_conf_track:
            dis_thre=0.5
            dis_remove=0.6
        else:
            dis_thre=0.3
            dis_remove=0.4
        
        remove = (g_track_ids[order] == q_track_id) | \
                (g_cam_ids[order] == q_cam_id) | \
                (dismat[index][order] > dis_thre) | \
                (speed_limit_remove) | (cam_remove)

        # remove all track g_time < q_time + min_time and g_time > q_time + max_time
        keep = np.invert(remove)
        #print(keep)
        # 是在remove query list，所以是很有意义的！
        remove_hard = (g_track_ids[order] == q_track_id) | \
                      (g_cam_ids[order] == q_cam_id) | \
                      (dismat[index][order]>dis_remove)
        # print(remove_hard)
        keep_hard = np.invert(remove_hard)
        if True not in keep_hard: # nothing is been kept,所有的track都匹配不上
            print('detected!!!!!')
            if q_cam_id not in list(rm_dict.keys()):
                rm_dict[q_cam_id] = {}
            rm_dict[q_cam_id][q_track_id] = True       

        sel_g_dis = dismat[index][order][keep]
        sel_g_track_ids = g_track_ids[order][keep]
        sel_g_cam_ids = g_cam_ids[order][keep]
        sel_g_track_list = []
        sel_g_camids_list = []
        selg_dis_list = []


        for i in range(sel_g_track_ids.shape[0]): # should be the length of sel_g_ids, it should be 1xlen(shape), in the case it only has one dimension so would be shape
            sel_id = sel_g_track_ids[i]
            sel_cam = sel_g_cam_ids[i]
            sel_dis = sel_g_dis[i]

            sel_g_track_list.append(sel_id)
            sel_g_camids_list.append(sel_cam)
            selg_dis_list.append(sel_dis)
       # print(sel_g_track_ids)

        if len(selg_dis_list) > 0:
            new_id+=1
            if q_cam_id in list(reid_dict.keys()):
                if q_track_id in list(reid_dict[q_cam_id]): # second time for the matching! 
                    if reid_dict[q_cam_id][q_track_id]["dis"]>min(selg_dis_list):
                        reid_dict[q_cam_id][q_track_id]["dis"] = min(selg_dis_list)
                        reid_dict[q_cam_id][q_track_id]["id"] = new_id
                else:
                    reid_dict[q_cam_id][q_track_id] = {"dis":min(selg_dis_list),"id":new_id}
            else:
                # that is the initalization part
                reid_dict[q_cam_id] = {}
                reid_dict[q_cam_id][q_track_id] = {"dis":min(selg_dis_list),"id":new_id} # assigining a new id top this track!!!!!!
        
        # this is the second loop
        for i in range(len(sel_g_track_list)):
            if sel_g_camids_list[i] in list(reid_dict.keys()):
                if sel_g_track_list[i] in list(reid_dict[sel_g_camids_list[i]]): 
                    if reid_dict[sel_g_camids_list[i]][sel_g_track_list[i]]["dis"]>selg_dis_list[i]:
                        reid_dict[sel_g_camids_list[i]][sel_g_track_list[i]]["dis"] = selg_dis_list[i]
                        reid_dict[sel_g_camids_list[i]][sel_g_track_list[i]]["id"] = new_id
                else:
                       reid_dict[sel_g_camids_list[i]][sel_g_track_list[i]] = {"dis":selg_dis_list[i],"id":new_id}
            else:
                reid_dict[sel_g_camids_list[i]] = {}
                reid_dict[sel_g_camids_list[i]][sel_g_track_list[i]] = {"dis":selg_dis_list[i],"id":new_id}
        
    return reid_dict,rm_dict

def reid_cat_original(reid_dict1,reid_dict2):
    overlap = {}
    # This for loop select the common cam track_id, and update one of  
    for cam_id in reid_dict2:
        if cam_id in reid_dict1:
            overlap[cam_id] = {}
    for cam_id in list(reid_dict2.keys()):
        if cam_id in list(reid_dict1.keys()):
            if overlap[cam_id] not in list(overlap.keys()):
                overlap[cam_id] = {}
            for track_id in list(reid_dict2[cam_id].keys()):
                if track_id in list(reid_dict1[cam_id].keys()):
                    overlap[cam_id][track_id] = reid_dict1[cam_id][track_id]['id'] # The 'id' of the reid_dict2 is defintely bigger than the first one

    #-----------------------update reid_dict2---------------------
    cam_id1, cam_id2 =list(reid_dict2.keys()) # defalut cam_id1 is the overlap cam!!!
    for track_id1 in list(reid_dict2[cam_id1].keys()):
        for track_id2 in list(reid_dict2[cam_id2].keys()):
            if reid_dict2[cam_id1][track_id1]['id'] == reid_dict2[cam_id2][track_id2]['id']:
                reid_dict2[cam_id1][track_id1]['id'] = overlap[cam_id1][track_id1]
                reid_dict2[cam_id2][track_id2]['id'] = overlap[cam_id1][track_id1]
                # break
    # ------------------cat the two dict-----------
    reid_dict1[cam_id2] = {}
    for track_id2 in list(reid_dict2[cam_id2].keys()):
        reid_dict1[cam_id2][track_id2] = reid_dict2[cam_id2][track_id2]

    return reid_dict1
                    # reid_dict2[cam_id][track_id]['id'] = reid_dict1[cam_id][track_id]['id']


def reid_cat(reid_dict1, reid_dict2):
    overlap = {}
    # Initialize overlap for cameras in both reid_dict1 and reid_dict2
    for cam_id in reid_dict2:
        if cam_id in reid_dict1:
            overlap[cam_id] = {}

    # Populate the overlap dictionary with the 'id' from reid_dict1 for overlapping track_ids
    for cam_id in overlap:
        for track_id in reid_dict2[cam_id]:
            if track_id in reid_dict1[cam_id]:
                # Assuming the 'id' from reid_dict1 should be used for overlaps
                overlap[cam_id][track_id] = reid_dict1[cam_id][track_id]['id']
                

    # Update reid_dict2 based on overlap information
    cam_id1, cam_id2 =list(reid_dict2.keys()) # defalut cam_id1 is the overlap cam!!!
    for track_id1 in list(reid_dict2[cam_id1].keys()):
        if track_id1 in overlap[cam_id1]:
            for track_id2 in list(reid_dict2[cam_id2].keys()):
                if reid_dict2[cam_id1][track_id1]['id'] == reid_dict2[cam_id2][track_id2]['id']:
                    reid_dict2[cam_id1][track_id1]['id'] = overlap[cam_id1][track_id1]
                    reid_dict2[cam_id2][track_id2]['id'] = overlap[cam_id1][track_id1]

    # Merge reid_dict2 into reid_dict1
    for cam_id in reid_dict2:
        if cam_id not in reid_dict1:
            reid_dict1[cam_id] = reid_dict2[cam_id]
        else:
            reid_dict1[cam_id].update(reid_dict2[cam_id]) # need to update because the 41&42 pair里的42的pair不完全等于42&43 pair里42的track

    return reid_dict1



def calc_length(output):
    calc_dict = {}
    for line in output:
        line = line.strip().split(" ")
        cam_id = int(line[0])
        track_id = int(line[1])
        if cam_id not in list(calc_dict.keys()):
            calc_dict[cam_id] = {}
        if track_id not in list(calc_dict[cam_id].keys()):
            calc_dict[cam_id][track_id] = 1
        else:
            calc_dict[cam_id][track_id]+=1
    return calc_dict

def update_output(output,reid_dict,rm_dict,f,max_length=20):
    calc_dict = calc_length(output) 
    i = 0 
  #  donot need to calculate again because all tracks already been filtered ine Dataprocess.py
    for line in output:
        line = line.strip().split(" ")
        cam_id = int(line[0])
        track_id = int(line[1])
# not going to update this track if it is in the rm_dict
        if cam_id in list(rm_dict.keys()):
            if track_id in list(rm_dict[cam_id].keys()):
                print(f' For general conunt on {i}', end = '\r')
                i+=1
                continue

        if calc_dict[cam_id][track_id] < max_length: 
            continue
        # if cam_id in list(reid_dict.keys()):
        #     if track_id in list(reid_dict[cam_id].keys()):
        #         line[1] = str(reid_dict[cam_id][track_id]["id"]) # In this case, update the ID
        # 修改dis_remove变化不大的原因应该是因为被hard remove掉的query set本身也不在reid_dict 里面
        if cam_id in list(reid_dict.keys()):
            if track_id in list(reid_dict[cam_id].keys()):
                line[1] = str(reid_dict[cam_id][track_id]["id"])
                # for f_info in f:
                #     f = f.strip().split(" ")
                #     f_cam_id = int(f[0])
                #     f_track_id = int(f[1])
                
                f.write(" ".join(line)+' -1 -1'+"\n")
        
    f.close()

def reid_dict_filter(reid_dict):
    track_counter = {}
    rm_id_list = []
    for cam_id in reid_dict:
        for track_id in reid_dict[cam_id]:
            if reid_dict[cam_id][track_id]['id'] not in track_counter:
                temp_flag = reid_dict[cam_id][track_id]['id']
                track_counter[temp_flag] = 1
            else:
                track_counter[reid_dict[cam_id][track_id]['id']] += 1

    for id in track_counter:
        if track_counter[id] == 1:
            rm_id_list.append(id)
    
    for cam_id in list(reid_dict.keys()):
        for track_id in list(reid_dict[cam_id].keys()):
            if reid_dict[cam_id][track_id]['id'] in rm_id_list:
                del reid_dict[cam_id][track_id]

    return reid_dict

def xytoxywh(input_path,output_path):
    with open(input_path, "r") as f:
        tracks = f.readlines()
    
    # Open the new file where the converted tracklets will be written
    with open(output_path, "w") as f_out:
        for track in tracks: 
            elements = track.split()
            # Extract the bounding box coordinates and convert them
            x1, y1, x2, y2 = map(int, elements[3:7])
            w = x2 - x1
            h = y2 - y1
            # Extract other elements
            cam_id, track_id, frame_num = map(int, elements[0:3])
            # Write the modified line to the new file
            output_line = f"{cam_id} {track_id} {frame_num} {x1} {y1} {w} {h} -1 -1\n"
            f_out.write(output_line)

def original_calc_reid(distmat,q_pids,g_pids,q_camids,g_camids,dis_thre=0.8,dis_remove=0.8):
    #?dis_remove is the hard remove, dis_thre is considering remove?d
    # q_pids is the query_track_id, g_pids is the gallery_track_id

    new_id = np.max(g_pids) # why???
    # print(np.max(g_pids))
    # print(np.max(q_pids))
    rm_dict = {}
    reid_dict = {}
    indices = np.argsort(distmat, axis=1)
    num_q, num_g = distmat.shape
    # print(np.min(distmat))
    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx] 
        order = indices[q_idx] # the order for the first query vs all gallery items
        remove = (g_pids[order] == q_pid) | (g_camids[order] == q_camid) | (distmat[q_idx][order]>dis_thre) 
        # | it means and. e.g. True | Flase returns True
        # return a bool array
        # Like [True False True False]
        keep = np.invert(remove) # bit-wise inversion

        remove_hard = (g_pids[order] == q_pid) | (g_camids[order] == q_camid) | (distmat[q_idx][order]>dis_remove)
        keep_hard = np.invert(remove_hard)
        if True not in keep_hard: # nothing is been kept
            if q_camid not in list(rm_dict.keys()):
                rm_dict[q_camid] = {}
            rm_dict[q_camid][q_pid] = True
        sel_g_dis = distmat[q_idx][order][keep] # selected_gallery_distance_matrix
        sel_g_pids = g_pids[order][keep]
        sel_g_camids = g_camids[order][keep]
        sel_g_pids_list = []
        sel_g_camids_list = []
        selg_dis_list = []
        

        for i in range(sel_g_pids.shape[0]): # should be the length of sel_g_ids, it should be 1xlen(shape), in the case it only has one dimension so would be shape
            sel_pid =  sel_g_pids[i]
            sel_cam = sel_g_camids[i]
            sel_dis = sel_g_dis[i]
            if sel_cam not in sel_g_camids_list and sel_cam!=q_camid:
                sel_g_pids_list.append(sel_pid)
                sel_g_camids_list.append(sel_cam)
                selg_dis_list.append(sel_dis)
                
    #-------------------------------initalize the reid_dict()--------------------------
        if len(selg_dis_list)>0:
            # camera is the first layer and query id is the second
            new_id+=1 # assgining new id xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
            if q_camid in list(reid_dict.keys()): # initalize [cam_id] in the dict reid_dict 
                if q_pid in list(reid_dict[q_camid]): # is matched before
                    if reid_dict[q_camid][q_pid]["dis"]>min(selg_dis_list):
                        reid_dict[q_camid][q_pid]["dis"] = min(selg_dis_list)
                        reid_dict[q_camid][q_pid]["id"] = new_id # give the query part a new id
                else:
                    reid_dict[q_camid][q_pid] = {"dis":min(selg_dis_list),"id":new_id}
            else:
                reid_dict[q_camid] = {}
                reid_dict[q_camid][q_pid] = {"dis":min(selg_dis_list),"id":new_id}#  'distance' and 'id' in the dict() 

    #-----------------------update the reid_dict()-----------------------------------

        for i in range(len(sel_g_pids_list)):
            if sel_g_camids_list[i] in list(reid_dict.keys()):
                if sel_g_pids_list[i] in list(reid_dict[sel_g_camids_list[i]]):
                    if reid_dict[sel_g_camids_list[i]][sel_g_pids_list[i]]["dis"]>selg_dis_list[i]:
                        reid_dict[sel_g_camids_list[i]][sel_g_pids_list[i]]["dis"] = selg_dis_list[i]
                        reid_dict[sel_g_camids_list[i]][sel_g_pids_list[i]]["id"] = new_id
                else:
                    reid_dict[sel_g_camids_list[i]][sel_g_pids_list[i]] = {"dis":selg_dis_list[i],"id":new_id}
            else:
                reid_dict[sel_g_camids_list[i]] = {}
                reid_dict[sel_g_camids_list[i]][sel_g_pids_list[i]] = {"dis":selg_dis_list[i],"id":new_id}


    total_distance = 0
    num = 0
    ave_distance = 0
    for cam_id in reid_dict:
        num += len(reid_dict[cam_id].keys())
        for track_id in reid_dict[cam_id]:
            total_distance += reid_dict[cam_id][track_id]['dis']
    if num != 0:
        ave_distance = total_distance / num
    else:
        ave_distance = 100000000000

    return reid_dict, rm_dict, ave_distance, num