import numpy as np
'''
CAM_DIST = [[  0, 40, 55,100,120,145],
            [ 40,  0, 15, 60, 80,105],
            [ 55, 15,  0, 40, 65, 90],
            [100, 60, 40,  0, 20, 45],
            [120, 80, 65, 20,  0, 25],
            [145,105, 90, 45, 25,  0]]
'''
SPEED_LIMIT = [[(0,0), (400,1300), (550,2000), (1000,2000), (1200, 2000), (1450, 2000)],
               [(400,1300), (0,0), (100,900), (600,2000), (800,2000), (1050,2000)],
               [(550,2000), (100,900), (0,0), (350,1050), (650,2000), (900, 2000)],
               [(1000,2000), (600,2000), (350,1050), (0,0), (150,500), (450, 2000)],
               [(1200, 2000), (800,2000), (650,2000), (150,500), (0,0), (250,900)],
               [(1450, 2000), (1050,2000), (900, 2000), (450, 2000), (250,900), (0,0)]]  

def calc_reid(dismat,q_track_ids,q_cam_ids, g_track_ids, g_cam_ids, q_times, g_times, new_id, dis_thre=0.28,dis_remove=0.3):
    # (dismat,q_track_ids,q_cam_ids, g_track_ids, g_cam_ids, q_times, g_times,new_id)
    # new_id = np.max(g_track_ids)
    rm_dict = {}
    reid_dict = {}
    indices = np.argsort(dismat, axis=1) 
    # print(indices)
    # num_q, num_g = dismat.shape    
    for index, q_track_id in enumerate(q_track_ids):


        q_cam_id = q_cam_ids[index]
        q_time = q_times[index]
        # g_times = np.array([time[0] for time in g_times])
        
        # check is that workable or not 
        order = indices[index] # the real order for the first query 

        speed_limit_min = np.array([SPEED_LIMIT[q_cam_id-41][g_cam_id-41][0]/10 for g_cam_id in g_cam_ids[order]])
        speed_limit_max = np.array([SPEED_LIMIT[q_cam_id-41][g_cam_id-41][1]/10 for g_cam_id in g_cam_ids[order]])
     
        remove = (g_track_ids[order] == q_track_id) | \
                (g_cam_ids[order] == q_cam_id) | \
                (dismat[index][order] > dis_thre) | \
                (g_times[order][:,0] < (q_time[1] + speed_limit_min)) | \
                (g_times[order][:,0] > (q_time[1] + speed_limit_max))

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
            # print(f'cam_id:{sel_g_camids_list[i]}')
            # print(f'track_id:{sel_g_track_list[i]}')
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
  #  calc_dict = calc_length(output) 
  #  donot need to calculate again because all tracks already been filtered ine Dataprocess.py
    for line in output:
        line = line.strip().split(" ")
        cam_id = int(line[0])
        track_id = int(line[1])
# not going to update this track if it is in the rm_dict
        if cam_id in list(rm_dict.keys()):
            if track_id in list(rm_dict[cam_id].keys()):
                continue

        # if calc_dict[cam_id][track_id] < max_length: 
        #     continue
        # if cam_id in list(reid_dict.keys()):
        #     if track_id in list(reid_dict[cam_id].keys()):
        #         line[1] = str(reid_dict[cam_id][track_id]["id"]) # In this case, update the ID
        if cam_id in list(reid_dict.keys()):
            if track_id in list(reid_dict[cam_id].keys()):
                line[1] = str(reid_dict[cam_id][track_id]["id"])
                # for f_info in f:
                #     f = f.strip().split(" ")
                #     f_cam_id = int(f[0])
                #     f_track_id = int(f[1])
                
                f.write(" ".join(line)+' -1 -1'+"\n")

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
