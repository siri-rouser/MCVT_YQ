import os
import json
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
import MeanShift_py.mean_shift as ms
import MeanShift_py.mean_shift_utils as ms_utils
from zone_class import ZONE

def point_in_rect(point,area):
    min_x,min_y,max_x,max_y = area
    if min_x < point[0] < max_x and min_y < point[1] < max_y:
        return True
    else:
        return False
    
def rectangles_overlap(rect1, rect2):

    min_x1, min_y1, max_x1, max_y1 = rect1
    min_x2, min_y2, max_x2, max_y2 = rect2
    
    # Check if there's a horizontal and vertical overlap
    horizontal_overlap = (min_x1 <= max_x2) and (min_x2 <= max_x1)
    vertical_overlap = (min_y1 <= max_y2) and (min_y2 <= max_y1)
    
    return horizontal_overlap and vertical_overlap

def main(seq):

#------------For data reading and initalization---------------------------
    abs_path = '/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/detect_merge'
    zone_data_path = os.path.join(abs_path, seq, f'{seq}_zone_data.json')
    track_data_path = os.path.join(abs_path,seq,f'{seq}_tracklet.pkl')
    new_track_path = os.path.join(abs_path,seq,f'{seq}_new_tracklet.pkl')
    abs_img_path = '/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/detection/images/test/S06/'
    img_save_path = os.path.join(abs_path, seq, 'auto_zone_gen.jpg')
    img_save_path1 = os.path.join(abs_path, seq, 'auto_zone_draw.jpg')
    exit_zone_img_path = os.path.join(abs_path, seq, 'exit_zone_draw.jpg')
    entry_zone_img_path = os.path.join(abs_path, seq, 'entry_zone_draw.jpg')
    img_path = os.path.join(abs_img_path, seq, 'img1/img000000.jpg')
    background_img = cv2.imread(img_path)

    vector_data = []
    entry_pos = []
    exit_pos = []
    Zone = {}

    with open(zone_data_path) as f:
        zone_data = json.load(f)
    
    with open(track_data_path,'rb') as f1:
        track_data = pickle.load(f1)

    for track_id, value in zone_data.items():
        vector_data.append(value['entry_vector'])
        vector_data.append(value['exit_vector'])
#------------END---------------------------


#------------For Vector Clustering---------------------------
    mean_shifter = ms.MeanShift()
    mean_shift_result = mean_shifter.cluster(vector_data, kernel_bandwidth=25)
    cluster_assignments = mean_shift_result.cluster_ids

    for track_id, value in zone_data.items():
        for i in range(len(value['entry_pos'])):
            entry_pos.append(value['entry_pos'][i])
            exit_pos.append(value['exit_pos'][i])

    # Prepare a dictionary to hold data for each cluster
    bbox_data = {i: [] for i in range(max(cluster_assignments) + 1)}

    for index, cluster_id in enumerate(cluster_assignments):
        real_index = int(index / 2)
        position_key = 'entry_pos' if index % 2 == 0 else 'exit_pos'
        bbox_data[cluster_id].extend(zone_data[f'{real_index}'][position_key])

#------------END---------------------------
    # Process each cluster's data
    bbox_results = {}
    total_clusters = 0
    for cluster_id, data in bbox_data.items():
        if data:
            result = mean_shifter.cam_cluster(data, kernel_bandwidth=90, img_height = background_img.shape[0])
            # result = mean_shifter.cluster(data, kernel_bandwidth=80)
            bbox_results[cluster_id] = result
            total_clusters += len(set(result.cluster_ids))
        max_cluster_id = cluster_id # set in here for avoid more iteration on zone cal

    # Prepare to overlay points on the image
    cat_original_points = []
    cat_shift_points = []
    areadraw_img = background_img.copy()
    entry_zone_img = background_img.copy()
    exit_zone_img = background_img.copy()
    
    for cluster_id, result in bbox_results.items():
        # it returns the result for certain vector cluster
        color = []
        original_points = result.original_points
        shifted_points = result.shifted_points
        assignments = result.cluster_ids
        cluster_idex = result.cluster_index + 1

        # for result saving
        cat_original_points.extend(original_points)
        cat_shift_points.extend(shifted_points)
        if cluster_id == 0:
            new_assignments = assignments
            assign_flag = max(assignments)+1
        else:
            new_assignments = assignments+assign_flag
            assign_flag = max(new_assignments)+1


        for i in range(max(new_assignments)+1):
            color.append(list(np.random.random(size=3) * 256))


        for point, assign in zip(original_points, new_assignments):
            if assign not in Zone:
                Zone[assign] = ZONE(list(point),assign)
            else:
                Zone[assign].zone_append(list(point))
      
            cv2.circle(background_img, (int(point[0]), int(point[1])), 10, color[assign], -1)


        print('error: point and assign matrix not matched') if len(original_points) != len(assignments) else None
        if cluster_id == max_cluster_id:
            for assign in Zone:
                if len(Zone[assign].points) < 3:
                    Zone[assign] = None # be careful about this
                    continue
                Zone[assign].zone_classify(entry_pos,exit_pos)
                Zone[assign].area_define()
                # if Zone[assign].zone_cls != 'undefined_zone':

            for assign1 in Zone:
                for assign2 in Zone:
                    if assign1 != assign2:
                        if (Zone[assign1] is not None) and (Zone[assign2] is not None):
                            if Zone[assign1].zone_cls == Zone[assign2].zone_cls and Zone[assign2].zone_cls != 'undefined_zone':
                                if rectangles_overlap(Zone[assign1].rect_area,Zone[assign2].rect_area):
                                    Zone[assign1].zone_merge(Zone[assign2])
                                    Zone[assign2] = None
                                    Zone[assign1].area_define()
                                    print('yes')
                                    continue

            for assign in Zone:
                if Zone[assign] is not None:
                    areadraw_img = Zone[assign].area_drawing(areadraw_img)
                    entry_zone_img = Zone[assign].required_area_drawing(entry_zone_img,'entry_zone')
                    exit_zone_img = Zone[assign].required_area_drawing(exit_zone_img,'exit_zone')

                    # if Zone[assign].zone_cls == 'entry_zone':
                    #     areadraw_img = Zone[assign].area_drawing(areadraw_img)

        # Mark cluster centers in red
        for center in shifted_points:
            cv2.drawMarker(background_img, (int(center[0]), int(center[1])), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=5)


# ------------------The rest part is for data writing---------------------
    for key,value in track_data.items():
        value['entry_zone_id'], value['entry_zone_cls'], value['exit_zone_id'], value['exit_zone_cls'] = [],[],[],[]
        for assign in Zone:
            if Zone[assign] is not None:
                if point_in_rect(value['entry_pos'][0],Zone[assign].rect_area):
                    if Zone[assign].zone_cls != 'exit_zone':
                        if value['entry_zone_cls'] == 'entry_zone' and Zone[assign].zone_cls == 'undefined_zone':
                            continue
                        if value['entry_zone_cls'] == 'entry_zone' and Zone[assign].zone_cls == 'entry_zone':
                            print('two entry zone overlapped, pls check through')
                            print(f"previous zone_id: {value['entry_zone_id']} new zone_id: {Zone[assign].zone_id}")
                            continue
                        value['entry_zone_id'] = Zone[assign].zone_id
                        value['entry_zone_cls'] = Zone[assign].zone_cls
                if point_in_rect(value['exit_pos'][0],Zone[assign].rect_area):
                    if Zone[assign].zone_cls != 'entry_zone':
                        if value['exit_zone_cls'] == 'exit_zone' and Zone[assign].zone_cls == 'undefined_zone':
                            continue
                        if value['exit_zone_cls'] == 'exit_zone' and Zone[assign].zone_cls == 'exit_zone':
                            print('two exit zone overlapped, pls check through')
                            print(f"previous zone_id: {value['exit_zone_id']} new zone_id: {Zone[assign].zone_id}")
                            continue
                        value['exit_zone_id'] = Zone[assign].zone_id
                        value['exit_zone_cls'] = Zone[assign].zone_cls


    # Save and show the final image
    cv2.imwrite(img_save_path, background_img)
    cv2.imwrite(img_save_path1, areadraw_img)
    cv2.imwrite(entry_zone_img_path, entry_zone_img)
    cv2.imwrite(exit_zone_img_path, exit_zone_img)
    pickle.dump(track_data, open(new_track_path, 'wb'), pickle.HIGHEST_PROTOCOL)

# -------------------------- END -------------------------------------

if __name__ == "__main__":
    seqs = ['c041', 'c042', 'c043', 'c044', 'c045', 'c046']
    # seqs = ['c041'] 
    for seq in seqs:
        print(f'Start processing {seq} ---')
        main(seq)
        print(f'Camera {seq} finished')
