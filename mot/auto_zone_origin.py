import os 
import math
import json
import numpy as np
import MeanShift_py.mean_shift as ms
import matplotlib.pyplot as plt
import cv2
import sys
sys.path.append('../')
from config import cfg


def main(seq):
    abs_path = '/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/detect_merge'
    zone_data_path = os.path.join(abs_path,seq,f'{seq}_zone_data.json')
    abs_img_path = '/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/detection/images/test/S06/'
    img_save_path = os.path.join(abs_path,seq,'auto_zone_gen.jpg')
    img_path = os.path.join(abs_img_path,seq,'img1/img000000.jpg')
    background_img=cv2.imread(img_path)

    vector_data = []
    bbox_data0 = []
    bbox_data1 = []

    with open(zone_data_path) as f:
        zone_data = json.load(f)

    for track_id,value in zone_data.items():
        # for i in range(len(value['entry_pos'])):
        #     data.append(value['entry_pos'][i])
        #     data.append(value['exit_pos'][i])

        vector_data.append(value['entry_vector'])
        vector_data.append(value['exit_vector'])


    mean_shifter = ms.MeanShift()
    mean_shift_result = mean_shifter.cluster(vector_data, kernel_bandwidth = 25)
    original_points =  mean_shift_result.original_points
    shifted_points = mean_shift_result.shifted_points
    cluster_assignments = mean_shift_result.cluster_ids
    cluster_index = mean_shift_result.cluster_index


    for index,flag in enumerate(cluster_assignments): # assign all points in cluster one to the first data_set
        # zone_data is indexed by it's sequence, with the [entry_pos] and [exit_pos] info
        if flag == 0:
            real_index = int(index/2)
            if (index % 2) == 0: # even number is entry_zone
                for i in range(len(zone_data[f'{real_index}']['entry_pos'])):
                    bbox_data0.append(zone_data[f'{real_index}']['entry_pos'][i])
            else:
                for i in range(len(zone_data[f'{real_index}']['exit_pos'])):
                    bbox_data0.append(zone_data[f'{real_index}']['exit_pos'][i])

    for index,flag in enumerate(cluster_assignments):
        if flag == 1:
            real_index = int(index/2)
            if (index % 2) == 0: # even number is entry_zone
                for i in range(len(zone_data[f'{real_index}']['entry_pos'])):
                    bbox_data1.append(zone_data[f'{real_index}']['entry_pos'][i])
            else:
                for i in range(len(zone_data[f'{real_index}']['exit_pos'])):
                    bbox_data1.append(zone_data[f'{real_index}']['exit_pos'][i])



    bbox_mean_shift_result0 = mean_shifter.cluster(bbox_data0, kernel_bandwidth = 60)
    bbox_original_points0 =  bbox_mean_shift_result0.original_points
    bbox_shifted_points0 = bbox_mean_shift_result0.shifted_points
    bbox_cluster_assignments0 = bbox_mean_shift_result0.cluster_ids
    bbox_cluster_index0 = bbox_mean_shift_result0.cluster_index


    bbox_mean_shift_result1 = mean_shifter.cluster(bbox_data1, kernel_bandwidth = 60)
    bbox_original_points1 =  bbox_mean_shift_result1.original_points
    bbox_shifted_points1 = bbox_mean_shift_result1.shifted_points
    bbox_cluster_assignments1 = bbox_mean_shift_result1.cluster_ids
    bbox_cluster_index1 = bbox_mean_shift_result1.cluster_index
    
    # Results combination----------------------
    
    bbox_original_points = np.concatenate([bbox_original_points0, bbox_original_points1])

    bbox_shifted_points = np.concatenate([bbox_shifted_points0, bbox_shifted_points1])

    bbox_cluster_assignments1 = bbox_cluster_assignments1 + max(bbox_cluster_assignments0) + 1
    bbox_cluster_assignments = np.concatenate([bbox_cluster_assignments0, bbox_cluster_assignments1])

    

#----------------------------------------------------------------------------
    # Prepare to overlay points on the image
    for point, cluster_id in zip(bbox_original_points, bbox_cluster_assignments):
        color = plt.cm.viridis(cluster_id / len(set(bbox_cluster_assignments)))  # Get a color from the colormap
        color = (color[0] * 255, color[1] * 255, color[2] * 255)  # Convert to BGR color space used by OpenCV
        cv2.circle(background_img, (int(point[0]), int(point[1])), 10, color, -1)  # Draw the cluster points

    # Mark cluster centers in red
    for center in bbox_shifted_points:
        cv2.drawMarker(background_img, (int(center[0]), int(center[1])), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=5)

    # Save and show the final image
    cv2.imwrite(img_save_path, background_img)
#----------------------------------------------------------------------------





if __name__ == "__main__":
    seqs = ['c041','c042','c043','c044','c045','c046']
    for seq in seqs:
        print(f'start processing {seq} ---')
        main(seq)
        print(f'camera {seq} finished',seq)



    # x = original_points[:,0]
    # y = original_points[:,1]
    # Cluster = cluster_assignments
    # centers = shifted_points

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # scatter = ax.scatter(x,y,c=Cluster,s=50)
    # for i,j in centers:
    #     ax.scatter(i,j,s=50,c='red',marker='+')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # plt.colorbar(scatter)

    # fig.savefig("mean_shift_result")
