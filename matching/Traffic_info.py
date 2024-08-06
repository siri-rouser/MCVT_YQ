import os
import numpy as np
import pickle
import cv2
import sys
sys.path.append('../')
from config import cfg


def get_zone(bbox,zone_img):
    cx = int((bbox[0] + bbox[2]) / 2) # x1+x2/2
    cy = int((bbox[1] + bbox[3]) / 2) # y1+y2/2
    # pix = self.zones[self.current_cam][cy, cx, :]
    pix = zone_img[cy,cx,:]
    zone_num = 0
    if pix[0] > 50 and pix[1] > 50 and pix[2] > 50:  # w
        zone_num = 1
    if pix[0] < 50 and pix[1] < 50 and pix[2] > 50:  # r
        zone_num = 2
    if pix[0] < 50 and pix[1] > 50 and pix[2] < 50:  # g
        zone_num = 3
    if pix[0] > 50 and pix[1] < 50 and pix[2] < 50:  # b
        zone_num = 4
    return zone_num



def main(seq):
    abs_path = '/home/yuqiang/yl4300/project/MCVT_YQ/matching/zone'
    abs_path1 = '/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/detect_merge'
    abs_path2 = '/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/Traffic_simulation'
    mot_path = os.path.join(abs_path1, seq, f'{seq}_mot.txt')

    zone_image = cv2.imread(os.path.join(abs_path,f'{seq}.png'))
    traffic_data_path = os.path.join(abs_path2,f'{seq}_traffic_info.txt')
    tracklets ={}


    with open(mot_path, 'r') as f:
        for line in f:
            elements = line.split()
            frame_num = float(elements[0])
            track_id = float(elements[1])
            x1, y1, x2, y2 = map(float, elements[2:6])
            bbox = (x1,y1,x2,y2)
            class_id = float(elements[6])
            zone_id = get_zone(bbox,zone_image)
            
            # Data reformating to dict{}
            if track_id not in tracklets:
                tracklets[track_id] = {}
                tracklets[track_id]['bbox'] = []
                tracklets[track_id]['time'] = []
                tracklets[track_id]['zone_id'] = []
                tracklets[track_id]['class_id'] = class_id
            tracklets[track_id]['bbox'].append(bbox)
            tracklets[track_id]['time'].append(frame_num)        
            tracklets[track_id]['zone_id'].append(zone_id)
    f.close()

    for track_id, value in tracklets.items():
        duration = tracklets[track_id]['time'][-1] - tracklets[track_id]['time'][0]
        fragile_flag = tracklets[track_id]['zone_id'][0] == tracklets[track_id]['zone_id'][-1]
        for idx in range(len(value['zone_id'])):
            if tracklets[track_id]['zone_id'][idx] == 0:
                index0 = idx
                break
    
        for index in range(len(value['zone_id'])):
            if fragile_flag:
                value['zone_id'][index] = str(value['zone_id'][index]) + 'e'  # e stands for entry
            else:
                if (value['zone_id'][index] == value['zone_id'][0] or (str(value['zone_id'][index]) + 'e') == str(value['zone_id'][0])) and value['zone_id'][index] != 0: # for fragile but reconnected tracklets
                    value['zone_id'][index] = str(value['zone_id'][index]) + 'e'
                elif (value['zone_id'][index] == value['zone_id'][-1] or (str(value['zone_id'][index]) + 'x')== str(value['zone_id'][-1]))  and value['zone_id'][index] != 0:
                    value['zone_id'][index] = str(value['zone_id'][index]) + 'x'  # x stands for exit
                elif value['zone_id'][index] == 0:
                    value['zone_id'][index] = value['zone_id'][index]
                else:
                    if index < index0:
                        value['zone_id'][index] = str(value['zone_id'][index]) + 'e'
                    else:
                        value['zone_id'][index] = str(value['zone_id'][index]) + 'x'



    with open(traffic_data_path, 'w') as result:
        for frame_num in range(2001):
            frame_num +=1
            for track_id, value in tracklets.items():
                for index, track_frame in enumerate(tracklets[track_id]['time']):
                    if track_frame == frame_num:
                        outputline = f"{frame_num} {tracklets[track_id]['class_id']} {track_id} {tracklets[track_id]['bbox'][index]} {tracklets[track_id]['zone_id'][index]}\n"
                        result.write(outputline)


            




if __name__ == "__main__":
    seqs = ['c041', 'c042', 'c043', 'c044', 'c045', 'c046']
    # seqs = ['c041']

    for seq in seqs:
        print(f'Starting processing camera {seq}')
        main(seq)
        print(f'Camera {seq} finished')

