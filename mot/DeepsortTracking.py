import os
import json
import Modified_Deepsort
import random
import cv2
import pickle
import time
import sys
sys.path.append('../')
from config import cfg


def main(seq):
    start_time = time.time()
    abs_path = '/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/detect_merge'
    source_abs_path = '/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/detection/images/test/S06'
    path_det = os.path.join(abs_path, seq, 'labels')
    path_feat = os.path.join(abs_path, seq, 'feature')
    path_source = os.path.join(source_abs_path, seq, 'img1')

    sorted_feat_files = sorted(os.listdir(path_feat))
    sorted_det_files = sorted(os.listdir(path_det))
    sorted_source_files = sorted(os.listdir(path_source))
    
    save_path = os.path.join(abs_path, seq, f'{seq}_mot.txt')
    tracker = Modified_Deepsort.Modified_Tracker()

    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(20)]

    video_out_path = os.path.join(abs_path, seq, 'out1.mp4')
    frame_size = (1280, 960) if seq in ['c041', 'c042', 'c043', 'c044'] else (1280, 720)
    cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, frame_size)
    
    feat_pkl_file = os.path.join(abs_path, seq, f'{seq}_mot_feat_new.pkl')
    mot_feat_data = {}

    with open(save_path, 'w') as final_result:
        for frame_number, frame_img in enumerate(sorted_source_files):
            frame_number += 1
            frame_img_path = os.path.join(path_source, frame_img)
            det_file_path = os.path.join(path_det, frame_img.replace('.jpg', '.txt'))
            feat_file_path = os.path.join(path_feat, frame_img.replace('.jpg', '.json'))
            
            frame_img = cv2.imread(frame_img_path)

            bbox = []
            confidence = []
            feats = []
            
            if os.path.exists(det_file_path):
                with open(det_file_path, 'r') as f:
                    for line in f:
                        elements = line.split()
                        x_central, y_central, w, h = map(float, elements[1:5])
                        x = x_central - w / 2
                        y = y_central - h / 2
                        bbox.append((x, y, w, h))
                        confidence.append(float(elements[5]))

            if os.path.exists(feat_file_path):
                with open(feat_file_path, 'r') as f:
                    feature_data = json.load(f)
                    for value in feature_data.values():
                        feats.append(value.get('feat', []))

            tracker.update(bbox, confidence, feats)
            for index, track in enumerate(tracker.tracks):
                image_name = f'img_{frame_number}_{index}'
                x, y, w, h = track.bbox
                x1, y1, x2, y2 = x, y, x + w, y + h
                x1, y1 = max(x1, 0), max(y1, 0)
                track_id = track.track_id
                feat = track.feat
                confidence = track.confidence
                output_line = f"{frame_number} {track_id} {x1} {y1} {x2} {y2}\n"
                final_result.write(output_line)
                mot_feat_data[image_name] = {'bbox': (x1, y1, x2, y2), 'frame': frame_number, 'track_id': track_id, 'feat': feat, 'conf': confidence}

                cv2.rectangle(frame_img, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)
                cv2.putText(frame_img, str(track_id), (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (colors[track_id % len(colors)]), 2)

            print(f'Processing {seq}, frame {frame_number}', end='\r')
            cap_out.write(frame_img)
        
        # Save mot_feat_data at the end
        with open(feat_pkl_file, 'wb') as f:
            pickle.dump(mot_feat_data, f, pickle.HIGHEST_PROTOCOL)

        tracker.clear()

    cap_out.release()
    end_time = time.time()
    print(f'Total time: {end_time - start_time} seconds for camera {seq}')
    print(f'Camera {seq} processing finished')

    return end_time - start_time


if __name__ == "__main__":
    seqs = ['c041', 'c042', 'c043', 'c044', 'c045', 'c046']
    num_process = 6
    t_total = 0

    for seq in seqs:
        print(f'Starting processing camera {seq}')
        t_total += main(seq)
        print(f'Camera {seq} finished')

    print(f'Total processing time: {t_total} seconds')
