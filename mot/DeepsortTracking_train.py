import os 
import json
import Modified_Deepsort 
import random
import cv2
import pickle
import sys
sys.path.append('../')
from config import cfg


def main(seq):
    cfg.merge_from_file(f'../config/{sys.argv[1]}')
    cfg.freeze()
    abs_path = cfg.DATA_DIR
    source_abs_path = cfg.DET_SOURCE_DIR

    # abs_path = '/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/detect_merge'
    # source_abs_path = '/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/detection/images/test/S06'
    path_det = os.path.join(abs_path , seq , 'labels')
    path_feat = os.path.join(abs_path , seq , 'feature')
    path_source = os.path.join(source_abs_path,seq,'img1')
    feat_files = os.listdir(path_feat)
    sorted_feat_files = sorted(feat_files)
    det_files = os.listdir(path_det)
    sorted_det_files = sorted(det_files) 
    source_files = os.listdir(path_source)
    sorted_source_files = sorted(source_files)
    save_path = os.path.join(abs_path, seq, f'{seq}_mot.txt')
    tracker = Modified_Deepsort.Modified_Tracker()

    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(20)]

    video_out_path = os.path.join(abs_path, seq, 'out1.mp4')

    cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'mp4v'), 10,(1280,960))
 #   mot_feat_data = {}
    feat_pkl_file = os.path.join(abs_path,seq,f'{seq}_mot_feat_new.pkl')
    mot_feat_data = {}
    with open(save_path, 'w') as final_result:
        for frame_number, frame_img in enumerate(sorted_source_files): 
            frame_number = frame_number+1
            frame_img_path = os.path.join(path_source,frame_img)
            det_file_path = os.path.join(path_det, frame_img.replace('.jpg','.txt')) # get the imgxxxxxx.txt file path here
            feat_file_path = os.path.join(path_feat,frame_img.replace('.jpg', '.json')) # get the imgxxxxxx.json file path here

            frame_img = cv2.imread(frame_img_path)

            feats = []
            feat = []
            bbox = []
            confidence = []
    
            if os.path.exists(det_file_path):
                with open(det_file_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        elements = line.split()
                        # Extracting bounding box and confidence score
                        x_central,y_central,w,h = map(float, elements[1:5]) # x is x_central, y is y_central here
                        x = x_central -w/2
                        y = y_central - h/2
                        bbox_temp = (x,y,w,h)
                        confidence_temp = float(elements[5])
                        bbox.append(bbox_temp)
                        confidence.append(confidence_temp)


            if os.path.exists(feat_file_path):
                with open(feat_file_path, 'r') as f:
                    feature_data = json.load(f)
                    for key, value in feature_data.items():
                        feat_temp = value.get('feat', [])
                        feats.append(feat_temp)

            # print(f'bboxs length:{len(bbox)}')
            # print(f'confidence score length:{len(confidence)}')
            # print(f'feats length:{len(feats)}')
            # Data use for deepsort tracker
            tracker.update(bbox,confidence,feats)
            for index, track in enumerate(tracker.tracks):
                image_name = f'img_{frame_number}_{index}'
                bbox = track.bbox
                x1, y1, x2, y2 = bbox
                if x1<0:
                    x1 = 0
                track_id = track.track_id
                feat = track.feat
                output_line = f"{frame_number} {track_id} {x1} {y1} {x2} {y2}\n"
                final_result.write(output_line)
                mot_feat_data[image_name] = {'bbox':bbox,'frame':frame_number,'track_id':track_id,'feat':feat}

                cv2.rectangle(frame_img, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)
                cv2.putText(frame_img, str(track_id), (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (colors[track_id % len(colors)]), 2)
            print(f'we are processing {seq}, current in frame {frame_number}', end='\r')

            cap_out.write(frame_img)
            
        pickle.dump(mot_feat_data, open(feat_pkl_file, 'wb'), pickle.HIGHEST_PROTOCOL)
        tracker.clear()
    cap_out.release()
    print(f'The process of camera:{seq} finished')


if __name__ == "__main__":
    seqs = ['c001','c002','c003','c004','c005']
    # seqs = ['c045']
    for seq in seqs:
        print(f'Start Processing camera {seq}')
        main(seq)
        print(f'Camera {seq} Finished')