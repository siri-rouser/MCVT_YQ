import os 
import json
import Modified_Deepsort 
import random
import cv2
import pickle


def main(seq):
    abs_path = '/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/detect_merge'
    source_abs_path = '/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/detection/images/test/S06'
    path_det = os.path.join(abs_path , seq , 'labels')
    path_feat = os.path.join(abs_path , seq , 'feature')
    path_source = os.path.join(source_abs_path,seq,'img1')
    feat_files = os.listdir(path_feat)
    sorted_feat_files = sorted(feat_files)
    det_files = os.listdir(path_det)
    sorted_det_files = sorted(det_files) 
    print(len(sorted_det_files))
    source_files = os.listdir(path_source)
    sorted_source_files = sorted(source_files)
    print(len(sorted_source_files))
    save_path = os.path.join(abs_path, seq, f'{seq}_mot.txt')
    tracker = Modified_Deepsort.Modified_Tracker()

    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(20)]

    video_out_path = os.path.join(abs_path, seq, 'out.mp4')

    cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), 10,(1280,960))
 #   mot_feat_data = {}
    feat_pkl_file = os.path.join(abs_path,seq,f'{seq}_mot_feat_new.pkl')
    mot_feat_data = {}
    with open(save_path, 'w') as final_result:
        for frame_number,det_file in enumerate(sorted_det_files):
            
            det_file_path = os.path.join(path_det, det_file) # get the imgxxxxxx.txt file path here
            feat_file_path = os.path.join(path_feat,det_file.replace('.txt', '.json')) # get the imgxxxxxx.json file path here
            frame_img_path = os.path.join(path_source,det_file.replace('.txt','.jpg')) # get imgxxxxx.jpg in here
            frame_img = cv2.imread(frame_img_path)
            feats = []
            feat = []
            bbox = []
            confidence = []
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
                    # detection = [bbox, confidence] # The detection in here contains the bbox info + confidence score
                    # detections.append(detection)
            #    print(f'bbox:{len(bbox)}')
    
            with open(feat_file_path, 'r') as f:
                feature_data = json.load(f)
                for key, value in feature_data.items():
                    feat_temp = value.get('feat', [])
                    feats.append(feat_temp)
            #    print(f'feats:{len(feats)}')

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
               # print(len(feat))
             
                output_line = f"{frame_number} {track_id} {x1} {y1} {x2} {y2}\n"
                final_result.write(output_line)
                mot_feat_data[image_name] = {'bbox':bbox,'frame':frame_number,'track_id':track_id,'feat':feat}

                cv2.rectangle(frame_img, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)
                cv2.putText(frame_img, str(track_id), (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (colors[track_id % len(colors)]), 2)
            print(frame_number)

            cap_out.write(frame_img)
            
        pickle.dump(mot_feat_data, open(feat_pkl_file, 'wb'), pickle.HIGHEST_PROTOCOL)
        tracker.clear()
    cap_out.release()
    print(f'The process of camera:{seq} finished')


if __name__ == "__main__":
    seqs = ['c041','c042','c043','c044','c045','c046']
    seqs = ['c046']
    for seq in seqs:
        print(f'Start Processing camera {seq}')
        main(seq)
        print(f'Camera {seq} Finished')