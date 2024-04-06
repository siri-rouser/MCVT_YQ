import ultralytics
from ultralytics import YOLO
from pathlib import Path
import cv2
import os
import torch
import numpy as np
from PIL import Image
import sys
sys.path.append('../../')
#from config import cfg
import pickle
 
seqs = ['c041','c042','c043','c044','c045','c046']
# seqs = ['c046']

# Load a model
model = YOLO("/home/yuqiang/yl4300/project/MCVT_YQ/detector/yolov9e.pt")  # load an official model

vehicles = [2, 3, 5, 7]

for seq in seqs:
    #DONOT FORGET TO MODIFY THE PATH!!!!!!!!!!!!!!
    source = f'/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/detection/images/test/S06/{seq}/img1/'
    ##############################################

    # Directories
    # save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    save_txt = True
    save_dir = Path(Path('/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/detect_merge') / Path(seq))
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    save_img = True
    dataset = ultralytics.data.loaders.LoadImagesAndVideos(source) # Hope this would be useful
    # Be careful about the dataset, I donot know what the output would be 
    # For the previous code, the webcam should be False, so all input should be image, pls check in 2E aggin
    # It will return [path], [im0], self.cap, s

    out_dict=dict()
    
    
    for path,img0,vid_cap,_ in dataset:
        img = np.array(img0[0], dtype=np.uint8) 
        result = model(img, conf=0.2, agnostic_nms=True, save_txt=True, imgsz=1280, classes=vehicles, iou=0.45)
        p = Path(path[0])  # to Path
        save_path = str(save_dir / 'dets_debug' / p.name)  # img.jpg
        txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
        txt_path_data = str(save_dir / 'labels_xy' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
        det_path = str(save_dir / 'dets' /p.stem)

        if not os.path.isdir(str(save_dir / 'dets')):
            os.makedirs(str(save_dir / 'dets'))
        if not os.path.isdir(str(save_dir / 'dets_debug')):
            os.makedirs(str(save_dir / 'dets_debug'))
        if not os.path.isdir(str(save_dir / 'labels_xy')):
            os.makedirs(str(save_dir / 'labels_xy'))


        central_data = []
        xy_data = []
 
        for i, result in enumerate(result):  # detections per image
            p, s, im0, frame = path, '', img0, getattr(dataset, 'frame', 0)

            p = Path(p[0])

            # .stem and name are attributes for path object:
            # for name, it would be the file name of the path, e.g. p = Path('/run/detect/c042/img001.jpg'), p.name = img001.jpg
            # for stem, it returns the final path component, without its suffix. same example it returns img001 without '.jpg'

            

            if len(result): # if any vehicle is been detected here
                img_copy = np.copy(img0)
                det_num=0
    
                for res in result:
                    
                    x1 = res.boxes.data.tolist()[0][0]
                    y1 = res.boxes.data.tolist()[0][1]
                    x2 = res.boxes.data.tolist()[0][2]
                    y2 = res.boxes.data.tolist()[0][3]
                    score = res.boxes.data.tolist()[0][4]
                    class_id = res.boxes.data.tolist()[0][5]
        
                    # Convert to [x_center, y_center, width, height]
                    x_center = (x1 + x2) / 2
                    y_center = (y1 + y2) / 2
                    width = x2 - x1
                    height = y2 - y1

                    # Prepare line for saving
                    # line = [class_id, x_center, y_center, width, height, score]
                    line = f"{class_id} {x_center} {y_center} {width} {height} {score}\n"
                    central_data.append(line)

                    # Change the data storage structure
                    x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
                    # line_data = [class_id, x1, y1, x2, y2, score]
                    line_data = f"{class_id} {x1} {y1} {x2} {y2} {score}\n"
                    xy_data.append(line_data)
                    



                    det_name = p.stem+"_{:0>3d}".format(det_num)
                    det_img_path = det_path+"_{:0>3d}.png".format(det_num)
                    det_class = -1 # leave it
                    det_conf = [] # leave it
                    # im_array = res.plot()
                    # im = Image.fromarray(im_array[..., ::-1])  # RGB PIL图像
                    # im=im.crop((x1, y1, x2, y2))
                    # im.save(det_img_path)
                    cv2.imwrite(det_img_path,img[y1:y2,x1:x2])
                    out_dict[det_name]= {
                        'bbox': (x1,y1,x2,y2),
                        'frame': p.stem,
                        'id': det_num,
                        'imgname': det_name+".png",
                        'class': det_class,
                        'conf': det_conf
                    }

                    det_num+=1
                    cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
                print('det_num:')
                print(det_num)


                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, img)
                    else:  # 'video'
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer

                            fourcc = 'mp4v'  # output video codec
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                        vid_writer.write(img)

                            # Write to file
      
        if save_txt:
            with open(txt_path + '.txt', 'w') as f: # a means append, what f.write in here will be appended into the end of the file
                f.writelines(central_data)

        # Write to file
        if save_txt:
            with open(txt_path_data + '.txt', 'w') as f: # a means append, what f.write in here will be appended into the end of the file
                f.writelines(xy_data)  

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
        pickle.dump(out_dict,open(str(save_dir / '{}_dets.pkl'.format(seq)),'wb'))