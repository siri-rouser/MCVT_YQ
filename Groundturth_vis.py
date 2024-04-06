import cv2
import os

def main(seq):
    abs_path = '/home/yuqiang/yl4300/project/MCVT_YQ/datasets/AIC22_Track1_MTMC_Tracking/test/S06'
    res_abs_path = '/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/detect_merge'
    label_path = os.path.join(res_abs_path, seq, 'labels_xy')
    det_label_file = os.listdir(label_path)
    sorted_det_label_file = sorted(det_label_file)

    source_abs_path = '/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/detection/images/test/S06'
    path_source = os.path.join(source_abs_path, seq, 'img1')

    source_files = os.listdir(path_source)
    sorted_source_files = sorted(source_files)
    if not sorted_source_files:
        print("No source files found. Exiting.")
        return

    # Assuming frame dimensions and FPS from the first image
    first_frame_path = os.path.join(path_source, sorted_source_files[0])
    first_frame = cv2.imread(first_frame_path)
    height, width, layers = first_frame.shape
    fps = 10 

    save_path = os.path.join(res_abs_path, seq, f'detected_{seq}_output.avi')
    video = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))


    for frame_number, source_file in enumerate(sorted_source_files):
        frame_img_path = os.path.join(path_source, source_file)
        frame_img = cv2.imread(frame_img_path)
        print(frame_img_path)
        temp_path = os.path.join(label_path, sorted_det_label_file[frame_number])
        print(temp_path)
        with open(temp_path, 'r') as f:  # ground truth
            lines = f.readlines()

        for line in lines:
            elements = line.split(' ')

            bbox = list(map(float, elements[1:5]))

            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        video.write(frame_img)

    video.release()
    print(f'Video for sequence {seq} has been saved to {save_path}')

if __name__ == "__main__":
    seqs = ['c041', 'c042', 'c043', 'c044', 'c045', 'c046']
    for seq in seqs:
        print(f'start processing {seq} ---')
        main(seq)
        print(f'camera {seq} finished', seq)
