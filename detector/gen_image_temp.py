import os
import cv2

def preprocess_video(video_path, dst_dir, roi_path=None):
    # Ensure the destination directory exists
    if not os.path.isdir(dst_dir):
        os.makedirs(dst_dir)
        print(f"{dst_dir} created.")

    # Load the ROI (Region of Interest) if provided
    ignor_region = None
    if roi_path:
        ignor_region = cv2.imread(roi_path)

    # Open the video file
    video = cv2.VideoCapture(video_path)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_current = 0

    # Process each frame
    while frame_current < frame_count - 1:
        frame_current = int(video.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = video.read()
        if not ret:
            break

        dst_f = f'img{frame_current:06d}.jpg'
        dst_f_path = os.path.join(dst_dir, dst_f)

        if not os.path.isfile(dst_f_path):
            if ignor_region is not None:
                frame = draw_ignore_regions(frame, ignor_region)
            cv2.imwrite(dst_f_path, frame)
            print(f'{dst_f} generated to {dst_dir}')
        else:
            print(f'{dst_f} already exists.')

    video.release()
    print('Processing complete.')

def draw_ignore_regions(img, region):
    if img is None:
        print('[Err]: Input image is none!')
        return None
    img = img * (region > 0)
    return img

if __name__ == '__main__':
    # Replace with your actual paths
    video_path = '/home/yuqiang/yl4300/project/MCVT_YQ/datasets/AIC22_Track1_MTMC_Tracking/test/S06/c047/test.mp4'
    dst_dir = '/home/yuqiang/yl4300/project/MCVT_YQ/datasets/temp'


    preprocess_video(video_path, dst_dir)
