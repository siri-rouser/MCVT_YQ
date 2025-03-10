"""Extract image feature for both det/mot image feature."""

import os
import pickle
import time
from glob import glob
from itertools import cycle
from multiprocessing import Pool, Queue
import tqdm
import shutil

import torch
from PIL import Image # This is a 'offical' image process library in python
import torchvision.transforms as T
from reid_inference.reid_model import build_reid_model
import sys
import numpy as np
sys.path.append('../')
from config import cfg

BATCH_SIZE = 64
NUM_PROCESS = 1
FLAG = 0
def chunks(l):
    return [l[i:i+BATCH_SIZE] for i in range(0, len(l), BATCH_SIZE)] # this function is making batchs, it retuns image_list[0-63,64-127 ....]

class ReidFeature():
    """Extract reid feature."""

    def __init__(self, gpu_id, _mcmt_cfg):
        print("init reid model")
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        self.model, self.reid_cfg = build_reid_model(_mcmt_cfg)
        # The model is called in here through the build_reid_model function
        # Then the configuration file are here and send to a make_model() function
        device = torch.device('cuda')
        self.model = self.model.to(device)
        self.model.eval()
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.val_transforms = T.Compose([T.Resize(self.reid_cfg.INPUT.SIZE_TEST, interpolation=3),\
                              T.ToTensor(), T.Normalize(mean=mean, std=std)]) # T.Compose 串联起多个变换, they first do T.Resize, then T.ToTensor thn T.Normalize

    def extract(self, img_path_list):
        """Extract image feature with given image path.
        Feature shape (2048,) float32."""

        img_batch = []
        for img_path in img_path_list:
            img = Image.open(img_path).convert('RGB') # IMG are open in here 
            img = self.val_transforms(img)
            img = img.unsqueeze(0) # Add one extra channel for the img, the previous one is img.shape = [channel, height,width], after the processing it will becoems img.shpe = [1,channel,height,width]
            img_batch.append(img)
        img = torch.cat(img_batch, dim=0) # torch.cat is the function to concatenate imgs

        with torch.no_grad():
            img = img.to('cuda')
            flip_feats = False
            if self.reid_cfg.TEST.FLIP_FEATS == 'yes': flip_feats = True
            if flip_feats:
                for i in range(2):
                    if i == 1:
                        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().cuda() 
                        # torch.arange returns [end-start/step] 1-D tensor
                        # img.size(3) returns the width of the img
                        # e.g. width = 100 it returns [99,98,97,...,1,0]
                        img = img.index_select(3, inv_idx)
                        print(img.shape)
                        # torch.index_select function is a function returns select_index correspond img
                        # so what this function do is to flip the img and then extract the featrue 
                        feat1 = self.model(img)
                    else:
                        feat2 = self.model(img)
                feat = feat2 + feat1
            else:
                feat = self.model(img)
        feat = feat.cpu().detach().numpy() # move features from gpu to cpu, and then detach() it(don't need the gradient) and convert it to an numpy array!
        return feat # Feature shape (2048,) float32.


def init_worker(gpu_id, _cfg):
    """init worker."""

    # pylint: disable=global-variable-undefined
    global model
    model = ReidFeature(gpu_id.get(), _cfg)


def process_input_by_worker_process(image_path_list):
    """Process_input_by_worker_process."""
    global FLAG 

    # print(image_path_list)
    # shutil.copy(image_path_list[0], '/home/yuqiang/yl4300/project/MCVT_YQ/reid')

    reid_feat_numpy = model.extract(image_path_list) # get all reid-feat here!
    # print(reid_feat_numpy)

    # # Save the reid_feat_numpy to a .npy file
    # save_path = '/home/yuqiang/yl4300/project/MCVT_YQ/reid/reid_feat.npy'  # Change the path as needed
    # np.save(save_path, reid_feat_numpy)
    # print(f"Features saved to {save_path}")


    feat_dict = {}
    for index, image_path in enumerate(image_path_list):
        feat_dict[image_path] = reid_feat_numpy[index]
    print(f'=======batch {FLAG} finished========')
    FLAG +=1
    return feat_dict # this is also the pool_output


def load_all_data(data_path):
    """Load all mode data."""
    print(f'data path is {data_path}')

    image_list = []
    for cam in os.listdir(data_path):
        image_dir = os.path.join(data_path, cam, 'dets') # .../AIC21-MTMC/datasets/algorithm_results/detected_reid1 + c40 + dets
        cam_image_list = glob(image_dir+'/*.png') # The glob function search for all file end with .png in the image_dir, the * means any potential name
        cam_image_list = sorted(cam_image_list) # Sort all files with ascending sequence like 1.png 2.png
        print(f'{len(cam_image_list)} images for {cam}') # returns: 402 images for c042
        image_list += cam_image_list 
    print(f'{len(image_list)} images in total')
    return image_list


def load_certain_data(data_path):
    image_list = []
    cams = []
    for cam in os.listdir(data_path):
        cams.append(cam)
    cam = cams[0]
    print(f'I am processing {cam}')

    image_dir = os.path.join(data_path, cam, 'dets') # .../AIC21-MTMC/datasets/algorithm_results/detected_reid1 + c41 + dets
    cam_image_list = glob(image_dir+'/*.png') # The glob function search for all file end with .png in the image_dir, the * means any potential name
    cam_image_list = sorted(cam_image_list) # Sort all files with ascending sequence like 1.png 2.png
    print(f'{len(cam_image_list)} images for {cam}') # returns: 402 images for c042
    image_list += cam_image_list 
    print(f'{len(image_list)} images in total')
    return image_list


def save_feature(output_path, data_path, pool_output):
    # output_path: '/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/detect_reid1/'
    # data_path: '/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/detect_merge/'
    """Save feature."""
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    all_feat_dic = {}
    for cam in os.listdir(data_path):
        dets_pkl_file = os.path.join(data_path, cam, f'{cam}_dets.pkl')
        det_dic = pickle.load(open(dets_pkl_file, 'rb'))
        all_feat_dic[cam] = det_dic.copy()

    for sample_dic in pool_output:
        for image_path, feat in sample_dic.items():
            cam = image_path.split('/')[-3]
            image_name = image_path.split('/')[-1].split('.')[0]
            # if image_name not in all_feat_dic[cam]:
            #     all_feat_dic[cam][image_name] = {}
            all_feat_dic[cam][image_name]['feat'] = feat
    for cam, feat_dic in all_feat_dic.items():
        if not os.path.isdir(os.path.join(output_path, cam)):
            os.makedirs(os.path.join(output_path, cam))
        feat_pkl_file = os.path.join(output_path, cam, f'{cam}_dets_feat.pkl')
        pickle.dump(feat_dic, open(feat_pkl_file, 'wb'), pickle.HIGHEST_PROTOCOL)
        print('save pickle in %s' % feat_pkl_file)


def extract_image_feat(_cfg):
    """Extract reid feat for each image, using multiprocessing."""

    image_list = load_all_data(_cfg.DET_IMG_DIR) # dataset load
    # image_list = load_certain_data(_cfg.DET_IMG_DIR)
    chunk_list = chunks(image_list) # take batchs
    print('=============data load finish=======')

    num_process = NUM_PROCESS # equal to 1 here, means single GPU i guess
    gpu_ids = Queue() # returns a queue contains different GPU IDs
    gpu_id_cycle_iterator = cycle(range(0, 2)) # returns 0 1 0 1 0 1 
    for _ in range(num_process):
        gpu_ids.put(next(gpu_id_cycle_iterator)) # the next function just return the result of a iterate process

    process_pool = Pool(processes=num_process, initializer=init_worker, initargs=(gpu_ids, _cfg, )) 
    # NUM_Process means how many threads are here, initializer is the model and initagrs is args for the model
    # Pool is the function pool of worker processes
    start_time = time.time()
    pool_output = list(tqdm.tqdm(process_pool.imap_unordered(\
                                 process_input_by_worker_process, chunk_list),
                                 total=len(chunk_list))) 
    # The computation starts! tqdm.tqdm () add the process bar to the command line
    # process_pool is an instance from multiprocessing
    # it will callinializer(*initargs) when it starts
    # the process_pool.imap_unordered call functions and then chunks
    process_pool.close()
    process_pool.join()

    # global model
    # model = ReidFeature(0)
    # for sub_list in chunk_list:
    #     ret = process_input_by_worker_process(sub_list)
    print('%.4f s' % (time.time() - start_time))

    save_feature(_cfg.DATA_DIR, _cfg.DET_IMG_DIR, pool_output)


def debug_reid_feat(_cfg):
    """Debug reid feature to make sure the same with Track2."""

    exp_reidfea = ReidFeature(0, _cfg)
    feat = exp_reidfea.extract(['debug/img000000_006.png', 'debug/img000014_007.png','debug/img000004_008.png'])
    print(feat)
    model_name = _cfg.REID_MODEL.split('/')[-1]
    np.save(f'debug/{model_name}.npy', feat)

def extract_image_feat1(_cfg):
    """Extract reid feat for each image, using multiprocessing."""

    image_list = load_all_data(_cfg.DET_IMG_DIR) # dataset load
    # image_list = load_certain_data(_cfg.DET_IMG_DIR)
    chunk_list = chunks(image_list) # take batchs
    print('=============data load finish=======')

    gpu_ids = Queue() # returns a queue contains different GPU IDs
    gpu_id_cycle_iterator = cycle(range(0, 2)) # returns 0 1 0 1 0 1 
    for _ in range(1):
        gpu_ids.put(next(gpu_id_cycle_iterator)) # the next function just return the result of a iterate process

    init_worker(gpu_id=gpu_ids, _cfg=_cfg)
    for chunk in chunk_list:
        process_input_by_worker_process(chunk)

    # num_process = NUM_PROCESS # equal to 1 here, means single GPU i guess
    # gpu_ids = Queue() # returns a queue contains different GPU IDs
    # gpu_id_cycle_iterator = cycle(range(0, 2)) # returns 0 1 0 1 0 1 
    # for _ in range(num_process):
    #     gpu_ids.put(next(gpu_id_cycle_iterator)) # the next function just return the result of a iterate process

    # process_pool = Pool(processes=num_process, initializer=init_worker, initargs=(gpu_ids, _cfg, )) 
    # # NUM_Process means how many threads are here, initializer is the model and initagrs is args for the model
    # # Pool is the function pool of worker processes
    # start_time = time.time()
    # pool_output = list(tqdm.tqdm(process_pool.imap_unordered(\
    #                              process_input_by_worker_process, chunk_list),
    #                              total=len(chunk_list))) 
    # The computation starts! tqdm.tqdm () add the process bar to the command line
    # process_pool is an instance from multiprocessing
    # it will callinializer(*initargs) when it starts
    # the process_pool.imap_unordered call functions and then chunks
    # process_pool.close()
    # process_pool.join()

    # # global model
    # # model = ReidFeature(0)
    # # for sub_list in chunk_list:
    # #     ret = process_input_by_worker_process(sub_list)
    # print('%.4f s' % (time.time() - start_time))

    # save_feature(_cfg.DATA_DIR, _cfg.DET_IMG_DIR, pool_output)

def main():
    """Main method."""

    cfg.merge_from_file(f'../config/{sys.argv[1]}')
    cfg.freeze()
    # debug_reid_feat(cfg)
    extract_image_feat(cfg)

def main1():
    cfg.merge_from_file(f'../config/{sys.argv[1]}')
    cfg.freeze()
    # debug_reid_feat(cfg)

    extract_image_feat1(cfg)


if __name__ == "__main__":
    # main1()
    main()