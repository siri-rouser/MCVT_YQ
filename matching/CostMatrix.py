import pickle
import numpy as np
import torch


class CostMatrix:

    def __init__(self,query_track_path,gallery_track_path):
        self.query_track_path = query_track_path
        self.gallery_track_path = gallery_track_path

    def time_spatioal_constains(self):
        pass

    def cost_matrix(self):
        q_feats,q_track_ids,q_cam_ids,q_times = self._track_operation(self.query_track_path)
        g_feats,g_track_ids,g_cam_ids,g_times = self._track_operation(self.gallery_track_path)
        
        m, n = q_feats.size(0), g_feats.size(0)
        distmat = torch.pow(q_feats, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                torch.pow(g_feats, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        # torch.power --> torch.sum(dim=1) make it to be one column, those two step calculate the L2 norm--> torch.expand make it expand to m,n(base on the size of qf and gf)
        # torch.t() is the transposed function
        distmat.addmm_(1, -2, q_feats, g_feats.t()) # here calculate the a^2+b^2-2ab 
        # in here, 1 is alpha, -2 is beta: 1*dismat -2*qf*gf.t()
        distmat = distmat.numpy()

        q_times = np.asarray(q_times)
        g_times = np.asarray(g_times)

        return distmat, q_track_ids, q_cam_ids, g_track_ids, g_cam_ids, q_times, g_times

    def _track_operation(self,tracklet_path):
        feats, track_ids, cam_ids = [], [], []
        times = []
        with open(tracklet_path,'rb') as f:
            track_info = pickle.load(f)
        
        with torch.no_grad():
            for track_id,value in track_info.items():
                feat_tensor = torch.tensor(value['feat'], dtype=torch.float).unsqueeze(0)# torch.unsequezze(0) add one dimenson to let it be [1x1024] ratherthan [1024]
                feats.append(feat_tensor)
                track_ids.append(track_id)
                cam_ids.append(int(value['cam'][-3:]))
                times.append([value['start_time'],value['end_time']])

            feats = torch.cat(feats,0)
            track_ids = np.asarray(track_ids)
            cam_ids = np.asarray(cam_ids)
       
        print('Got features for set, obtained {}-by-{} matrix'.format(feats.size(0), feats.size(1)))

        return feats,track_ids,cam_ids,times
    
