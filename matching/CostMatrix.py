import pickle
import numpy as np
import torch
import sys
# from sklearn.neighbors import KernelDensity

class CostMatrix:

    def __init__(self,query_track_path,gallery_track_path):
        self.query_track_path = query_track_path
        self.gallery_track_path = gallery_track_path

    def cost_matrix(self,metric):
        q_feats,q_track_ids,q_cam_ids,q_times,q_entry_zone,q_exit_zone = self._track_operation(self.query_track_path)
        g_feats,g_track_ids,g_cam_ids,g_times,g_entry_zone,g_exit_zone = self._track_operation(self.gallery_track_path)

        if metric == 'Euclidean_Distance':    
            distmat = self.euclidean_distance(q_feats, g_feats)
        elif metric == 'Cosine_Distance':
            distmat = self.cosine_distance(q_feats,g_feats)
        else:
            sys.exit('Please input the right metric')

        q_times = np.asarray(q_times)
        g_times = np.asarray(g_times)
        # zone is a int variable
        return distmat, q_track_ids, q_cam_ids, g_track_ids, g_cam_ids, q_times, g_times, q_entry_zone,q_exit_zone,g_entry_zone,g_exit_zone 

    def _track_operation(self,tracklet_path):
        feats, track_ids, cam_ids, entry_zones, exit_zones = [], [], [], [], []
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
                entry_zones.append([value['entry_zone_cls'], value['entry_zone_id']])
                exit_zones.append([value['exit_zone_cls'], value['exit_zone_id']])

            feats = torch.cat(feats,0)
            track_ids = np.asarray(track_ids)
            cam_ids = np.asarray(cam_ids)
       
        print('Got features for set, obtained {}-by-{} matrix'.format(feats.size(0), feats.size(1)))

        return feats,track_ids,cam_ids,times,entry_zones,exit_zones
    
    def euclidean_distance(self, q_feats, g_feats):
        m, n = q_feats.size(0), g_feats.size(0)
        distmat = torch.pow(q_feats, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                torch.pow(g_feats, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        # torch.power --> torch.sum(dim=1) make it to be one column, those two step calculate the L2 norm--> torch.expand make it expand to m,n(base on the size of qf and gf)
        # torch.t() is the transposed function
        distmat.addmm_(1, -2, q_feats, g_feats.t()) # here calculate the a^2+b^2-2ab 
        # in here, 1 is alpha, -2 is beta: 1*dismat -2*qf*gf.t()
        distmat = distmat.numpy()

        return distmat
    
    def cosine_distance(self,q_feats, g_feats):
        q_feats = torch.nn.functional.normalize(q_feats, p=2, dim=1) # p=2 means sqrt(||q_feats||^2)
        g_feats = torch.nn.functional.normalize(g_feats, p=2, dim=1)

        # Compute the cosine similarity
        cosine_sim = torch.mm(q_feats, g_feats.t())

        # Since cosine distance is 1 - cosine similarity
        distmat = 1 - cosine_sim

        # Convert the distance matrix from torch tensor to numpy array
        distmat = distmat.numpy()

        return distmat

class CostMatrix_Zone(CostMatrix):
    def __init__(self,entry_zone_data,exit_zone_data):
        self.entry_zone_data = entry_zone_data
        self.exit_zone_data = exit_zone_data

    def cost_matrix_zone(self, metric):
        q_feats,q_track_ids,q_cam_ids,q_times,q_entry_zone,q_exit_zone = self.data_extract(self.entry_zone_data)
        g_feats,g_track_ids,g_cam_ids,g_times,g_entry_zone,g_exit_zone = self.data_extract(self.exit_zone_data)


        if metric == 'Euclidean_Distance':    
            distmat = self.euclidean_distance(q_feats, g_feats)
        elif metric == 'Cosine_Distance':
            distmat = self.cosine_distance(q_feats,g_feats)
        else:
            sys.exit('Please input the right metric')

        q_times = np.asarray(q_times)
        g_times = np.asarray(g_times)
        # zone is a int variable
        return distmat, q_track_ids, q_cam_ids, g_track_ids, g_cam_ids, q_times, g_times, q_entry_zone,q_exit_zone,g_entry_zone,g_exit_zone 

    def data_extract(self,data):
        feats, track_ids, cam_ids, entry_zones, exit_zones,times = [], [], [], [], [], []
        with torch.no_grad():
            for key,value in data.items():
                feat_tensor = torch.tensor(value['feat'], dtype=torch.float).unsqueeze(0)
                feats.append(feat_tensor)
                track_ids.append(value['track_id'])
                cam_ids.append(int(value['cam'][-3:]))
                times.append([value['start_time'],value['end_time']])
                entry_zones.append(value['entry_zone_id'])
                exit_zones.append(value['exit_zone_id'])

            feats = torch.cat(feats,0)
            track_ids = np.asarray(track_ids)
            cam_ids = np.asarray(cam_ids)
        return feats,track_ids,cam_ids,times,entry_zones,exit_zones
