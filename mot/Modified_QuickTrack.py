from QuickTrack import QT
from typing import List

class Modified_QT:    
    tracker = None
    encoder = None
    tracks = None

    def __init__(self):
        classPath: str = 'QuickTrack/QT/default.names'
        threshold: float = 0.1
        maxDisplacement: List[int] = [150, 100]
        maxColourDif: int = 2000
        maxShapeDif: float = 0.5
        weights: List[int] = [0.4, 0.6]
        # print(weights)
        maxAge: int = 20
        colour: str = 'no' # you may need to change that if you want to have more colour
        vitalScale: float = 0.7
        assign: str = 'LinAssign'
        self.tracker = QT.QuickTrack(classPath,threshold,maxDisplacement,maxColourDif,maxShapeDif,weights,maxAge,colour,vitalScale,assign)

    def update(self,bboxs,confidences,feats,frame_img):
        if len(bboxs) == 0:
            self.tracker.update([],[],frame_img)  
            self.update_tracks()
            return
        
        dets = []
        for bbox_id, bbox in enumerate(bboxs):
            det_temp = list(bbox[0:4]) 
            det_temp.append(confidences[bbox_id])
            det_temp.append(float(2.0))
            dets.append(det_temp)

        self.tracker.update(dets,feats,frame_img)
        self.update_tracks()


    def update_tracks(self):
        # matching_index = [index_bbox,index_detected_bbox]
        tracks = []
        for i,track in enumerate(self.tracker.tracks):
            # if not track.is_confirmed() or track.time_since_update > 1:
            #     continue
            # if track.time_since_update > 1:
            #     continue
            # bbox = track.to_tlbr() # This value is from [mean,variance]
            bbox = track.bbox
            feat = track.features[-1]
            confidence = track.conf
            # print(len(track.features))
         #   print(len(feat))
            # track.features = []

            id = track.Id

            tracks.append(Track(id, bbox,feat,confidence))

        self.tracks = tracks


class Track:
    track_id = None
    bbox = None
    feat = None
    confidence = None

    def __init__(self, id, bbox,feat,confidence):
        self.track_id = id
        self.bbox = bbox
        self.feat = feat
        self.confidence = confidence