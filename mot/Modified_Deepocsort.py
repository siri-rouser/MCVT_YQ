from deep_sort.deep_sort.tracker import Tracker as DeepSortTracker
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort import nn_matching
import numpy as np
from DeepOCSORT.trackers import integrated_ocsort_embedding as tracker_module






class Modified_Tracker:
    tracker = None
    encoder = None
    tracks = None

    def __init__(self):
        parser = tracker_module.args.make_parser()
        parser.add_argument("--result_folder", type=str, default="results/trackers/")
        parser.add_argument("--test_dataset", action="store_true")
        parser.add_argument("--exp_name", type=str, default="exp1")
        parser.add_argument("--min_box_area", type=float, default=10, help="filter out tiny boxes")
        parser.add_argument(
            "--aspect_ratio_thresh",
            type=float,
            default=1.6,
            help="threshold for filtering out boxes of which aspect ratio are above the given value.",
        )
        parser.add_argument(
            "--post",
            action="store_true",
            help="run post-processing linear interpolation.",
        )
        parser.add_argument("--w_assoc_emb", type=float, default=0.75, help="Combine weight for emb cost")
        parser.add_argument(
            "--alpha_fixed_emb",
            type=float,
            default=0.95,
            help="Alpha fixed for EMA embedding",
        )
        parser.add_argument("--emb_off", action="store_true")
        parser.add_argument("--cmc_off", action="store_true")
        parser.add_argument("--aw_off", action="store_true")
        parser.add_argument("--aw_param", type=float, default=0.5)
        parser.add_argument("--new_kf_off", action="store_true")
        parser.add_argument("--grid_off", action="store_true")
        args = parser.parse_args()

        oc_sort_args = dict(
        args=args,
        det_thresh=args.track_thresh, # i change it to 0.1 here
        iou_threshold=args.iou_thresh,
        asso_func=args.asso,
        delta_t=args.deltat,
        inertia=args.inertia,
        w_association_emb=args.w_assoc_emb,
        alpha_fixed_emb=args.alpha_fixed_emb,
        embedding_off=args.emb_off,
        cmc_off=args.cmc_off,
        aw_off=args.aw_off,
        aw_param=args.aw_param,
        new_kf_off=args.new_kf_off,
        grid_off=args.grid_off,
    )

        self.tracker = tracker_module.ocsort.OCSort(**oc_sort_args)

    def update(self,bboxs,confidence,feats,img):
        if len(bboxs) == 0:
            empty_pred = np.empty((0, 5))
            self.tracker.predict()
            self.tracker.update(empty_pred)  
            self.update_tracks()
            return
        dets = np.empty((0, 5))
        for bbox_id, bbox in enumerate(bboxs):
            np.append(dets, np.array(bbox,confidence[bbox_id]),axis=0)
            # dets.append(Detection(bbox, confidence[bbox_id], feats[bbox_id])) 
            # 同时我也认为每一个det[i]是一个实例，每次是在调用这个detection类的下面的实例
            # add detection class into the dets list, and the variable dets is able to use all methods of Detections,e.g dets[0].to_xyah
        # print(f'dets:{dets[bbox_id].feature}')

        self.tracker.update(dets,img)
    #    matched_feature = self.result_match(bboxs,feats)
        self.update_tracks()

    def update_tracks(self):
        # matching_index = [index_bbox,index_detected_bbox]
        tracks = []
        for i,track in enumerate(self.tracker.tracks):
            # if not track.is_confirmed() or track.time_since_update > 1:
            #     continue
            if track.time_since_update > 1:
                continue
            # bbox = track.to_tlbr() # This value is from [mean,variance]
            bbox = track.detection_bbox[-1]
            feat = track.features[-1]
            confidence = track.confidence[-1]
            # print(len(track.features))
         #   print(len(feat))
            # track.features = []

            id = track.track_id

            tracks.append(Track(id, bbox,feat,confidence))

        self.tracks = tracks

    def result_match(self,bbox,feats):
        detected_bboxs = []
        tracks = []
        matching_index = []
        matched_feature = []
        for track in self.tracker.tracks:
            detected_bboxs.append(track.to_tlbr()) # min x, min y, max x, max y :x1,y1,x2,y2

        for index_bbox, bbox in enumerate(bbox):
            for index_detected_bbox,detected_bbox in enumerate(detected_bboxs):
                if self._calculate_iou(detected_bbox,bbox)>=0.5:
                    if index_bbox not in matching_index:
                        matching_index.append(index_bbox)
                        break
        matched_feature.append(feats[index_bbox]) 
        return matched_feature

    def _calculate_iou(self,boxA, boxB):
        # Determine the coordinates of the intersection rectangle
        # print(f"boxA:{boxA}")
        # print(f"boxB:{boxB}")
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # Compute the area of intersection
        interArea = max(0, xB - xA) * max(0, yB - yA)

        # Compute the area of both the prediction and ground-truth rectangles
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        # Compute the intersection over union by taking the intersection area and dividing it by the sum of prediction + ground-truth areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou

    def clear(self):
        self.tracks=[]


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