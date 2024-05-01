import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import cv2

class ZONE():
    def __init__(self,point,zone_id):
        self.points = []
        self.points.append(point) 
        self.zone_id = zone_id
        self.zone_cls = 'undefined_zone'

    def zone_append(self,point):
        self.points.append(point)

    def zone_classify(self,entry_pos,exit_pos):
        # The input is zone_data

        entry_num = 0
        exit_num = 0

        for point in self.points:
            if point in entry_pos:
                entry_num+=1
            elif point in exit_pos:
                exit_num+=1
        
        Entry_density = entry_num/(entry_num+exit_num)
        Exit_density = exit_num/(entry_num+exit_num)
        Traffic_density = 1 - abs(entry_num-exit_num)/(entry_num+exit_num)

        if len(self.points) > 5:
            if Entry_density > 0.8:
                self.zone_cls = 'entry_zone'
            elif Exit_density > 0.8:
                self.zone_cls = 'exit_zone'
            elif Traffic_density > 0.7:
                self.zone_cls = 'traffic_awareness_zone'
            else:
                self.zone_cls = 'undefined_zone'
        else:
            self.zone_cls = 'undefined_zone'

    def gpt_area_drawing(self,background_img):
        nppoints = np.array(self.points)
        dbscan = DBSCAN(eps=120, min_samples=6)
        clusters = dbscan.fit_predict(nppoints)
        main_cluster_points = nppoints[clusters == 0]

        # Draw all points with different colors for each cluster
        unique_clusters = np.unique(clusters)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]  # Different colors for clusters

        for point, cluster in zip(nppoints, clusters):
            if cluster != -1:  # Ignore noise points if any
                color = colors[cluster % len(colors)]  # Cycle through colors if there are many clusters
                cv2.circle(background_img, (int(point[0]), int(point[1])), 5, color, -1)  # Draw filled circle

        # Plotting the convex hull over the image if the main cluster has enough points
        if len(main_cluster_points) > 2:
            hull = ConvexHull(main_cluster_points)
            hull_points = main_cluster_points[hull.vertices]
            pts = hull_points.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(background_img, [pts], True, (0, 255, 255), 2)  # Draw the convex hull

        else:
            print("Not enough points in the main cluster to form a convex hull.")

        # Display the result in a window (for testing) or save to file
        # Save the resulting image with the drawing
        cv2.imwrite('plot_on_background.png', background_img)

        return background_img
    
    def area_define(self):
        min_x = min(point[0] for point in self.points)-5
        max_x = max(point[0] for point in self.points)+5
        min_y = min(point[1] for point in self.points)-5
        max_y = max(point[1] for point in self.points)+5

        self.rect_area = min_x,min_y,max_x,max_y

    def area_drawing(self,background_img):
        rd_color = list(np.random.random(size=3) * 256)
        texton = f'Zone_id{self.zone_id} {self.zone_cls}'
        cv2.rectangle(background_img,(int(self.rect_area[0]),int(self.rect_area[1])),(int(self.rect_area[2]),int(self.rect_area[3])),rd_color,8)
        cv2.putText(background_img, texton, (int(self.rect_area[0]),int(self.rect_area[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        return background_img
    
    def required_area_drawing(self, background_img, require):
        if self.zone_cls == require:
            rd_color = list(np.random.random(size=3) * 256)
            text = f'Zone_id{self.zone_id} {self.zone_cls}'
            cv2.rectangle(background_img,(int(self.rect_area[0]),int(self.rect_area[1])),(int(self.rect_area[2]),int(self.rect_area[3])),rd_color,8)
            cv2.putText(background_img, text, (int(self.rect_area[0]),int(self.rect_area[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        return background_img

class cross_zone():
    def __init__(self) -> None:
        pass