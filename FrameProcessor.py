import numpy as np
import cv2
import scipy.ndimage.measurements

class FrameProcessor():
    """
    Frame processor - implement pipeline to detect vehicles in frame
    """

    def __init__(
        self,
        imageEngine, # Image engine
        classifierFast, # Image classifier for initial classification
        classifierAccurate, # Image classifier for additional classification
        baseWindowSize, # Base window size for detection
        detectionRegions, # Detection regions list with elements: [(y1,x1), (y2,x2), (sz1, sz2, sz3), (ovlp_y, ovlp_x)], where (y1,x1), (y2,x2) - left top and bottom right coordinates of region, sz1, sz2,... - windowses used for detection, (ovlp_y, ovlp_x) - window overlap [0, 1)
        visualization = False, # If this parameter is enabled - produce visualization of frame processing step by step.
        heatMapFrames = 1, # Number of historical frames used in heatmap, must be 1 for single images
        heatMapThreshold = 2, # Filter less number of matches
        heatMapTotalMin = 200, # Filter number of total heat points for correct detection
        heatMapTotalMinFrame = 200, # Filter number of total heat points for correct detection (one frame)
        heatMapEdgeThereshold = 0.75, # Parameter to detect multiple objects with overlap edges
        heatMapRegionThreshold = 0.5, # Parameter to recognize that some square region belongs to object
        heatMapConvolutionWindowSize = 64, # Convolution window size to split big region on horizontal sub-regions
        objsDetFrames = 7, # Number of frames where object will be tracked if possible
        objsDetCrossFrameMaxDist = 48, # Max distance between objects in sequential frames
        objMergeThreshold = 0.6, # If some objects overlap, it can be merged if overlap percent exceed this parameter
        objHeightWidthLimit = 1.5, # Limit of height to width proportion
        annotationWindowСolor = (255, 0, 0), # Annotation color of windows
        annotationWindowСolorList = [(0, 255, 0), (0, 0, 255), (255, 0, 255)], # List of colors to annotate tracked objects with color
        annotationWindowThickness = 3, # Annotation thickness of window
        ):

        self.imageEngine = imageEngine
        self.classifierFast = classifierFast
        self.classifierAccurate = classifierAccurate
        self.baseWindowSize = baseWindowSize
        self.detectionRegions = detectionRegions
        
        self.heatMapFrames = heatMapFrames
        self.heatMapThreshold = heatMapThreshold
        self.heatMapTotalMin = heatMapTotalMin
        self.heatMapTotalMinFrame = heatMapTotalMinFrame
        self.heatMapEdgeThereshold = heatMapEdgeThereshold
        self.heatMapRegionThreshold = heatMapRegionThreshold
        self.heatMapConvolutionWindowSize = heatMapConvolutionWindowSize

        self.objsDetFrames = objsDetFrames
        self.objsDetCrossFrameMaxDist = objsDetCrossFrameMaxDist
        self.objMergeThreshold = objMergeThreshold
        self.objHeightWidthLimit = objHeightWidthLimit

        self.annotationWindowСolor = annotationWindowСolor
        self.annotationWindowСolorList = annotationWindowСolorList
        self.annotationWindowThickness = annotationWindowThickness

        self.visualization = visualization
        self.visOrigImage = None
        self.visAllBoxesImage = None
        self.visVehicleBoxesImage = None
        self.visHeatMapImage = None
        
        self.isImageAnnotated = False
        self.visImageAnnotated = None

        self.regionHistory = []
        self.objectsHistory = []
        self.objLastLabel = 0

    def processFrame(
        self,
        img # Source image in OpenCV BGR format
        ):
        """
        Frame processing entry point
        """

        if self.visualization:
            self.visOrigImage = img.copy()
            self.visAllBoxesImage = img.copy()
            self.visVehicleBoxesImage = img.copy()

            self.visHeatMapImage = np.zeros((img.shape[0], img.shape[1], 3), dtype = np.uint8)
            self.visHeatMapImage[:,:,0] = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2GRAY) // 2
            self.visHeatMapImage[:,:,1] = self.visHeatMapImage[:,:,0]
            self.visHeatMapImage[:,:,2] = self.visHeatMapImage[:,:,0]
        
        isFirstPrint = True
        self.isImageAnnotated = False
        self.visImageAnnotated = img.copy()
        
        conv_win_half = self.heatMapConvolutionWindowSize // 2 # Half of convolution window size
        conv_win = np.ones(self.heatMapConvolutionWindowSize) # Convolution windows - contains all values 1.0

        region_set = []
        heatMap = np.zeros_like(img[:,:,0], dtype = np.float)

        # For each feature size and region on image find all features with sliding window
        for detection_region in self.detectionRegions:
            for feature_size in detection_region[2]:
                blur_kernel_size = (feature_size // 8) - 1
                img_blur = cv2.GaussianBlur(img, (blur_kernel_size, blur_kernel_size), 0)

                features = self.imageEngine.getImageFeatures(
                    img_blur,
                    img_window = (detection_region[0], detection_region[1]),
                    featureSize = (self.baseWindowSize, self.baseWindowSize),
                    featureScale = float(feature_size) / float(self.baseWindowSize),
                    overlap = detection_region[3],
                    visualise = False)

                features = np.array(features)
                x_predict = []

                for feature in features:
                    box0, box1, f_vector = feature[0:3]
                    box0 = (int(box0[0]), int(box0[1]))
                    box1 = (int(box1[0]), int(box1[1]))

                    x_predict += [f_vector]

                    if self.visualization:
                        cv2.rectangle(self.visAllBoxesImage, box0[::-1], box1[::-1], self.annotationWindowСolor, self.annotationWindowThickness)

                x_predict = np.array(x_predict)
                features = np.array(features)

                if len(x_predict) > 0:
                    y_predict = self.classifierFast.predict(x_predict)
                    x_predict = x_predict[y_predict == 1]
                    features = features[y_predict == 1]

                    if len(x_predict) > 0:
                        y_predict = self.classifierAccurate.predict(x_predict)
                        features = features[y_predict == 1]

                        for feature in features:
                            box0, box1, f_vector = feature[0:3]
                            box0 = (int(box0[0]), int(box0[1]))
                            box1 = (int(box1[0]), int(box1[1]))

                            # Add region in list of regions and on heat map
                            region_set += [[box0, box1]]
                            heatMap[box0[0]:box1[0], box0[1]:box1[1]] += 1

                            if self.visualization:
                                cv2.rectangle(self.visVehicleBoxesImage, box0[::-1], box1[::-1], self.annotationWindowСolor, self.annotationWindowThickness)
                                cv2.rectangle(self.visHeatMapImage, box0[::-1], box1[::-1], (127, 127, 0), 1)

        if self.visualization:
            heat_map_img = heatMap.copy().astype(np.float)
            max_heat_map = np.max(heat_map_img)
            if max_heat_map > 0:
                heat_map_img = (heat_map_img / max_heat_map * 128.0) + 128.0
            heat_map_img = np.clip(heat_map_img, 128, 255).astype(np.uint8)

            (self.visHeatMapImage[:,:,2])[heat_map_img > 128] = heat_map_img[heat_map_img > 128]

        self.regionHistory += [region_set]
        if len(self.regionHistory) > self.heatMapFrames:
            self.regionHistory = self.regionHistory[1:]

        if len(self.regionHistory) >= self.heatMapFrames:
            # Combine history regions in one set
            region_set = []
            for cur_reg_set in self.regionHistory:
                region_set += cur_reg_set

            isObjectDetected = True
            detectedObjects_all = []

            # Repeat object detection in cycle to detect overlapped objects
            while isObjectDetected:
                isObjectDetected = False
                detectedObjects = []

                # Create heat map from hisorical regions set
                heatMaps_total = np.zeros_like(img[:,:,0], dtype = np.float)
                for box0, box1 in region_set:
                    heatMaps_total[box0[0]:box1[0], box0[1]:box1[1]] += 1

                # Filter heat map
                heatMaps_total[(heatMaps_total < self.heatMapThreshold) | (heatMap <= 0)] = 0
            
                # Detect separate regions based on heat map (labeling)
                labels_matrix, labels_num = scipy.ndimage.measurements.label(heatMaps_total)
                if labels_num > 0:
                    for label_idx in range(1, labels_num + 1):
                        labels_matrix_filter = labels_matrix == label_idx
                        label_coord = np.array(labels_matrix_filter.nonzero())
                        box0 = (np.min(label_coord[0]), np.min(label_coord[1]))
                        box1 = (np.max(label_coord[0]) + 1, np.max(label_coord[1]) + 1)

                        # Algorithm to find maximal inner region - biggest possible sub-region of labeled region
                        def get_best_fit_box(edges, this_box0, this_box1):
                            if len(edges) <= 0:
                                return this_box0, this_box1, np.sum(heatMaps_total[this_box0[0]:this_box1[0], this_box0[1]:this_box1[1]])

                            best_sz = -1
                            for idx in range(len(edges)):
                                cur_edge = edges[idx]
                                labels_matrix_filter_box = labels_matrix_filter[this_box0[0]:this_box1[0], this_box0[1]:this_box1[1]]
                                new_box0, new_box1 = this_box0, this_box1

                                if cur_edge == 0:
                                    filter_arr = np.argmax(labels_matrix_filter_box, axis=0)
                                    filter_arr.sort()
                                    new_box0 = (new_box0[0] + filter_arr[int(len(filter_arr) * self.heatMapEdgeThereshold)], new_box0[1])
                                elif cur_edge == 1:
                                    filter_arr = np.argmax(labels_matrix_filter_box[::-1, :], axis=0)
                                    filter_arr.sort()
                                    new_box1 = (new_box1[0] - filter_arr[int(len(filter_arr) * self.heatMapEdgeThereshold)], new_box1[1])
                                elif cur_edge == 2:
                                    filter_arr = np.argmax(labels_matrix_filter_box, axis=1)
                                    filter_arr.sort()
                                    new_box0 = (new_box0[0], new_box0[1] + filter_arr[int(len(filter_arr) * self.heatMapEdgeThereshold)])
                                else:
                                    filter_arr = np.argmax(labels_matrix_filter_box[::-1, :], axis=1)
                                    filter_arr.sort()
                                    new_box1 = (new_box1[0], new_box1[1] - filter_arr[int(len(filter_arr) * self.heatMapEdgeThereshold)])

                                cur_box0, cur_box1, cur_sz = get_best_fit_box(edges[:idx] + edges[(idx + 1):], new_box0, new_box1)
                                if cur_sz > best_sz:
                                    best_box0, best_box1, best_sz = cur_box0, cur_box1, cur_sz

                            return best_box0, best_box1, best_sz

                        # Calculate metrix of region (like total heat map sum) and check if region meet all requirements, otherwise it will be rejected
                        def calc_box_parameters(this_box0, this_box1):
                            l_heat_max = int(np.max(heatMaps_total[this_box0[0]:this_box1[0], this_box0[1]:this_box1[1]]))
                            l_heat_sum = int(np.sum(heatMaps_total[this_box0[0]:this_box1[0], this_box0[1]:this_box1[1]]) / 100)
                            l_heat_sum_frame = int(np.sum(heatMap[this_box0[0]:this_box1[0], this_box0[1]:this_box1[1]]) / 100)

                            l_is_valid = (
                                (l_heat_sum >= self.heatMapTotalMin) and
                                (l_heat_sum_frame >= self.heatMapTotalMinFrame) and
                                ((this_box1[0] - this_box0[0]) > 0) and
                                ((this_box1[1] - this_box0[1]) > 0) and
                                ((float(this_box1[0] - this_box0[0]) / float(this_box1[1] - this_box0[1])) <= self.objHeightWidthLimit))

                            if l_is_valid:
                                for box0_det, box1_det, object_center, object_metrics in detectedObjects_all:
                                    if (this_box0[0] >= box0_det[0]) and (this_box1[0] <= box1_det[0]) and (this_box0[1] >= box0_det[1]) and (this_box1[1] <= box1_det[1]):
                                        l_is_valid = False
                                        break

                            return l_is_valid, l_heat_max, l_heat_sum, l_heat_sum_frame

                        box0, box1, box_sz = get_best_fit_box([0,1,2,3], box0, box1)

                        if box_sz > 0:
                            is_valid, heat_max, heat_sum, heat_sum_frame = calc_box_parameters(box0, box1)
                            if is_valid:
                                is_box_splitted = True
                                while is_box_splitted:
                                    is_box_splitted = False
                                        
                                    box_center_x = (box1[1] + box0[1]) // 2
                                    if (box1[1] - box0[1]) > self.heatMapConvolutionWindowSize:
                                        # Detect object horizontal center
                                        vert_sum = np.sum(heatMaps_total[box0[0]:box1[0], box0[1]:box1[1]], axis=0)
                                        conv_matrix = np.convolve(vert_sum, conv_win, mode = 'valid')
                                        argmax_left = np.argmax(conv_matrix)
                                        object_center_x = argmax_left + box0[1] + conv_win_half

                                        # Object center is displaced, it must be several objects combined in one region, need split
                                        if object_center_x < ((box1[1] + 2 * box0[1]) // 3):
                                            argmax_right = np.argmax(conv_matrix[(len(conv_matrix) // 2):]) + (len(conv_matrix) // 2)
                                            argmin_left = np.argmin(conv_matrix[argmax_left + 1:argmax_right]) + argmax_left + 1
                                            object_split_x = argmin_left + box0[1] + conv_win_half
                                            if ((object_split_x - box0[1]) >= conv_win_half) and ((box1[1] - object_split_x) >= conv_win_half):
                                                box0_new = (box0[0], box0[1])
                                                box1_new = (box1[0], object_split_x)

                                                box0_res = (box0[0], object_split_x)
                                                box1_res = (box1[0], box1[1])

                                                is_valid_l, heat_max_l, heat_sum_l, heat_sum_frame_l = calc_box_parameters(box0_new, box1_new)
                                                is_valid_r, heat_max_r, heat_sum_r, heat_sum_frame_r = calc_box_parameters(box0_res, box1_res)

                                                if is_valid_l and is_valid_r:
                                                    is_box_splitted = True

                                                    box0 = box0_new
                                                    box1 = box1_new
                                                    heat_max, heat_sum, heat_sum_frame = heat_max_l, heat_sum_l, heat_sum_frame_l
                                        elif object_center_x > ((2 * box1[1] + box0[1]) // 3):
                                            argmax_right = argmax_left
                                            argmax_left = np.argmax(conv_matrix[:(len(conv_matrix) // 2)])
                                            argmin_left = np.argmin(conv_matrix[argmax_left + 1:argmax_right]) + argmax_left + 1
                                            object_split_x = argmin_left + box0[1] + conv_win_half
                                            if ((object_split_x - box0[1]) >= conv_win_half) and ((box1[1] - object_split_x) >= conv_win_half):
                                                box0_res = (box0[0], box0[1])
                                                box1_res = (box1[0], object_split_x)

                                                box0_new = (box0[0], object_split_x)
                                                box1_new = (box1[0], box1[1])

                                                is_valid_l, heat_max_l, heat_sum_l, heat_sum_frame_l = calc_box_parameters(box0_new, box1_new)
                                                is_valid_r, heat_max_r, heat_sum_r, heat_sum_frame_r = calc_box_parameters(box0_res, box1_res)

                                                if is_valid_l and is_valid_r:
                                                    is_box_splitted = True

                                                    box0 = box0_new
                                                    box1 = box1_new
                                                    heat_max, heat_sum, heat_sum_frame = heat_max_l, heat_sum_l, heat_sum_frame_l
                                    else:
                                        object_center_x = box_center_x

                                    # Detect object vertical center (used on future steps)
                                    box_center_y = (box1[0] + box0[0]) // 2
                                    if (box1[0] - box0[0]) > self.heatMapConvolutionWindowSize:
                                        horiz_sum = np.sum(heatMaps_total[box0[0]:box1[0], box0[1]:box1[1]], axis=1)
                                        object_center_y = np.argmax(np.convolve(horiz_sum, conv_win, mode = 'valid')) + box0[0] + conv_win_half
                                    else:
                                        object_center_y = box_center_y

                                    detectedObjects += [[box0, box1]]
                                    detectedObjects_all += [[box0, box1, (object_center_y, object_center_x), (heat_max, heat_sum, heat_sum_frame)]]

                                    if is_box_splitted:
                                        box0 = box0_res
                                        box1 = box1_res
                                        heat_max, heat_sum, heat_sum_frame = heat_max_r, heat_sum_r, heat_sum_frame_r

                isObjectDetected = len(detectedObjects) > 0
                if isObjectDetected:
                    # If any objects was detected, remove regions from historical region set which overlap with this object on aproximately 50%
                    # Remain regions will be used in next cycle
                    region_set_new = []
                    for box0, box1 in region_set:
                        is_extra_region = True
                        for box0_det, box1_det in detectedObjects:
                            intersect0 = (max(box0[0], box0_det[0]), max(box0[1], box0_det[1]))
                            intersect1 = (min(box1[0], box1_det[0]), min(box1[1], box1_det[1]))

                            if ((intersect1[0] > intersect0[0]) and
                                (intersect1[1] > intersect0[1]) and
                                ((float((intersect1[0] - intersect0[0]) * (intersect1[1] - intersect0[1])) / float((box1[0] - box0[0]) * (box1[1] - box0[1]))) >= self.heatMapRegionThreshold)):
                                is_extra_region = False
                                break

                        if is_extra_region:
                            region_set_new += [[box0, box1]]

                    region_set = region_set_new

            if len(detectedObjects_all) >= 2:
                # Merge overlapped objects which overlap much enough (near 50%) - one-to-one validation
                is_obj_merged = True
                while is_obj_merged:
                    is_obj_merged = False
                    detectedObjects_all_new = []
                    for box0_det, box1_det, object_center, object_metrics in detectedObjects_all:
                        is_extra_region = True
                        for idx in range(len(detectedObjects_all_new)):
                            l_box0_det, l_box1_det, l_object_center, l_object_metrics = detectedObjects_all_new[idx]
                            intersect0 = (max(l_box0_det[0], box0_det[0]), max(l_box0_det[1], box0_det[1]))
                            intersect1 = (min(l_box1_det[0], box1_det[0]), min(l_box1_det[1], box1_det[1]))

                            if ((intersect1[0] > intersect0[0]) and
                                (intersect1[1] > intersect0[1]) and
                                (((float((intersect1[0] - intersect0[0]) * (intersect1[1] - intersect0[1])) / float((box1_det[0] - box0_det[0]) * (box1_det[1] - box0_det[1]))) >= self.objMergeThreshold) or
                                 ((float((intersect1[0] - intersect0[0]) * (intersect1[1] - intersect0[1])) / float((l_box1_det[0] - l_box0_det[0]) * (l_box1_det[1] - l_box0_det[1]))) >= self.objMergeThreshold))):
                            
                                is_extra_region = False

                                if ((box1_det[0] - box0_det[0]) * (box1_det[1] - box0_det[1])) > ((l_box1_det[0] - l_box0_det[0]) * (l_box1_det[1] - l_box0_det[1])):
                                    detectedObjects_all_new[idx] = [box0_det, box1_det, object_center, object_metrics]
                                    is_obj_merged = True
                                break

                        if is_extra_region:
                            detectedObjects_all_new += [[box0_det, box1_det, object_center, object_metrics]]

                    detectedObjects_all = detectedObjects_all_new

            if len(detectedObjects_all) >= 2:
                # If object overlap with multiple objects enough (near 50%), remove it - one-to-multiple validation
                detectedObjects_all_new = []
                for idx0 in range(len(detectedObjects_all)):
                    box0_det, box1_det, object_center, object_metrics = detectedObjects_all[idx0]
                    box0_sz = (box1_det[0] - box0_det[0]) * (box1_det[1] - box0_det[1])
                    box0_intersect_sz = 0

                    for idx1 in range(len(detectedObjects_all)):
                        if idx0 != idx1:
                            l_box0_det, l_box1_det, l_object_center, l_object_metrics = detectedObjects_all[idx1]
                            intersect0 = (max(l_box0_det[0], box0_det[0]), max(l_box0_det[1], box0_det[1]))
                            intersect1 = (min(l_box1_det[0], box1_det[0]), min(l_box1_det[1], box1_det[1]))

                            intersect_sz = (intersect1[0] - intersect0[0]) * (intersect1[1] - intersect0[1])
                            if intersect_sz > 0:
                                box0_intersect_sz += intersect_sz

                    if (float(box0_intersect_sz) / float(box0_sz)) < self.objMergeThreshold:
                        detectedObjects_all_new += [[box0_det, box1_det, object_center, object_metrics]]

                detectedObjects_all = detectedObjects_all_new
            
            for idx in range(len(self.objectsHistory)):
                obj_idx, is_assigned, box0, box1, hist_center, hist_metrics, size_history, object_label = self.objectsHistory[idx]
                self.objectsHistory[idx] = (obj_idx, False, box0, box1, hist_center, hist_metrics, size_history, object_label)

            # Match objects detected on current frame with objects history (object tracking list) or add new
            for box0_det, box1_det, object_center, object_metrics in detectedObjects_all:
                obj_history_idx = -1
                obj_history_distance = 0

                for idx in range(len(self.objectsHistory)):
                    obj_idx, is_assigned, box0, box1, hist_center, hist_metrics, size_history, object_label = self.objectsHistory[idx]
                    if ((not is_assigned) and
                        (object_center[0] >= box0[0]) and
                        (object_center[0] < box1[0]) and
                        (object_center[1] >= box0[1]) and
                        (object_center[1] < box1[1])):

                        cur_distance = ((object_center[0] - hist_center[0]) ** 2) + ((object_center[1] - hist_center[1]) ** 2)
                        if (obj_history_idx < 0) or (cur_distance < obj_history_distance):
                            obj_history_idx, obj_history_distance = idx, cur_distance
                
                if obj_history_idx >= 0:
                    obj_idx, is_assigned, box0, box1, hist_center, hist_metrics, size_history, object_label = self.objectsHistory[obj_history_idx]
                    self.objectsHistory[obj_history_idx] = (obj_idx, True, box0_det, box1_det, object_center, object_metrics, size_history + [(box0_det, box1_det, object_center)], object_label)
                else:
                    self.objLastLabel += 1
                    self.objectsHistory += [(self.objLastLabel, True, box0_det, box1_det, object_center, object_metrics, [(box0_det, box1_det, object_center)], 0)]
                    
                    if isFirstPrint:
                        isFirstPrint = False
                        print()
                    print("Next object #{} detected. Position: ({}, {}), size: ({}, {})".format(self.objLastLabel, object_center[0], object_center[1], box1_det[0] - box0_det[0], box1_det[1] - box0_det[1]))

            # Some objects from history can leave unmatched. Anyway, try to track object center and detect object. Or consider it is vanished (stop tracking)
            objects_history_new = []
            for obj_idx, is_assigned, box0, box1, hist_center, hist_metrics, size_history, object_label in self.objectsHistory:
                is_object_displayed = False

                if is_assigned:
                    if len(size_history) > self.objsDetFrames:
                        size_history = size_history[1:]

                    avg_size_x = 0
                    avg_size_y = 0
                    avg_cnt = 0

                    for cur_box0, cur_box1, cur_obj_center in size_history:
                        if ((cur_box1[0] - cur_box0[0]) > 0) and ((cur_box1[1] - cur_box0[1]) > 0):
                            last_box0, last_box1 = cur_box0, cur_box1
                            avg_cnt += 1
                            avg_size_x += cur_box1[1] - cur_box0[1]
                            avg_size_y += cur_box1[0] - cur_box0[0]
                    
                    if avg_cnt > 0:
                        if avg_cnt > 1:
                            avg_size_x = int(float(avg_size_x) / float(avg_cnt))
                            avg_size_y = int(float(avg_size_y) / float(avg_cnt))
                            box0 = (
                                int(hist_center[0] - (float(hist_center[0] - last_box0[0]) / float(last_box1[0] - last_box0[0]) * float(avg_size_y))),
                                int(hist_center[1] - (float(hist_center[1] - last_box0[1]) / float(last_box1[1] - last_box0[1]) * float(avg_size_x))))

                            box1 = (box0[0] + avg_size_y, box0[1] + avg_size_x)

                        objects_history_new += [(obj_idx, True, box0, box1, hist_center, hist_metrics, size_history, object_label)]
                        is_object_displayed = True
                else:
                    size_history += [((0,0), (0,0), (0,0))]
                    if len(size_history) > self.objsDetFrames:
                        size_history = size_history[1:]

                    for cur_box0, cur_box1, cur_obj_center in size_history:
                        if ((cur_box1[0] - cur_box0[0]) > 0) and ((cur_box1[1] - cur_box0[1]) > 0):
                            is_object_displayed = True
                            break;

                    if is_object_displayed:
                        det0 = (max(0, hist_center[0] - self.objsDetCrossFrameMaxDist), max(0, hist_center[1] - self.objsDetCrossFrameMaxDist))
                        det1 = (min(heatMap.shape[0], hist_center[0] + self.objsDetCrossFrameMaxDist), min(heatMap.shape[1], hist_center[1] + self.objsDetCrossFrameMaxDist))

                        det_map = heatMap[det0[0]:det1[0], det0[1]:det1[1]]

                        if np.max(det_map) > 0:
                            if (det1[1] - det0[1]) > self.heatMapConvolutionWindowSize:
                                new_center_x = np.argmax(np.convolve(np.sum(det_map, axis=0), conv_win, mode = 'valid')) + det0[1] + conv_win_half
                            else:
                                new_center_x = (det1[1] + det0[1]) // 2

                            if (det1[0] - det0[0]) > self.heatMapConvolutionWindowSize:
                                new_center_y = np.argmax(np.convolve(np.sum(det_map, axis=1), conv_win, mode = 'valid')) + det0[0] + conv_win_half
                            else:
                                new_center_y = (det1[0] + det0[0]) // 2

                            is_object_vanished = False
                            for l_obj_idx, l_is_assigned, l_box0, l_box1, l_hist_center, l_hist_metrics, l_size_history, l_object_label in self.objectsHistory:
                                if l_is_assigned:
                                    if (new_center_y >= l_box0[0]) and (new_center_y < l_box1[0]) and (new_center_x >= l_box0[1]) and (new_center_x < l_box1[1]):
                                        is_object_vanished = True
                                        break

                            if is_object_vanished:
                                is_object_displayed = False
                            else:
                                center_shift = (new_center_y - hist_center[0], new_center_x - hist_center[1])
                                box0 = (min(heatMap.shape[0], max(0, box0[0] + center_shift[0])), min(heatMap.shape[1], max(0, box0[1] + center_shift[1])))
                                box1 = (min(heatMap.shape[0], max(0, box1[0] + center_shift[0])), min(heatMap.shape[1], max(0, box1[1] + center_shift[1])))
                                hist_center = (new_center_y, new_center_x)
                                objects_history_new += [(obj_idx, True, box0, box1, hist_center, hist_metrics, size_history, object_label)]
                        else:
                            is_object_displayed = False

                if is_object_displayed:
                    if self.visualization:
                        heat_max, heat_sum, heat_sum_frame = hist_metrics

                        cv2.rectangle(self.visHeatMapImage, box0[::-1], box1[::-1], (255, 255, 0), 2)

                        cv2.putText(
                            self.visHeatMapImage,
                            '{}, {}, {}'.format(heat_max, heat_sum, heat_sum_frame),
                            (box0[1], box0[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            thickness = 1,
                            lineType = cv2.LINE_AA)
                else:
                    if isFirstPrint:
                        isFirstPrint = False
                        print()
                    print("Object #{} is vanished".format(obj_idx))

            self.objectsHistory = objects_history_new

            # Label tracked objects with labels reusing
            next_obj_label = 1
            used_labels = set()
            for obj_idx, is_assigned, box0, box1, hist_center, hist_metrics, size_history, object_label in self.objectsHistory:
                if object_label > 0:
                    used_labels.add(object_label)

            for idx in range(len(self.objectsHistory)):
                obj_idx, is_assigned, box0, box1, hist_center, hist_metrics, size_history, object_label = self.objectsHistory[idx]
                if is_assigned and (object_label <= 0):
                    while next_obj_label in used_labels:
                        next_obj_label += 1

                    object_label = next_obj_label
                    next_obj_label += 1

                    self.objectsHistory[idx] = (obj_idx, is_assigned, box0, box1, hist_center, hist_metrics, size_history, object_label)

            
            for obj_idx, is_assigned, box0, box1, hist_center, hist_metrics, size_history, object_label in self.objectsHistory:
                if is_assigned and (object_label > 0):
                    annotation_color = self.annotationWindowСolor

                    if object_label <= len(self.annotationWindowСolorList):
                        annotation_color = self.annotationWindowСolorList[object_label - 1]

                    cv2.rectangle(self.visImageAnnotated, box0[::-1], box1[::-1], annotation_color, self.annotationWindowThickness)

                    cv2.putText(
                        self.visImageAnnotated,
                        'Object {}'.format(object_label),
                        (box0[1], box0[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        thickness = 1,
                        lineType = cv2.LINE_AA)

            self.isImageAnnotated = True

        return self.visImageAnnotated
