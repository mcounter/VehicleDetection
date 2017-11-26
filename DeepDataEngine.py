import numpy as np
import cv2
import pickle
import csv
import os
import glob
import matplotlib.image as mpimg

from ImageEngine import ImageEngine

class DeepDataEngine:
    """
    Class contains main functions for data management.
    Can be reused for deep learning purpose with minimal changes
    """

    def __init__(
        self,
        set_name, # Name of data set, like 'train' or 'valid'
        storage_dir = './deep_storage', # Folder where data files will be stored
        mem_size = 512 * 1024 * 1024, # Desired maximum size of each file in data storage
        batch_size = 256 # Batch size (used in training and validation process)
        ):
        """
        Initialize class instance
        """

        self.set_name = set_name
        self.storage_dir = storage_dir
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.storage_files = []
        self.storage_file_active = -1
        self.storage_buf_x = None
        self.storage_buf_y = None
        self.storage_size = -1

    def createGenerationPlan(
        filePathVehicles, # Path to vehicles samples
        filePathNonVehicles, # Path to non-vehicles samples
        testSplit = None, # Split test set (0..1)
        groupSplitThreshold = 10000, # Metric threshold to split between groups
        augmentData = False, # Augment data
        normalizeDataGroups = False, # Split data per groups and do normalization
        ):
        """
        Static method, creates generation plan from set of images.
        """

        # Add image in generation plan and calculate it metric for normalization
        def processImageSet(images, img_label, imgEng, img_shape = None, formatFilter = ['.JPG', '.JPEG', '.PNG']):
            local_gen_plan = []

            for image_path in images:
                if os.path.isfile(image_path) and (str(os.path.splitext(image_path)[1]).upper() in formatFilter):
                    img = cv2.imread(image_path)

                    if img_shape is None:
                        img_shape = img.shape

                    cur_img_shape = img.shape
                    if (img_shape[0] != cur_img_shape[0]) or (img_shape[1] != cur_img_shape[1]):
                        img = cv2.resize(img, (img_shape[1], img_shape[0]), interpolation = cv2.INTER_CUBIC)

                    features = imgEng.getImageFeatures(img, visualise = False)
                    img_metric = features[0][2]

                    local_gen_plan += [[(image_path, 0), img_label, img_metric]]
                    if augmentData:
                        local_gen_plan += [[(image_path, 1), img_label, img_metric]]
                    
                    #if len(local_gen_plan) >= np.random.randint(100, 110):
                    #    break

            return local_gen_plan, img_shape

        # Comparator function
        def compare_metric(x1, x2):
            metric1 = x1[2]
            metric2 = x2[2]

            sz1 = len(metric1)
            sz2 = len(metric2)
            sz = min(sz1, sz2)

            for idx in range(sz):
                if metric1[idx] < metric2[idx]:
                    return -1
                elif metric1[idx] > metric2[idx]:
                    return 1

            if sz1 < sz2:
                return -1

            if sz1 > sz2:
                return 1

            return 0

        # Comparator wrapper class
        def cmp_to_key(mycmp):
            class K(object):
                def __init__(self, obj, *args):
                    self.obj = obj
                def __lt__(self, other):
                    return mycmp(self.obj, other.obj) < 0
                def __gt__(self, other):
                    return mycmp(self.obj, other.obj) > 0
                def __eq__(self, other):
                    return mycmp(self.obj, other.obj) == 0
                def __le__(self, other):
                    return mycmp(self.obj, other.obj) <= 0
                def __ge__(self, other):
                    return mycmp(self.obj, other.obj) >= 0
                def __ne__(self, other):
                    return mycmp(self.obj, other.obj) != 0
            return K

        # Split data on groups
        def get_data_groups(gen_plan):
            train_plan = []
            test_plan = []

            if len(gen_plan) > 0:
                groups = []

                gen_plan = sorted(gen_plan, key=cmp_to_key(compare_metric))
                last_group = [gen_plan[0]]
                last_metric = np.array(gen_plan[0][2])

                for idx in range(1, len(gen_plan)):
                    cur_metric = np.array(gen_plan[idx][2])

                    diff = np.mean(np.square(last_metric - cur_metric))
                    if diff >= groupSplitThreshold:
                        groups += [last_group]
                        last_group = []
                    
                    last_group += [gen_plan[idx]]
                    last_metric = cur_metric

                groups += [last_group]

            return groups, len(gen_plan)

        # Normalaze groups
        def normalize_group(data_groups, group_size):
            data_groups_new = []
            for cur_group in data_groups:
                if len(cur_group) > group_size:
                    np.random.shuffle(cur_group)
                    cur_group = cur_group[:group_size]
                else:
                    while len(cur_group) < group_size:
                        diff = group_size - len(cur_group)
                        np.random.shuffle(cur_group)
                        cur_group += cur_group[:diff]

                data_groups_new += [cur_group]
            return data_groups_new

        # Merge groups to one plan
        def groups_to_plan(data_groups):
            gen_plan = []
            for cur_group in data_groups:
                gen_plan += cur_group

            return gen_plan

        # Split test data equally from all groups
        def split_test_data(data_groups, split_num):
            gen_plan_train = []
            gen_plan_test = []

            cur_num = split_num
            np.random.shuffle(data_groups)

            for cur_group in data_groups:
                if len(cur_group) <= cur_num:
                    cur_num -= len(cur_group)
                    gen_plan_test += cur_group
                else:
                    gen_plan_train += cur_group

            return gen_plan_train, gen_plan_test

        img_shape = None

        # Use image engine to retrieve image metric - short form of feature vector
        imgEng = ImageEngine(
            load_setup = False,
            color_space = 'YUV',
            hog_features = True,
            hog_channels = [0],
            hog_orientations = 9,
            hog_pix_per_cell = 16,
            hog_cells_per_block = 1,
            hog_block_norm = 'L2-Hys',
            hog_transform_sqrt = False,
            spatial_features = False,
            histogram_features = True,
            histogram_channels = [1,2],
            histogram_bins = 16)

        vehicle_plan, img_shape = processImageSet(glob.glob(filePathVehicles + '/**/*.*', recursive = True), 1, imgEng, img_shape = img_shape)
        nonvehicle_plan, img_shape = processImageSet(glob.glob(filePathNonVehicles + '/**/*.*', recursive = True), 0, imgEng, img_shape = img_shape)

        vehicle_groups, vehicle_num = get_data_groups(vehicle_plan)
        if normalizeDataGroups:
            vehicle_groups = normalize_group(vehicle_groups, vehicle_num // len(vehicle_groups))

        nonvehicle_groups, nonvehicle_num = get_data_groups(nonvehicle_plan)
        if normalizeDataGroups:
            nonvehicle_groups = normalize_group(nonvehicle_groups, nonvehicle_num // len(nonvehicle_groups))

        if testSplit is None:
            return groups_to_plan(vehicle_groups) + groups_to_plan(nonvehicle_groups)

        vehicle_train, vehicle_test = split_test_data(vehicle_groups, int(vehicle_num * testSplit))
        nonvehicle_train, nonvehicle_test = split_test_data(nonvehicle_groups, int(vehicle_num * testSplit))

        return vehicle_train + nonvehicle_train, vehicle_test + nonvehicle_test

    def _unpickleFromFile(self, file_path):
        """
        Unpickle file with data.
        """

        with open(file_path, mode='rb') as f:
            data_set = pickle.load(f)
    
        X_data, y_data = data_set['features'], data_set['labels']

        assert(len(X_data) == len(y_data))

        return X_data, y_data

    def _pickleToFile(self, file_path, X_data, y_data):
        """
        Pickle file with data.
        """

        with open(file_path, mode='wb') as f:
            data_set = {'features' : X_data, 'labels' : y_data}
            pickle.dump(data_set, f, pickle.HIGHEST_PROTOCOL)

    def _unpickleStorageSize(self):
        """
        Unpickle file with storage size (cached to avoid reload all files to calculate it).
        """

        storage_size = 0

        try:
            with open('{}/{}_ext.ext'.format(self.storage_dir, self.set_name), mode='rb') as f:
                data_set = pickle.load(f)
    
            storage_size = data_set['storage_size']
        except:
            pass

        return storage_size

    def _pickleStorageSize(self, storage_size):
        """
        Unpickle file with storage size (cached to avoid reload all files to calculate it).
        """

        with open('{}/{}_ext.ext'.format(self.storage_dir, self.set_name), mode='wb') as f:
            data_set = {'storage_size' : storage_size}
            pickle.dump(data_set, f, pickle.HIGHEST_PROTOCOL)

    def _loadStorage(self):
        """
        Load information about data storage - size and files with data.
        In this way storage is initialized for reading.
        """

        self.storage_files = []
        self.storage_file_active = -1

        set_file_base_name = self.set_name + '_';

        try:
            os.makedirs(self.storage_dir)
        except:
            pass

        try:
            for file_name in os.listdir(self.storage_dir):
                file_path = self.storage_dir + '/' + file_name
                if (os.path.exists(file_path) and
                    os.path.isfile(file_path) and
                    (str(os.path.splitext(file_path)[1]).upper() in ('.DAT')) and
                    (str(file_name[:len(set_file_base_name)]).upper() == str(set_file_base_name).upper())):
                    
                    self.storage_files += [file_path]

        except:
            pass

        self.storage_size = self._unpickleStorageSize()

    def _delete_storage(self):
        """
        Delete data storage.
        """

        for file_name in self.storage_files:
            try:
                os.remove(file_name)
            except:
                pass

        self.storage_files = []
        self.storage_size = 0
        self._pickleStorageSize(self.storage_size)

    def initStorage(self):
        """
        Initialize storage for reading, call _loadStorage().
        """

        self._loadStorage()

    def createStorage(
        self,
        generation_plan, # Generation plan - python list
        imgEng = None, # Image engine
        savePreProcessedFeatures = True, # If True, feature is vector of image features received by ImageEngine (for classificators), otherwise - raw image (for deep learning)
        override = True): # Indicates that old storage must be deleted if exists. Otherwise it will be augmented with new files.
        """
        Create data storage from generation plan.
        """

        if len(generation_plan) <= 0:
            return

        self._loadStorage()

        if override:
            self._delete_storage()

        # In case storage already has some data, find index of next file.
        file_idx = -1
        for file_name in self.storage_files:
            cur_idx = int(file_name[-10:-4])
            file_idx = max(file_idx, cur_idx)

        file_idx += 1

        # Read first image to determine shape
        image = cv2.imread(generation_plan[0][0][0])
        image_shape = image.shape

        # Create empty X and Y buffers of fixed size. These buffers will be populated with inbound and outbound data and pickled to disk.
        if savePreProcessedFeatures:
            features = imgEng.getImageFeatures(image, visualise = False)
            features_size = features[0][2].shape[0]
            features_bytes = features[0][2].nbytes
            buf_size = int(self.mem_size / features_bytes)

            x_buf = np.zeros((buf_size, features_size), dtype = np.float64)
        else:
            buf_size = int(self.mem_size / ((image_shape[0] * image_shape[1] * image_shape[2]) * 1))
            x_buf = np.zeros((buf_size, image_shape[0], image_shape[1], image_shape[2]), dtype = np.uint8)

        y_buf = np.zeros((buf_size, 1), dtype = np.uint8)

        # Shuffle generation plan to have random distribution across all data files.
        np.random.shuffle(generation_plan)
        
        buf_pos = 0

        for plan_line in generation_plan:
            img_path, img_flip= plan_line[0]
            y_label = plan_line[1]
            
            # Load images from disk
            image = cv2.imread(img_path)
            cur_shape = image.shape
            if (image_shape[0] != cur_shape[0]) or (image_shape[1] != cur_shape[1]):
                image = cv2.resize(image, (image_shape[1], image_shape[0]), interpolation = cv2.INTER_CUBIC)

            if img_flip:
                image = cv2.flip(image, 1)

            if savePreProcessedFeatures:
                features = imgEng.getImageFeatures(image, visualise = False)
                x_buf[buf_pos] = features[0][2]
            else:
                x_buf[buf_pos] = image

            y_buf[buf_pos, 0] = y_label
                        
            buf_pos += 1

            if buf_pos >= buf_size:
                # Pickle buffer to file
                self._pickleToFile('{}/{}_{:0>6}.dat'.format(self.storage_dir, self.set_name, file_idx), x_buf, y_buf)
                self.storage_size += buf_size
                self._pickleStorageSize(self.storage_size)
                file_idx += 1
                buf_pos = 0

        if buf_pos > 0:
            # Pickle non-full last buffer to file
            x_buf = x_buf[:buf_pos]
            y_buf = y_buf[:buf_pos]
            self._pickleToFile('{}/{}_{:0>6}.dat'.format(self.storage_dir, self.set_name, file_idx), x_buf, y_buf)
            self.storage_size += buf_pos
            self._pickleStorageSize(self.storage_size)

        # Initialize storage for reading
        self._loadStorage()

    def _readNextStorageFile(self):
        """
        Read next storage file from disk.
        """

        self.storage_buf_x, self.storage_buf_y = self._unpickleFromFile(self.storage_files[self.storage_file_active])
        self.storage_buf_y = self.storage_buf_y.reshape(-1)

        permutation = np.random.permutation(len(self.storage_buf_x))
        self.storage_buf_x = self.storage_buf_x[permutation]
        self.storage_buf_y = self.storage_buf_y[permutation]

    def initRead(self):
        """
        Initialize data reading - shuffle file list and read next non-empty file.
        """

        np.random.shuffle(self.storage_files)
        self.storage_file_active = 0
        self._readNextStorageFile()

        while len(self.storage_buf_x) <= 0:
            if (self.storage_file_active + 1) < len(self.storage_files):
                self.storage_file_active += 1
                self._readNextStorageFile()
            else:
                break

    def canReadMore(self):
        """
        Determine that data storage is fully read and to read more need be initialized with initRead() function.
        """

        return len(self.storage_buf_x) > 0

    def readNext(self):
        """
        Read next batch for training or validation.
        If end of current file is reached, next file is automatically read from disk and append to current buffer.
        Only one last buffer per epoch can have size less that batch_size.
        """

        x_data = np.array(self.storage_buf_x[:self.batch_size])
        y_data = np.array(self.storage_buf_y[:self.batch_size])

        batch_buf_size = len(x_data)
        self.storage_buf_x = self.storage_buf_x[batch_buf_size:]
        self.storage_buf_y = self.storage_buf_y[batch_buf_size:]

        try_read_next = True

        while try_read_next:
            try_read_next = False

            if len(self.storage_buf_x) <= 0:
                if (self.storage_file_active + 1) < len(self.storage_files):
                    self.storage_file_active += 1
                    self._readNextStorageFile()

                    if len(self.storage_buf_x) > 0:
                        if len(x_data) <= 0:
                            x_data = np.array(self.storage_buf_x[:self.batch_size])
                            y_data = np.array(self.storage_buf_y[:self.batch_size])

                            batch_buf_size = len(x_data)
                            self.storage_buf_x = self.storage_buf_x[batch_buf_size:]
                            self.storage_buf_y = self.storage_buf_y[batch_buf_size:]
                        elif len(x_data) < self.batch_size:
                            size_orig = len(x_data)
                            batch_remain = self.batch_size - size_orig
                            x_data = np.append(x_data, np.array(self.storage_buf_x[:batch_remain]), axis = 0)
                            y_data = np.append(y_data, np.array(self.storage_buf_y[:batch_remain]), axis = 0)

                            batch_buf_size = len(x_data) - size_orig
                            self.storage_buf_x = self.storage_buf_x[batch_buf_size:]
                            self.storage_buf_y = self.storage_buf_y[batch_buf_size:]

                    if len(self.storage_buf_x) <= 0:
                        try_read_next = True

        return x_data, y_data

    def _generator(self):
        """
        Infinite generator compatible with Keras
        """

        while True:
            self.initRead()
            while self.canReadMore():
                yield self.readNext()

    def getGenerator(self):
        """
        Return number of unique batches can be read per epoch and generator instance. Compatible with Keras.
        """

        gen_step_max = self.storage_size // self.batch_size
        if (self.storage_size % self.batch_size) > 0:
            gen_step_max += 1

        return gen_step_max, self._generator()

    def getInOutShape(self):
        """
        Get shape of input and output data.
        """

        self.initRead()
        if self.canReadMore():
            x_data, y_data = self.readNext()
            return x_data.shape[1:], y_data.shape[1:]

        return (), ()

    def readAllData(self):
        """
        Read whole data amount.
        """

        self.initRead()

        x_data_all, y_data_all = self.readNext()

        while self.canReadMore():
            x_data, y_data = self.readNext()

            x_data_all = np.append(x_data_all, x_data, axis = 0)
            y_data_all = np.append(y_data_all, y_data, axis = 0)

        return x_data_all, y_data_all


