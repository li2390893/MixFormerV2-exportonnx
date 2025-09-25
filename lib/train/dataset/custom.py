import os
import torch
import numpy as np
import pandas
import csv
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings


class Custom(BaseVideoDataset):
    """ Custom dataset with LASOT-like structure.
    
    Dataset should be organized as:
    data/custom/
    └── drone/
        ├── drone-1/
        │   ├── img/
        │   │   ├── 00000001.jpg
        │   │   └── ...
        │   ├── groundtruth.txt
        │   ├── full_occlusion.txt
        │   └── out_of_view.txt
        └── drone-2/
            └── ...
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, vid_ids=None,
                 split=None, data_fraction=None):
        """
        args:
            root - path to the custom dataset.
            image_loader (jpeg4py_loader) - The function to read the images.
            vid_ids - List containing the ids of the videos used for training.
            split - Not used for custom dataset, kept for compatibility.
            data_fraction - Fraction of dataset to be used.
        """
        root = env_settings().custom_dir if root is None else root
        super().__init__('Custom', root, image_loader)

        # Keep a list of all classes
        self.class_list = [f for f in os.listdir(self.root)
                           if os.path.isdir(os.path.join(self.root, f))]
        self.class_to_id = {cls_name: cls_id
                            for cls_id, cls_name in enumerate(self.class_list)}

        self.sequence_list = self._build_sequence_list(vid_ids, split)

        if data_fraction is not None:
            self.sequence_list = random.sample(
                self.sequence_list,
                int(len(self.sequence_list)*data_fraction))

        self.seq_per_class = self._build_class_list()

    def _build_sequence_list(self, vid_ids=None, split=None):
        sequence_list = []
        
        # Get all sequences from all classes
        for class_name in self.class_list:
            class_path = os.path.join(self.root, class_name)
            if os.path.isdir(class_path):
                # Get all video sequences in this class
                sequences = [d for d in os.listdir(class_path)
                             if os.path.isdir(os.path.join(class_path, d))]
                
                if vid_ids is not None:
                    # Filter by video IDs if specified
                    filtered_sequences = []
                    for seq in sequences:
                        # Extract video ID from sequence name
                        # (e.g., "drone-1" -> 1)
                        try:
                            seq_id = int(seq.split('-')[-1])
                            if seq_id in vid_ids:
                                filtered_sequences.append(seq)
                        except (ValueError, IndexError):
                            # If we can't parse ID, include all sequences
                            filtered_sequences.append(seq)
                    sequences = filtered_sequences
                
                sequence_list.extend(sequences)
        
        return sequence_list

    def _build_class_list(self):
        seq_per_class = {}
        for seq_id, seq_name in enumerate(self.sequence_list):
            class_name = seq_name.split('-')[0]
            if class_name in seq_per_class:
                seq_per_class[class_name].append(seq_id)
            else:
                seq_per_class[class_name] = [seq_id]

        return seq_per_class

    def get_name(self):
        return 'custom'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_num_classes(self):
        return len(self.class_list)

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "groundtruth.txt")
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None,
                             dtype=np.float32, na_filter=False,
                             low_memory=False).values
        return torch.tensor(gt)

    def _read_target_visible(self, seq_path):
        # Read full occlusion and out_of_view
        occlusion_file = os.path.join(seq_path, "full_occlusion.txt")
        out_of_view_file = os.path.join(seq_path, "out_of_view.txt")

        try:
            with open(occlusion_file, 'r', newline='') as f:
                occlusion = torch.ByteTensor([int(v) for v in
                                              list(csv.reader(f))[0]])
            with open(out_of_view_file, 'r') as f:
                out_of_view = torch.ByteTensor([int(v) for v in
                                                list(csv.reader(f))[0]])

            target_visible = ~occlusion & ~out_of_view
        except FileNotFoundError:
            # If occlusion files don't exist, assume all frames are visible
            print(f"Warning: Occlusion files not found for {seq_path}, "
                  f"assuming all frames are visible")
            # Get number of frames from groundtruth
            gt_file = os.path.join(seq_path, "groundtruth.txt")
            with open(gt_file, 'r') as f:
                num_frames = len(f.readlines())
            target_visible = torch.ones(num_frames, dtype=torch.bool)

        return target_visible

    def _get_sequence_path(self, seq_id):
        seq_name = self.sequence_list[seq_id]
        class_name = seq_name.split('-')[0]
        
        return os.path.join(self.root, class_name, seq_name)

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = self._read_target_visible(seq_path) & valid.byte()

        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_path(self, seq_path, frame_id):
        # frames start from 1
        return os.path.join(seq_path, 'img', '{:08}.jpg'.format(frame_id+1))

    def _get_frame(self, seq_path, frame_id):
        return self.image_loader(self._get_frame_path(seq_path, frame_id))

    def _get_class(self, seq_path):
        raw_class = seq_path.split('/')[-2]
        return raw_class

    def get_class_name(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        obj_class = self._get_class(seq_path)

        return obj_class

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)

        obj_class = self._get_class(seq_path)
        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({'object_class_name': obj_class,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta
