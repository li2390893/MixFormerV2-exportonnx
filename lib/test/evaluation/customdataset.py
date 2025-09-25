import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os


class CustomDataset(BaseDataset):
    """
    Custom dataset with LASOT-like structure for testing

    Dataset should be organized as:
    data/custom/
    └── drone/
        ├── drone-1/
        │   ├── img/
        │   │   ├── 00000001.jpg
        │   │   └── ...
        │   ├── groundtruth.txt
        │   ├── full_occlusion.txt (optional)
        │   └── out_of_view.txt (optional)
        └── drone-2/
            └── ...
    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.custom_path
        self.sequence_list = self._get_sequence_list()
        self.clean_list = self.clean_seq_list()

    def clean_seq_list(self):
        clean_lst = []
        for i in range(len(self.sequence_list)):
            cls, _ = self.sequence_list[i].split('-')
            clean_lst.append(cls)
        return clean_lst

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s)
                            for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        class_name = sequence_name.split('-')[0]
        anno_path = '{}/{}/{}/groundtruth.txt'.format(
            self.base_path, class_name, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',',
                                     dtype=np.float32)

        # Try to load occlusion files
        occlusion_label_path = '{}/{}/{}/full_occlusion.txt'.format(
            self.base_path, class_name, sequence_name)
        out_of_view_label_path = '{}/{}/{}/out_of_view.txt'.format(
            self.base_path, class_name, sequence_name)

        try:
            full_occlusion = load_text(str(occlusion_label_path),
                                     delimiter=',', dtype=np.float32,
                                     backend='numpy')
            out_of_view = load_text(str(out_of_view_label_path),
                                  delimiter=',', dtype=np.float32,
                                  backend='numpy')
            target_visible = np.logical_and(full_occlusion == 0,
                                          out_of_view == 0)
        except (FileNotFoundError, OSError):
            # If occlusion files don't exist, assume all frames are visible
            if ground_truth_rect is not None:
                target_visible = np.ones(ground_truth_rect.shape[0],
                                       dtype=bool)
            else:
                target_visible = np.array([True])

        frames_path = '{}/{}/{}/img'.format(self.base_path, class_name,
                                          sequence_name)

        frames_list = []
        if ground_truth_rect is not None:
            frames_list = ['{}/{:08d}.jpg'.format(frames_path, frame_number)
                          for frame_number in range(1,
                                                  ground_truth_rect.shape[0] + 1)]

        target_class = class_name
        if ground_truth_rect is not None:
            return Sequence(sequence_name, frames_list, 'custom',
                           ground_truth_rect.reshape(-1, 4),
                           object_class=target_class,
                           target_visible=target_visible)
        else:
            return None

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        sequence_list = []
        
        # Get all sequences from all classes
        for class_name in os.listdir(self.base_path):
            class_path = os.path.join(self.base_path, class_name)
            if os.path.isdir(class_path):
                # Get all video sequences in this class
                sequences = [d for d in os.listdir(class_path)
                           if os.path.isdir(os.path.join(class_path, d))]
                sequence_list.extend(sequences)
        
        return sorted(sequence_list)
