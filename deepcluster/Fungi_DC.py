"""
Dataloading for Fungi.
"""

from utils import *
import numpy as np
import torch
from torch.utils.data import dataset, sampler, dataloader
import glob
import os

NUM_SAMPLES_PER_CLASS = 20
class FungiDataset(dataset.Dataset):

    _BASE_PATH = 'data/fungi'

    def __init__(self, transform=None, features='images', mode='train'):
        self._fungus_folders_old = []
        suffix = '*.JPG' if features == 'images' else '*.pt'
        for fungus_folder in glob.glob(os.path.join(self._BASE_PATH, features, '*')):
            if len(glob.glob(os.path.join(fungus_folder, suffix))) >= NUM_SAMPLES_PER_CLASS:
                self._fungus_folders_old.append(fungus_folder)
        # shuffle classes
        np.random.default_rng(0).shuffle(self._fungus_folders)
        self.num_classes = len(self._fungus_folders)
        if mode == 'train':
            self.num_classes = 0.7 * self.num_classes

        self._fungus_folders = self._fungus_folders_old[:self.num_classes]
        self._transform = transform if transform else get_default_transform()

    def __getitem__(self, class_idxs):
        suffix = '*.JPG' if self.features == 'images' else '*.pt'
        all_file_paths = glob.glob(
            os.path.join(self._fungus_folders[class_idx], suffix)
        )
        sampled_file_paths = np.random.default_rng().choice(
            all_file_paths,
            size=N,
            replace=False
        )

            if self.features == 'images':
                images = [load_image(file_path, self._transform) for file_path in sampled_file_paths]
            else:
                images = [load_features(file_path) for file_path in sampled_file_paths]

            # split sampled examples into support and query
            images_support.extend(images[:self._num_support])
            images_query.extend(images[self._num_support:])
            labels_support.extend([label] * self._num_support)
            labels_query.extend([label] * self._num_query)

        # aggregate into tensors
        images_support = torch.stack(images_support)  # shape (N*S, C, H, W)
        labels_support = torch.tensor(labels_support)  # shape (N*S)
        if not images_query:
            images_query = torch.tensor(images_query)
        else:
            images_query = torch.stack(images_query)
        labels_query = torch.tensor(labels_query)

        return images_support, labels_support, images_query, labels_query

class UnifLabelSampler(Sampler):
    """Samples elements uniformely accross pseudolabels.
        Args:
            N (int): size of returned iterator.
            images_lists: dict of key (target), value (list of data with this target)
    """

    def __init__(self, N, images_lists):
        self.N = N
        self.images_lists = images_lists
        self.indexes = self.generate_indexes_epoch()

    def generate_indexes_epoch(self):
        nmb_non_empty_clusters = 0
        for i in range(len(self.images_lists)):
            if len(self.images_lists[i]) != 0:
                nmb_non_empty_clusters += 1

        size_per_pseudolabel = int(self.N / nmb_non_empty_clusters) + 1
        res = np.array([])

        for i in range(len(self.images_lists)):
            # skip empty clusters
            if len(self.images_lists[i]) == 0:
                continue
            indexes = np.random.choice(
                self.images_lists[i],
                size_per_pseudolabel,
                replace=(len(self.images_lists[i]) <= size_per_pseudolabel)
            )
            res = np.concatenate((res, indexes))

        np.random.shuffle(res)
        res = list(res.astype('int'))
        if len(res) >= self.N:
            return res[:self.N]
        res += res[: (self.N - len(res))]
        return res

    def __iter__(self):
        return iter(self.indexes)

    def __len__(self):
        return len(self.indexes)