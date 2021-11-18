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

    def __init__(self, num_support, num_query, features, transform=None):
        self._fungus_folders = []
        suffix = '*.JPG' if features == 'images' else '*.pt'
        for fungus_folder in glob.glob(os.path.join(self._BASE_PATH, features, '*')):
            if len(glob.glob(os.path.join(fungus_folder, suffix))) >= NUM_SAMPLES_PER_CLASS:
                self._fungus_folders.append(fungus_folder)
        # shuffle classes
        np.random.default_rng(0).shuffle(self._fungus_folders)
        self.num_classes = len(self._fungus_folders)

        self._transform = transform if transform else get_default_transform()
        self._num_support = num_support
        self._num_query = num_query
        self.features = features

    def __getitem__(self, class_idxs):
        """Constructs a task.

        Data for each class is sampled uniformly at random without replacement.
        The ordering of the labels corresponds to that of class_idxs.

        Args:
            class_idxs (tuple[int]): class indices that comprise the task

        Returns:
            images_support (Tensor): task support images
                shape (num_way * num_support, channels, height, width)
            labels_support (Tensor): task support labels
                shape (num_way * num_support,)
            images_query (Tensor): task query images
                shape (num_way * num_query, channels, height, width)
            labels_query (Tensor): task query labels
                shape (num_way * num_query,)
        """
        images_support, images_query = [], []
        labels_support, labels_query = [], []

        suffix = '*.JPG' if self.features == 'images' else '*.pt'
        for label, class_idx in enumerate(class_idxs):
            # get a class's examples and sample from them
            all_file_paths = glob.glob(
                os.path.join(self._fungus_folders[class_idx], suffix)
            )
            sampled_file_paths = np.random.default_rng().choice(
                all_file_paths,
                size=self._num_support + self._num_query,
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


class FungiSampler(sampler.Sampler):
    """Samples task specification keys for FungiDataset."""

    def __init__(self, split_idxs, num_way, num_tasks):
        """Inits FungiSampler.

        Args:
            split_idxs (range): indices that comprise the
                training/validation/test split
            num_way (int): number of classes per task
            num_tasks (int): number of tasks to sample
        """
        super().__init__(None)
        self._split_idxs = split_idxs
        self._num_way = num_way
        self._num_tasks = num_tasks

    def __iter__(self):
        return (
            np.random.default_rng().choice(
                self._split_idxs,
                size=self._num_way,
                replace=False
            ) for _ in range(self._num_tasks)
        )

    def __len__(self):
        return self._num_tasks


def identity(x):
    return x


def get_fungi_dataloader(
        split,
        batch_size,
        num_way,
        num_support,
        num_query,
        num_tasks_per_epoch,
        features
):
    """Returns a dataloader.DataLoader for Fungi.

    Args:
        split (str): one of 'train', 'val', 'test'
        batch_size (int): number of tasks per batch
        num_way (int): number of classes per task
        num_support (int): number of support examples per class
        num_query (int): number of query examples per class
        num_tasks_per_epoch (int): number of tasks before DataLoader is
            exhausted
        features (str): which feature directory to use
    """
    fungi_dataset = FungiDataset(num_support, num_query, features)
    NUM_TRAIN_CLASSES = int(fungi_dataset.num_classes * 0.7)
    NUM_VAL_CLASSES = int(fungi_dataset.num_classes * 0.15)
    NUM_TEST_CLASSES = fungi_dataset.num_classes - NUM_TRAIN_CLASSES - NUM_VAL_CLASSES

    if split == 'train':
        split_idxs = range(NUM_TRAIN_CLASSES)
    elif split == 'val':
        split_idxs = range(
            NUM_TRAIN_CLASSES,
            NUM_TRAIN_CLASSES + NUM_VAL_CLASSES
        )
    elif split == 'test':
        split_idxs = range(
            NUM_TRAIN_CLASSES + NUM_VAL_CLASSES,
            NUM_TRAIN_CLASSES + NUM_VAL_CLASSES + NUM_TEST_CLASSES
        )
    else:
        raise ValueError

    return dataloader.DataLoader(
        dataset=fungi_dataset,
        batch_size=batch_size,
        sampler=FungiSampler(split_idxs, num_way, num_tasks_per_epoch),
        num_workers=2,
        collate_fn=identity,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
