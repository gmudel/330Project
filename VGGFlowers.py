"""Dataloading for VGG Flowers. There are a
minimum of 40 images for each category. """

import os.path
import scipy.io
from utils import *
import numpy as np
import torch
from torch.utils.data import dataset, sampler, dataloader

class VGGFlowersDataset(dataset.Dataset):

    _BASE_PATH = 'data/vggflowers'

    def __init__(self, num_support, num_query, transform=None, labels_file='imagelabels.mat'):
        labels_mat = scipy.io.loadmat(os.path.join(self._BASE_PATH, labels_file))
        num_classes = len(set(labels_mat['labels'][0]))

        # self.labels_to_images[i] = [list of img_id with label i+1]
        self._labels_to_images = [[] for _ in range(num_classes)]
        for img_id, class_id in enumerate(labels_mat['labels'][0]):
            self._labels_to_images[class_id - 1].append(img_id + 1)

        # shuffle classes
        np.random.default_rng(0).shuffle(self._labels_to_images)
        self.num_classes = len(self._labels_to_images)

        self._transform = transform if transform else get_default_transform()
        self._num_support = num_support
        self._num_query = num_query

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

        for label, class_idx in enumerate(class_idxs):
            # randomly choose K + Q samples from self._labels_to_images[class_idx]
            sampled_img_ids = np.random.default_rng().choice(
                self._labels_to_images[class_idx],
                size=self._num_support + self._num_query,
                replace=False
            )
            images = [load_image(os.path.join(self._BASE_PATH, 'images',
                                              'image_{:05d}.jpg'.format(img_id)),
                      self._transform)
                      for img_id in sampled_img_ids]

            # split sampled examples into support and query
            images_support.extend(images[:self._num_support])
            images_query.extend(images[self._num_support:])
            labels_support.extend([label] * self._num_support)
            labels_query.extend([label] * self._num_query)

        # aggregate into tensors
        images_support = torch.stack(images_support)  # shape (N*S, C, H, W)
        labels_support = torch.tensor(labels_support)  # shape (N*S)
        images_query = torch.stack(images_query)
        labels_query = torch.tensor(labels_query)

        return images_support, labels_support, images_query, labels_query


class VGGFlowersSampler(sampler.Sampler):
    """Samples task specification keys for VGGFlowersDataset."""

    def __init__(self, split_idxs, num_way, num_tasks):
        """Inits VGGFlowersSampler.

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


def get_vggflowers_dataloader(
        split,
        batch_size,
        num_way,
        num_support,
        num_query,
        num_tasks_per_epoch
):
    """Returns a dataloader.DataLoader for VGGFlowers.

    Args:
        split (str): one of 'train', 'val', 'test'
        batch_size (int): number of tasks per batch
        num_way (int): number of classes per task
        num_support (int): number of support examples per class
        num_query (int): number of query examples per class
        num_tasks_per_epoch (int): number of tasks before DataLoader is
            exhausted
    """
    flowers_dataset = VGGFlowersDataset(num_support, num_query)
    NUM_TRAIN_CLASSES = int(flowers_dataset.num_classes * 0.7)
    NUM_VAL_CLASSES = int(flowers_dataset.num_classes * 0.15)
    NUM_TEST_CLASSES = flowers_dataset.num_classes - NUM_TRAIN_CLASSES - NUM_VAL_CLASSES

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
        dataset=flowers_dataset,
        batch_size=batch_size,
        sampler=VGGFlowersSampler(split_idxs, num_way, num_tasks_per_epoch),
        num_workers=2,
        collate_fn=identity,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
