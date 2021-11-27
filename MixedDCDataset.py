"""
Dataloading for MixedDCDataset. Each training task can be a fungi task with probability p and a
flowers task w/ prob 1-p. The fungi tasks labels are generated by deep clusters.
"""

from utils import *
import numpy as np
import torch
from torch.utils.data import dataset, sampler, dataloader
import glob
import os
import scipy.io

NUM_SAMPLES_PER_CLASS = 20

class MixedDCDataset(dataset.Dataset):

    # CODE DUPLICATION MAKES IT LOOK LONG
    _BASE_FUNGI_PATH = 'data/fungi'
    _BASE_FLOWERS_PATH = 'data/vggflowers'

    def __init__(self, num_support, num_query, features):
        # processing fungi
        self._fungus_folders = []
        for fungus_folder in glob.glob(os.path.join(self._BASE_FUNGI_PATH, features, '*')):
            if len(glob.glob(os.path.join(fungus_folder, '*.pt'))) >= NUM_SAMPLES_PER_CLASS:
                self._fungus_folders.append(fungus_folder)
        # shuffle classes
        np.random.default_rng(0).shuffle(self._fungus_folders)
        self.num_fungi_classes = len(self._fungus_folders)
        # TODO: FUNGI WITH PSEUDO LABELS

        # processing flowers
        flowers_labels_mat = scipy.io.loadmat(os.path.join(self._BASE_FLOWERS_PATH, 'imagelabels.mat'))
        self.num_flowers_classes = len(set(flowers_labels_mat['labels'][0]))
        # self.labels_to_images[i] = [list of img_id with label i+1]
        self._flowers_labels_to_images = [[] for _ in range(self.num_flowers_classes)]
        for img_id, class_id in enumerate(flowers_labels_mat['labels'][0]):
            self._flowers_labels_to_images[class_id - 1].append(img_id + 1)
        # shuffle classes
        np.random.default_rng(0).shuffle(self._flowers_labels_to_images)

        # meta learning params
        self._num_support = num_support
        self._num_query = num_query
        self.features = features

    def __getitem__(self, item):
        """Constructs a task.

        At train time (item[0] = True), use len(item[1])=N-D classes of flowers with ground
        truths labels. Additionally, sample D classes of fungi. For each of the D classes,
        sample (K + Q) images. Finally cluster the D(K + Q) images with D cluster center and min
        cluster size of (K + Q) to obtain fungi labels.

        At test time (item[0] = False), use N classes of fungi with ground truth labels.

        Args:
            item (tuple[bool, tuple[int], tuple[int]]): (whether it's a training task,
            class indices for flowers, class indices for fungi)

        Returns:
            images_support (Tensor): task support images
                shape (num_way * num_support, feature_len)
            labels_support (Tensor): task support labels
                shape (num_way * num_support,)
            images_query (Tensor): task query images
                shape (num_way * num_query, feature_len)
            labels_query (Tensor): task query labels
                shape (num_way * num_query,)
        """
        is_train, is_fungi, class_idxs = item
        images_support, images_query = [], []
        labels_support, labels_query = [], []
        suffix = '*.JPG' if self.features == 'images' else '*.pt'
        if is_train:
            if is_fungi:
                # TODO: sample from fungi with pseudo labels
            else:
                # sample from flowers with ground truth labels
                for label, class_idx in enumerate(class_idxs):
                    # randomly choose K + Q samples from self._labels_to_images[class_idx]
                    sampled_img_ids = np.random.default_rng().choice(
                        self._flowers_labels_to_images[class_idx],
                        size=self._num_support + self._num_query,
                        replace=False
                    )
                    if self.features == 'images':
                        images = [load_image(os.path.join(self._BASE_FLOWERS_PATH, 'images',
                                                          'image_{:05d}.jpg'.format(img_id)),
                                             self._transform)
                                  for img_id in sampled_img_ids]
                    else:
                        images = [load_features(os.path.join(self._BASE_FLOWERS_PATH, self.features,
                                                             'image_{:05d}.pt'.format(img_id))) for
                                  img_id in sampled_img_ids]

                    # split sampled examples into support and query
                    images_support.extend(images[:self._num_support])
                    images_query.extend(images[self._num_support:])
                    labels_support.extend([label] * self._num_support)
                    labels_query.extend([label] * self._num_query)
        else:
            # sample from fungi with ground truth labels at val/test time
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
        images_query = torch.stack(images_query)
        labels_query = torch.tensor(labels_query)

        return images_support, labels_support, images_query, labels_query


class MixedTrainSampler(sampler.Sampler):
    """Samples train task specification keys for MixedDCDataset."""

    def __init__(self, fungi_split_idxs, flowers_split_idxs, num_way, fungi_portion, num_tasks):
        """Inits MixedSampler.

        Args:
            split_idxs (range): indices that comprise the
                training/validation/test split
            num_way (int): number of classes per task
            fungi_portion (float): proportion of unsupervised fungi tasks
            num_tasks (int): number of tasks to sample
        """
        super().__init__(None)
        self._fungi_split_idxs = fungi_split_idxs
        self._flowers_split_idxs = flowers_split_idxs
        self._num_way = num_way
        self._fungi_portion = fungi_portion
        self._num_tasks = num_tasks

    def __iter__(self):
        itrs = []
        for _ in range(self._num_tasks):
            x = np.random.uniform()
            if x < self._fungi_portion:
                # TODO: choose from pseudo
                itrs.append((True, True, np.random.default_rng().choice(self._fungi_split_idxs,
                                                                        size=self._num_way,
                                                                        replace=False)))
            else:
                itrs.append((True, False, np.random.default_rng().choice(self._flowers_split_idxs,
                                                                         size=self._num_way,
                                                                         replace=False)))
        return itrs

    def __len__(self):
        return self._num_tasks


class MixedTestSampler(sampler.Sampler):
    """Samples val/test task specification keys for MixedDCDataset."""

    def __init__(self, fungi_split_idxs, num_way, num_tasks):
        """Inits MixedTestSampler.

        Args:
            split_idxs (range): indices that comprise the
                validation/test split
            num_way (int): number of classes per task
            num_tasks (int): number of tasks to sample
        """
        super().__init__(None)
        self._fungi_split_idxs = fungi_split_idxs
        self._num_way = num_way
        self._num_tasks = num_tasks

    def __iter__(self):
        return ((False, True,
            np.random.default_rng().choice(
                self._fungi_split_idxs,
                size=self._num_way,
                replace=False
            )) for _ in range(self._num_tasks)
        )

    def __len__(self):
        return self._num_tasks

def identity(x):
    return x


def get_mixed_dataloader(
        split,
        batch_size,
        num_way,
        num_support,
        num_query,
        fungi_portion,
        num_tasks_per_epoch,
        features
):
    """Returns a dataloader.DataLoader for mixed Fungi and flowers dataset.

    Args:
        split (str): one of 'train', 'val', 'test'
        batch_size (int): number of tasks per batch
        num_way (int): number of classes per task
        num_support (int): number of support examples per class
        num_query (int): number of query examples per class
        num_distract (int): number of fungi classes per task (not using ground truth)
                                num_distracts <= num_way
        num_tasks_per_epoch (int): number of tasks before DataLoader is
            exhausted
        features (str): which feature directory to use
    """
    # TODO: CHANGE FUNGI TRAIN
    dataset = MixedDCDataset(num_support, num_query, features)
    NUM_FUNGI_TRAIN_CLASSES = int(dataset.num_fungi_classes * 0.7)
    NUM_FUNGI_VAL_CLASSES = int(dataset.num_fungi_classes * 0.15)
    NUM_FUNGI_TEST_CLASSES = dataset.num_fungi_classes - NUM_FUNGI_TRAIN_CLASSES - NUM_FUNGI_VAL_CLASSES

    NUM_FLOWERS_TRAIN_CLASSES = int(dataset.num_flowers_classes * 0.7)

    if split == 'train':
        fungi_split_idxs = range(NUM_FUNGI_TRAIN_CLASSES)
        flowers_split_idxs = range(NUM_FLOWERS_TRAIN_CLASSES)
        sampler = MixedTrainSampler(fungi_split_idxs, flowers_split_idxs, num_way, fungi_portion,
                                    num_tasks_per_epoch)
    elif split == 'val':
        fungi_split_idxs = range(
            NUM_FUNGI_TRAIN_CLASSES,
            NUM_FUNGI_TRAIN_CLASSES + NUM_FUNGI_VAL_CLASSES
        )
        sampler = MixedTestSampler(fungi_split_idxs, num_way, num_tasks_per_epoch)
    elif split == 'test':
        fungi_split_idxs = range(
            NUM_FUNGI_TRAIN_CLASSES + NUM_FUNGI_VAL_CLASSES,
            NUM_FUNGI_TRAIN_CLASSES + NUM_FUNGI_VAL_CLASSES + NUM_FUNGI_TEST_CLASSES
        )
        sampler = MixedTestSampler(fungi_split_idxs, num_way, num_tasks_per_epoch)
    else:
        raise ValueError

    return dataloader.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=2,
        collate_fn=identity,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )