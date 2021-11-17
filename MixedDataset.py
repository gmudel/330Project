"""
Dataloading for MixedDataset. Each task can have up to N classes from unlabeled distribution (
Fungi). Number of classes of flowers + number of classes of fungi = N. All Fungi labels are
obtained from reclustering. All flowers labels are ground truths.
"""

from utils import *
import numpy as np
import torch
from torch.utils.data import dataset, sampler, dataloader
import glob
import os
import scipy.io
from k_means_constrained import KMeansConstrained

NUM_SAMPLES_PER_CLASS = 20

class MixedDataset(dataset.Dataset):

    # TODO: change these paths if necessary
    # CODE DUPLICATION MAKES IT LOOK LONG
    _BASE_FUNGI_PATH = 'data/fungi'
    _BASE_FLOWERS_PATH = 'data/vggflowers'

    def __init__(self, num_support, num_query):
        # processing fungi
        self._fungus_folders = []
        for fungus_folder in glob.glob(os.path.join(self._BASE_FUNGI_PATH, 'features', '*')):
            if len(glob.glob(os.path.join(fungus_folder, '*.pt'))) >= NUM_SAMPLES_PER_CLASS:
                self._fungus_folders.append(fungus_folder)
        # shuffle classes
        np.random.default_rng(0).shuffle(self._fungus_folders)
        self.num_fungi_classes = len(self._fungus_folders)

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

        # if is_train: len(fungi_idxs) = D, len(flowers_idxs) = N - D
        # else: len(fungi_idxs) = N, len(flowers_idxs) = 0
        is_train, flowers_idxs, fungi_idxs = item
        images_support, images_query = [], []
        labels_support, labels_query = [], []

        if is_train:
            # this is a training task
            # we need (N - D) * (K + Q) image features from flowers
            # use ground truth labels for flowers
            for label, class_idx in enumerate(flowers_idxs):
                # randomly choose K + Q samples from self._flowers_labels_to_images[class_idx]
                sampled_img_ids = np.random.default_rng().choice(
                    self._flowers_labels_to_images[class_idx],
                    size=self._num_support + self._num_query,
                    replace=False
                )
                # TODO: 1. implement load_features
                #       2. change file path once we have the features
                images = [load_features(os.path.join(self._BASE_FLOWERS_PATH, 'features',
                                                     'image_{:05d}.pt'.format(img_id)))
                          for img_id in sampled_img_ids]

                # split sampled examples into support and query
                images_support.extend(images[:self._num_support])
                images_query.extend(images[self._num_support:])
                labels_support.extend([label] * self._num_support)
                labels_query.extend([label] * self._num_query)

            # we need D * (K + Q) image features from fungi
            # use cluster labels for fungi
            fungi_features = []
            for label, class_idx in enumerate(fungi_idxs, start=len(flowers_idxs)):
                # for each class collect (K + Q) image features
                # TODO: change paths to where fungi features are stored
                all_file_paths = glob.glob(
                    os.path.join(self._fungus_folders[class_idx], '*')
                )
                sampled_file_paths = np.random.default_rng().choice(
                    all_file_paths,
                    size=self._num_support + self._num_query,
                    replace=False
                )
                features = [load_features(file_path) for file_path in sampled_file_paths]
                fungi_features.extend(features)
                labels_support.extend([label] * self._num_support)
                labels_query.extend([label] * self._num_query)

            # cluster the fungi features
            fungi_features = torch.stack(fungi_features)
            kmeans = KMeansConstrained(n_clusters=len(fungi_idxs),
                                       size_min=self._num_support + self._num_query, random_state=0)
            predicted_labels = kmeans.fit_predict(fungi_features)
            for cluster in range(kmeans.n_clusters):
                cluster_features = fungi_features[np.where(predicted_labels == cluster)]
                images_support.extend(cluster_features[:self._num_support])
                images_query.extend(cluster_features[self._num_support:])

        else:
            # this is a test task
            # use ground truth labels for fungi
            for label, class_idx in enumerate(fungi_idxs):
                # get a class's examples and sample from them
                # TODO: change paths to where fungi features are stored
                all_file_paths = glob.glob(
                    os.path.join(self._fungus_folders[class_idx], '*')
                )
                sampled_file_paths = np.random.default_rng().choice(
                    all_file_paths,
                    size=self._num_support + self._num_query,
                    replace=False
                )

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


class MixedSampler(sampler.Sampler):
    """Samples task specification keys for MixedDataset."""

    def __init__(self, fungi_split_idxs, flowers_split_idxs, num_way, num_distract, num_tasks,
                 is_train):
        """Inits MixedSampler.

        Args:
            split_idxs (range): indices that comprise the
                training/validation/test split
            num_way (int): number of classes per task
            num_distract (int): number of fungi classes per task (not using ground truth)
                                num_distracts <= num_way
            num_tasks (int): number of tasks to sample
            is_train (bool): whether it's a train sampler
                            test sampler samples ground truth fungi only
        """
        super().__init__(None)
        self._fungi_split_idxs = fungi_split_idxs
        self._flowers_split_idxs = flowers_split_idxs
        self._num_way = num_way
        self._num_distract = num_distract
        self._num_tasks = num_tasks
        self._is_train = is_train

    def __iter__(self):
        if self._is_train:
            return ((self._is_train,
                     np.random.default_rng().choice(self._flowers_split_idxs,
                                                    size=self._num_way - self._num_distract, replace=False),
                     np.random.default_rng().choice(self._fungi_split_idxs, size=self._num_distract, replace=False))
                    for _ in range(self._num_tasks))
        else:
            # only sample from fungi
            return ((self._is_train, (),
                     np.random.default_rng().choice(self._fungi_split_idxs, size=self._num_way, replace=False))
                    for _ in range(self._num_tasks))

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
        num_distract,
        num_tasks_per_epoch
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
    """
    dataset = MixedDataset(num_support, num_query)
    NUM_FUNGI_TRAIN_CLASSES = int(dataset.num_fungi_classes * 0.7)
    NUM_FUNGI_VAL_CLASSES = int(dataset.num_fungi_classes * 0.15)
    NUM_FUNGI_TEST_CLASSES = dataset.num_fungi_classes - NUM_FUNGI_TRAIN_CLASSES - NUM_FUNGI_VAL_CLASSES

    dataset = MixedDataset(num_support, num_query)
    NUM_FLOWERS_TRAIN_CLASSES = int(dataset.num_flowers_classes * 0.7)
    NUM_FLOWERS_VAL_CLASSES = int(dataset.num_flowers_classes * 0.15)
    NUM_FLOWERS_TEST_CLASSES = dataset.num_flowers_classes - NUM_FLOWERS_TRAIN_CLASSES - \
                               NUM_FLOWERS_VAL_CLASSES

    if split == 'train':
        fungi_split_idxs = range(NUM_FUNGI_TRAIN_CLASSES)
        flowers_split_idxs = range(NUM_FLOWERS_TRAIN_CLASSES)
    elif split == 'val':
        fungi_split_idxs = range(
            NUM_FUNGI_TRAIN_CLASSES,
            NUM_FUNGI_TRAIN_CLASSES + NUM_FUNGI_VAL_CLASSES
        )
        flowers_split_idxs = range(
            NUM_FLOWERS_TRAIN_CLASSES,
            NUM_FLOWERS_TRAIN_CLASSES + NUM_FLOWERS_VAL_CLASSES
        )
    elif split == 'test':
        fungi_split_idxs = range(
            NUM_FUNGI_TRAIN_CLASSES + NUM_FUNGI_VAL_CLASSES,
            NUM_FUNGI_TRAIN_CLASSES + NUM_FUNGI_VAL_CLASSES + NUM_FUNGI_TEST_CLASSES
        )
        # no need for flowers at test time
        flowers_split_idxs = range(
            NUM_FLOWERS_TRAIN_CLASSES + NUM_FLOWERS_VAL_CLASSES,
            NUM_FLOWERS_TRAIN_CLASSES + NUM_FLOWERS_VAL_CLASSES + NUM_FLOWERS_TEST_CLASSES
        )
    else:
        raise ValueError

    return dataloader.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=MixedSampler(fungi_split_idxs, flowers_split_idxs, num_way, num_distract,
                             num_tasks_per_epoch, split != 'test'),
        num_workers=2,
        collate_fn=identity,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
