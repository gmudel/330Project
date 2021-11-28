import pandas as pd
import pickle
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

fname = '~/330/330Project/logs/deepcluster/archalexnet.clustering\:Kmeans.num_clusters\:762.batch\:256.lr\:0.05.weight_decay\:-5.0.reassign\:1.0/clusters'
fname = '/home/gmudel/330/330Project/logs/deepcluster/fungi_clustering/clusters'
train_path = "~/330/330Project/data/fungi/train"

d = {
        'filename' : [],
        'pseudo_label' : [],
        'true_label' : [],
        'image_idx' : [],
    }

with open(fname, 'rb') as f:
    b = pickle.load(f)
    final_clusters = b[-1]

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    tra = [transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize]

    dataset = datasets.ImageFolder(train_path, transform=transforms.Compose(tra))
    dataloader = torch.utils.data.DataLoader(dataset,
                                                batch_size=256,
                                                num_workers=2,
                                                pin_memory=True)

    for pseudo_label, image_indices in enumerate(final_clusters):
        for img_idx in image_indices:
            fname, true_label = dataloader.dataset.samples[img_idx]
            chopped_fname = fname[fname.find('train/') + len('train/')::]
            d['image_idx'].append(img_idx)
            d['filename'].append(chopped_fname)
            d['pseudo_label'].append(pseudo_label)
            d['true_label'].append(true_label)
    
    df = pd.DataFrame(d)
    df.to_csv('cluster_assignments.csv', index=False)
