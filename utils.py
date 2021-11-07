import imageio
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torch
import numpy as np

def get_default_transform():
    return T.Compose([
        T.ToTensor(),
        T.Resize(256),
        T.CenterCrop(224),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def load_image(file_path, transform=None):
    """Loads and transforms an image.

    Args:
        file_path (str): file path of image

    Returns:
        a Tensor containing image data
            shape (3, 224, 224)
    """
    img = imageio.imread(file_path)
    img = np.ascontiguousarray(img)
    return transform(img)


def load_features(file_path):
    # TODO: read features stored at file_path
    pass


def score(logits, labels):
    """Returns the mean accuracy of a model's predictions on a set of examples.

    Args:
        logits (torch.Tensor): model predicted logits
            shape (examples, classes)
        labels (torch.Tensor): classification labels from 0 to num_classes - 1
            shape (examples,)
    """

    assert logits.dim() == 2
    assert labels.dim() == 1
    assert logits.shape[0] == labels.shape[0]
    y = torch.argmax(logits, dim=-1) == labels
    y = y.type(torch.float)
    return torch.mean(y).item()

# img = load_image('data/vggflowers/images/image_00123.jpg', get_default_transform())
# plt.imshow(img.numpy().transpose((1, 2, 0)))
# plt.show()
