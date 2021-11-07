import torchvision.models as models
import torch.nn as nn

# uncomment these two lines only if you have problem with ssl
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context  # ssl workaround, not good


class Model1(nn.Module):

    def __init__(self, device):
        """ A wrapper model for extracting features used in Protonet
        A densenet followed by some linear layers
        """
        super().__init__()
        self.model = models.densenet161(pretrained=True)
        # freeze the densenet weights
        for param in self.model.parameters():
            param.requires_grad = False

        classifier_input = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(nn.Linear(classifier_input, 1024),
                                                 nn.ReLU(),
                                                 nn.Linear(1024, 512),
                                                 nn.ReLU(),
                                                 nn.Linear(512, 256),
                                                 nn.ReLU(),
                                                 nn.Linear(256, 128))
        self.model.to(device)

    def forward(self, images):
        """Computes the latent representation of a batch of images.

        Args:
            images (Tensor): batch of images
                shape (num_images, channels, height, width)

        Returns:
            a Tensor containing a batch of latent representations
                shape (num_images, latents)
        """
        return self.model(images)


class Model2(nn.Module):

    def __init__(self, input_len, device):
        """ MLP
        """
        super().__init__()
        self.input_len = input_len
        self.model = nn.Sequential(nn.Linear(self.input_len, 1024),
                                              nn.ReLU(),
                                              nn.Linear(1024, 512),
                                              nn.ReLU(),
                                              nn.Linear(512, 256),
                                              nn.ReLU(),
                                              nn.Linear(256, 128))
        self.model.to(device)

    def forward(self, images):
        """Computes the latent representation of a batch of images.

        Args:
            images (Tensor): batch of images
                shape (num_images, input_len)

        Returns:
            a Tensor containing a batch of latent representations
                shape (num_images, latents)
        """
        return self.model(images)