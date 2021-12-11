'''
Source from
https://github.com/BlackHC/ddu_dirty_mnist/blob/master/ddu_dirty_mnist/dirty_mnist.py
'''

__all__ = ['MNIST_NORMALIZATION', 'AmbiguousMNIST', 'FastMNIST', 'DirtyMNIST']

# Cell

import os
from typing import IO, Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.error import URLError

import torch
from torch.utils import data
from torchvision.datasets.mnist import MNIST, VisionDataset
from torchvision.datasets.utils import download_url, extract_archive, verify_str_arg
from torchvision.transforms import Compose, Normalize, ToTensor
from dataloader.utils import *
import shutil
# Cell

MNIST_NORMALIZATION = Normalize((0.1307,), (0.3081,))

# Cell

# based on torchvision.datasets.mnist.py (https://github.com/pytorch/vision/blob/37eb37a836fbc2c26197dfaf76d2a3f4f39f15df/torchvision/datasets/mnist.py)
class OODMNIST(VisionDataset):
    """
    Ambiguous-MNIST Dataset
    Please cite:
        @article{mukhoti2021deterministic,
          title={Deterministic Neural Networks with Appropriate Inductive Biases Capture Epistemic and Aleatoric Uncertainty},
          author={Mukhoti, Jishnu and Kirsch, Andreas and van Amersfoort, Joost and Torr, Philip HS and Gal, Yarin},
          journal={arXiv preprint arXiv:2102.11582},
          year={2021}
        }
    Args:
        root (string): Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        normalize (bool, optional): Normalize the samples.
        device: Device to use (pass `num_workers=0, pin_memory=False` to the DataLoader for max throughput)
    """

    amirrors = ["http://github.com/BlackHC/ddu_dirty_mnist/releases/download/data-v1.0.0/"]
    mirrors = [
        'http://yann.lecun.com/exdb/mnist/',
        'https://ossci-datasets.s3.amazonaws.com/mnist/',]

    aresources = dict(
        amnist_data=("amnist_samples.pt", "4f7865093b1d28e34019847fab917722"),
        amnist_targets=("amnist_labels.pt", "3bfc055a9f91a76d8d493e8b898c3c95"))
    resources = dict(
        mnist_train_data = ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        mnist_train_targets = ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        mnist_test_data = ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
        mnist_test_targets = ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c")
    )
    eurl = 'https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip'
    emd5 = "58c8d27c78d21e728a6bc7b3cc06412e"
    eresources = ['emnist-letters-train-images-idx3-ubyte.gz','emnist-letters-train-labels-idx1-ubyte.gz'
                ,'emnist-letters-test-images-idx3-ubyte.gz','emnist-letters-test-labels-idx1-ubyte.gz']
    def __init__(
        self,
        root: str,
        *,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        normalize: bool = True,
        noise_stddev=0.0,
        device=None,
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        self.amnist_data = torch.load(self.aresource_path("amnist_data"), map_location=device)
        self.amnist_targets = torch.load(self.aresource_path("amnist_targets"), map_location=device)
        image_file = f"{'train' if self.train else 't10k'}-images-idx3-ubyte"
        label_file = f"{'train' if self.train else 't10k'}-labels-idx1-ubyte"
        
        self.mnist_data = read_image_file(os.path.join(self.data_mfolder, image_file))
        self.mnist_targets = read_label_file(os.path.join(self.data_mfolder, label_file))

        image_file = f"emnist-letters-{'train' if self.train else 'test'}-images-idx3-ubyte"
        label_file = f"emnist-letters-{'train' if self.train else 'test'}-labels-idx1-ubyte"
        self.emnist_data = read_image_file(os.path.join(self.data_efolder, image_file))
        self.emnist_targets = read_label_file(os.path.join(self.data_efolder, label_file))
        if normalize:
            self.amnist_data = self.amnist_data.sub_(0.1307).div_(0.3081)
            self.mnist_data = self.mnist_data.div(255.)
            self.mnist_data = self.mnist_data.sub_(0.1307).div_(0.3081)
            self.emnist_data = self.emnist_data.float().div(255.)
            self.emnist_data = self.emnist_data.sub_(0.1307).div_(0.3081)
        # Each sample has `num_multi_labels` many labels.
        num_multi_labels = self.amnist_targets.shape[1]
        # Flatten the multi-label dataset into a single-label dataset with samples repeated x `num_multi_labels` many times
        self.amnist_data = self.amnist_data.expand(-1, num_multi_labels, 28, 28).reshape(-1, 1, 28, 28)
        self.amnist_targets = self.amnist_targets.reshape(-1)
        if self.train:
            amnist_data_range = slice(None, 60000)
            self.amnist_data = self.amnist_data[amnist_data_range]
            self.amnist_targets = self.amnist_targets[amnist_data_range]
            mnist_data_range = slice(None, 1000)
            self.mnist_data = self.mnist_data[mnist_data_range]
            self.mnist_targets = self.mnist_targets[mnist_data_range]
            emnist_data_range = slice(None, 1000)
            self.emnist_data = self.emnist_data[emnist_data_range]
            self.emnist_targets = self.emnist_targets[emnist_data_range].fill_(10)
            self.data = torch.cat((self.mnist_data.unsqueeze(1),self.amnist_data,self.emnist_data.unsqueeze(1)),dim=0)
            self.targets = torch.cat((self.mnist_targets,self.amnist_targets,self.emnist_targets),dim=0)
        else:
            self.data = self.mnist_data
            self.targets = self.mnist_targets
        if noise_stddev > 0.0:
            self.data += torch.randn_like(self.data) * noise_stddev


    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    @property
    def data_afolder(self) -> str:
        return os.path.join(self.root,'Dirty_MNIST')
    @property
    def data_mfolder(self) -> str:
        return os.path.join(self.root,'MNIST')

    @property
    def data_efolder(self) -> str:
        return os.path.join(self.root,'EMNIST/gzip/')

    def resource_path(self, name):
        return os.path.join(self.root,'MNIST' ,self.resources[name][0])

    def aresource_path(self, name):
        return os.path.join(self.root,'Dirty_MNIST',self.aresources[name][0])

    def eresource_path(self, name):
        return os.path.join(self.root,'EMNIST/gzip',self.eresources[name])

    def _check_exists(self) -> bool:
        a = all(os.path.exists(self.resource_path(name)) for name in self.resources)
        b = all(os.path.exists(self.aresource_path(name)) for name in self.aresources)
        c = all(os.path.exists(self.eresource_path(name)) for name in range(len(self.eresources)))
        return a & b & c

    def download(self) -> None:
        """Download the data if it doesn't exist in data_folder already."""

        if self._check_exists():
            return

        os.makedirs(self.data_afolder, exist_ok=True)
        os.makedirs(self.data_mfolder, exist_ok=True)
        os.makedirs(self.data_efolder, exist_ok=True)
        # download files
        for filename, md5 in self.aresources.values():
            for mirror in self.amirrors:
                url = "{}{}".format(mirror, filename)
                try:
                    print("Downloading {}".format(url))
                    download_url(url, root=self.data_afolder, filename=filename, md5=md5)
                except URLError as error:
                    print("Failed to download (trying next):\n{}".format(error))
                    continue
                except:
                    raise
                finally:
                    print()
                break
            else:
                raise RuntimeError("Error downloading {}".format(filename))
        # download files
        for filename, md5 in self.resources.values():
            for mirror in self.mirrors:
                url = "{}{}".format(mirror, filename)
                try:
                    print("Downloading {}".format(url))
                    download_url(
                        url, root=self.root + '/MNIST',
                        filename=filename,
                        md5=md5
                    )
                    archive = os.path.join(self.root + '/MNIST', filename)
                    extract_archive(archive)
                except URLError as error:
                    print(
                        "Failed to download (trying next):\n{}".format(error)
                    )
                    continue
                finally:
                    print()
                break
            else:
                raise RuntimeError("Error downloading {}".format(filename))
        print("Downloding EMNIST")
        download_url(self.eurl, root=self.root + '/EMNIST',md5=self.emd5)
        for f in self.eresources:
            archive = os.path.join(self.root + '/EMNIST/gzip', f)
            extract_archive(archive)
    
if __name__ == '__main__':
    dataset = DirtyMNIST(root= '../dataset/',download=True)
    loader =  torch.utils.data.DataLoader(dataset,batch_size=10)
    for a,b in loader:
        pass