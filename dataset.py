# %%
import torchvision.transforms as transforms
import torch.utils.data as Data
import os
from torch import Tensor
from PIL import Image
# the extensions of image we want to extract
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def has_file_allowed_extension(filename: str, extensions: list[str]) -> bool:
    """
    check if the file has the extensions in the argument
    """
    # transfer to lower case
    filename_lower = filename.lower()
    # return the value
    return any(filename_lower.endswith(extension) for extension in extensions)


def extract_images(root: str, extensions: list[str]) -> list[str]:
    """
    extract the relative paths of image files that meet the conditions

    Args:
        root: the root dir of the image folder
        extensions: the extensions of the image file

    Returns:
        list[str]: the relative paths of the images in the folder
    """
    images = []
    for root, _, fnames in sorted(os.walk(root)):
        for fname in sorted(fnames):
            if has_file_allowed_extension(fname, extensions):
                path = os.path.join(root, fname)
                # append 0 means the label is 0
                images.append((path, 0))
    return images


class ImageFolder(Data.Dataset):
    """
    The class for extracting images from a folder, usually the data is unaligned
    """

    def __init__(self, root: str, extensions: list[str], transform: transforms.Compose = None, target_transform: transforms.Compose = None) -> None:
        # extract images in the root path
        self.images = extract_images(root, extensions)
        # save arguments to member variables
        self.extensions = extensions
        self.transfrom = transform
        self.target_transform = target_transform
        self.root = root

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        """
        override the original function
        """
        path, target = self.images[index]
        # load image
        with open(path, "rb") as f:
            image = Image.open(f)
            image = image.convert("RGB")
        # Transform
        if self.transfrom is not None:
            image = self.transfrom(image)
        # Target Transfrom
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        """
        override the original function
        """
        return len(self.images)

    def __repr__(self) -> str:
        fmt_str = "Dataset " + self.__class__.__name__+"\n"
        fmt_str += f"Number of datapoints: {self.__len__()}\n"
        fmt_str += f"Root location: {self.root}\n"
        fmt_str += f"Transforms (if Any): {self.transfrom.__repr__()}\n"
        fmt_str += f"Target Transforms (if Any): {self.target_transform.__repr__()}\n"
        return fmt_str
