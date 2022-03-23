# %%
from numpy import ndarray
from torch import Tensor
import yaml
import os
import torch.nn as nn
from glob import glob
import torch as t
import cv2
import numpy as np
from shutil import copyfile
from model import Discriminator

def load_config(config_file_path: str) -> dict:
    """
    load config from a yaml file and returns a dictionary that contains the configuration and print them 
    --------
    params:
        config_file_path: your config file path, normally is the config.yaml, you should not change the name
    """
    # check the file location
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(
            "the file does not exist, please check the path")

    with open(config_file_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), yaml.FullLoader)

    print("###################YOUR SETTINGS###################")
    for key in args.keys():
        print(f"[{key}]".ljust(30, " "), f"--->{args[key]}")

    create_folder(args)
    check_config(args)
    return args


def create_folder(config: dict) -> None:
    """
    Create a directory to store training results
    --------
    params: 
        config, a dictionary that stores the settings
    """
    result_path = os.path.join(config["result_dir"], config["dataset"])
    if not os.path.exists(result_path):
        os.makedirs(os.path.join(result_path, "img"))
        os.makedirs(os.path.join(result_path, "model"))
        os.makedirs(os.path.join(result_path, "test"))
    copyfile("config.yaml", os.path.join(
        config["result_dir"], config["dataset"], "config.yaml"))


def check_config(config: dict) -> bool:
    """
    Check for incorrect parameters
    ------
    """
    assert config["batch_size"] >= 1
    assert config["iteration"] >= 1
    assert config["ch"] >= 1
    assert len(config["dataset"]) > 0


def add_spectral_norm(m: Discriminator) -> Discriminator:
    """
    add Spectral Normalization to the Module
    ------
    parameter:
        m: the module that you want to add Normalization
    returns:
        the module with Spectral Normalization
    """
    for name, layer in m.named_children():

        m.add_module(name, add_spectral_norm(layer))
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        return nn.utils.spectral_norm(m)
    else:
        return m


def find_latest_model(result_path: str, dataset_name: str) -> tuple[int, str]:
    """
    Find the latest model that meets the criteria
    -------
    Parameter:
        result_path: the path to store the results
        dataset_name: the name of the dataset
    Returns:
        the iteration number of the model, -1 if not exists\n
        the relative path of the latest model, None if not exists\n
    """
    model_list = glob(os.path.join(result_path, dataset_name, "model", "*.pt"))

    if not len(model_list) == 0:
        model_list.sort()
        start_iter = int(model_list[-1].split("_")[-1].split("0")[0])

        return start_iter, os.path.join(result_path, dataset_name, "model", model_list[-1])
    else:
        return -1, None


def save_model(genA2B: nn.Module, genB2A: nn.Module, disGA: nn.Module, disGB: nn.Module, disLA: nn.Module, disLB: nn.Module, dir: str, dataset: str, step: int) -> None:
    """
    save the model to the specific directory
    ------
    Parameter:
        genA2B, genB2A, disGA, disGB, disLA, disLB: the module you want to save
        dir: the directory you want to save
        dataset: the dataset name
        step: the training step
    Returns:
        None
    """
    params = {}
    params['genA2B'] = genA2B.state_dict()
    params['genB2A'] = genB2A.state_dict()
    params['disGA'] = disGA.state_dict()
    params['disGB'] = disGB.state_dict()
    params['disLA'] = disLA.state_dict()
    params['disLB'] = disLB.state_dict()
    t.save(params, os.path.join(
        dir, dataset + '_params_%07d.pt' % step))


def tensor2numpy(x: Tensor) -> ndarray:
    """
    transform tensor to numpy.ndarray
    ------
    Args:
        x: the tensor
    Returns:
        the ndarray obj
    """
    return x.detach().cpu().numpy().transpose(1, 2, 0)


def denormalization(x: Tensor) -> Tensor:
    """
    Regularize images
    ------
    Args:
        x: the input image
    Returns:
        the Regularized image
    """
    return x*0.5+0.5


def RGB2BGR(x: ndarray) -> ndarray:
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)


def cam(x, size=256) -> ndarray:
    """
    generate heatmap accroding to the attention module
    -------
    Args:
        x: the attention module's output
        size: the image size
    Returns:
        the heatmap
    """
    # norm
    x = x-np.min(x)
    cam_img = x/np.max(x)
    # to uint8
    cam_img = np.uint8(255*cam_img)
    # resize
    cam_img = cv2.resize(cam_img, (size, size))
    cam_img = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
    return cam_img/255.0


def handle_generated_image(x: Tensor) -> ndarray:
    """
    Combines operations such as regularization and tensor to numpy
    """
    return RGB2BGR(tensor2numpy(denormalization(x)))


def handle_cam_heatmap(x: Tensor, size: int) -> ndarray:
    """
    Combined with cam, tensor to numpy and other operations
    """
    return cam(tensor2numpy(x), size=size)
