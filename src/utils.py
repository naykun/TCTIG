import pickle
import torch
import collections
import os
from omegaconf import DictConfig, OmegaConf, errors as OCErrors
MAX_SIZE_LIMIT = 65533

def object_to_byte_tensor(obj, max_size=4094):
    """
    Encode Python objects to PyTorch byte tensors
    """
    assert max_size <= MAX_SIZE_LIMIT
    byte_tensor = torch.zeros(max_size, dtype=torch.uint8)

    obj_enc = pickle.dumps(obj)
    obj_size = len(obj_enc)
    if obj_size > max_size:
        raise Exception(
            f"objects too large: object size {obj_size}, max size {max_size}"
        )

    byte_tensor[0] = obj_size // 256
    byte_tensor[1] = obj_size % 256
    byte_tensor[2 : 2 + obj_size] = torch.ByteTensor(list(obj_enc))
    return byte_tensor


def byte_tensor_to_object(byte_tensor, max_size=MAX_SIZE_LIMIT):
    """
    Decode PyTorch byte tensors to Python objects
    """
    assert max_size <= MAX_SIZE_LIMIT

    obj_size = byte_tensor[0].item() * 256 + byte_tensor[1].item()
    obj_enc = bytes(byte_tensor[2 : 2 + obj_size].tolist())
    obj = pickle.loads(obj_enc)
    return obj

def load_yaml(f):
    # Convert to absolute path for loading includes
    mapping = OmegaConf.load(f)
    if mapping is None:
        mapping = OmegaConf.create()

    includes = mapping.get("includes", [])

    if not isinstance(includes, collections.abc.Sequence):
        raise AttributeError(
            "Includes must be a list, {} provided".format(type(includes))
        )

    include_mapping = OmegaConf.create()

    config_root_dir = "./config"

    for include in includes:
        original_include_path = include
        include = os.path.join(config_root_dir, include)

        # If path doesn't exist relative to MMF root, try relative to current file
        if not os.path.exists(include):
            include = os.path.join(os.path.dirname(f), original_include_path)

        current_include_mapping = load_yaml(include)
        include_mapping = OmegaConf.merge(include_mapping, current_include_mapping)

    mapping.pop("includes", None)

    mapping = OmegaConf.merge(include_mapping, mapping)

    return mapping