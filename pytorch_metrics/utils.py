from torch import Tensor

def is_binary(data: Tensor) -> bool:
    return ((data == 0) | (data == 1)).all()