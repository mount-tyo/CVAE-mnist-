from config import *



def to_onehot(label, SIZE):
    return torch.eye(SIZE, device=DEVICE, dtype=torch.float32)[label]