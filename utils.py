from config import *



def to_onehot(label):
    return torch.eye(CLASS_SIZE, device=DEVICE, dtype=torch.float32)[label]