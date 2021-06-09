import cv2
import random
from config import *


def rand_rotate(img, label):
    """Rotate image.

    Args:
        img (tensor): １枚の画像

    Returns:
        img (tensor): 回転後の画像
        deg (int)   : 回転数(ラベル)
    """
    img = img.to('cpu').detach().numpy().copy()
    deg = random.randint(0,3)
    for i in range(deg):
        img[0] = cv2.rotate(img[0], cv2.ROTATE_90_CLOCKWISE)
    img = torch.from_numpy(img.astype(np.float32)).clone()
    # print(f"type label = {type(label)}")
    # print(f"label = {label}")
    # print(f"type deg = {type(deg)}")
    # print(f'deg = {deg}')
    # print(f"label = {label}")
    
    return img, int(deg)



'''
# Load dataset 
test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    transform=transforms.ToTensor(),
    download=True,
)
target_image, label = list(test_dataset)[48]

# Rotate image
num_img = target_image.to('cpu').detach().numpy().copy()

num_img = cv2.rotate(num_img[0], cv2.ROTATE_90_CLOCKWISE)
target_image = torch.from_numpy(num_img.astype(np.float32)).clone()

# Change plt format
y = target_image.reshape(28, 28).cpu().detach().numpy()

# Plt setting
fig, ax = plt.subplots()
ax.imshow(y)
ax.set_title(f'Generation(label={label})')
ax.tick_params(
    labelbottom=False,
    labelleft=False,        
    bottom=False,
    left=False,
)

# Plot
plt.show()
input()
plt.close(fig) 
'''