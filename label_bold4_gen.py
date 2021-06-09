from config import *
from utils import to_onehot
from models import CVAE

# model = CVAE(ZDIM).to(DEVICE)     # gpu
model = CVAE(ZDIM)                  # cpu

# gpuでのモデル呼び出し
# model.load_state_dict(torch.load('./model-e100.pth'))
# cpu でのモデル呼び出し
model.load_state_dict(torch.load('./saved_models/normal/model-e100.pth', map_location=torch.device('cpu')))

test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    transform=transforms.ToTensor(),
    download=True,
)
target_image, label = list(test_dataset)[48]

# x = target_image.view(1, 28*28).to(DEVICE)
x = target_image.view(1, 28*28)

with torch.no_grad():
    mean, a = model.encode(x, to_onehot(label, CLASS_SIZE))
z = mean

print(f'z = {z.cpu().detach().numpy().squeeze()}')

os.makedirs(f'img/cvae/generation/fat', exist_ok=True)
for label in range(CLASS_SIZE):
    with torch.no_grad():
        y = model.decode(z, to_onehot(label, CLASS_SIZE))
    y = y.reshape(28, 28).cpu().detach().numpy()
    
    fig, ax = plt.subplots()
    ax.imshow(y)
    ax.set_title(f'Generation(label={label})')
    ax.tick_params(
        labelbottom=False,
        labelleft=False,
        bottom=False,
        left=False,
    )
    plt.savefig(f'img/cvae/generation/fat/img{label}')
    plt.close(fig) 