from config import *
from utils import to_onehot
from models import CVAE

model = CVAE(ZDIM).to(DEVICE)
model.load_state_dict(torch.load('./model-e100.pth'))

test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    transform=transforms.ToTensor(),
    download=True,
)
target_image, label = list(test_dataset)[48]

x = target_image.view(1, 28*28).to(DEVICE)
with torch.no_grad():
    mean, _ = model.encode(x, to_onehot(label))
z = mean

print(f'z = {z.cpu().detach().numpy().squeeze()}')

os.makedirs(f'img/cvae/generation/fat', exist_ok=True)
for label in range(CLASS_SIZE):
    with torch.no_grad():
        y = model.decode(z, to_onehot(label))
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