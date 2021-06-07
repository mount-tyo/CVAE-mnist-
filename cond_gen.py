from config import *
from models import CVAE
from utils import to_onehot

# Generation data with label '5'
NUM_GENERATION = 100

model = CVAE(ZDIM).to(DEVICE)
model.load_state_dict(torch.load('./model-e100.pth'))

os.makedirs(f'img/cvae/generation/label5/', exist_ok=True)
model.eval()
for i in range(NUM_GENERATION):
    z = torch.randn(ZDIM, device=DEVICE).unsqueeze(dim=0)
    label = torch.tensor([5], device=DEVICE)
    with torch.no_grad():
        y = model.decode(z, to_onehot(label))
    y = y.reshape(28, 28).cpu().detach().numpy()

    # Save image
    fig, ax = plt.subplots()
    ax.imshow(y)
    ax.set_title(f'Generation(label={label.cpu().detach().numpy()[0]})')
    ax.tick_params(
        labelbottom=False,
        labelleft=False,
        bottom=False,
        left=False,
    )
    plt.savefig(f'img/cvae/generation/label5/img{i + 1}')
    plt.close(fig) 
