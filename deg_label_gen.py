from tqdm import tqdm

from config import *
from models import CVAE
from utils import to_onehot

# Generation data with label '5'
NUM_GENERATION = 100
DEG = 3
LABEL = 5

model = CVAE(ZDIM).to(DEVICE)
model.load_state_dict(torch.load('./saved_models/zdim32/nor+deg_label/model_e500.pth'))

print(model)

os.makedirs(f'img/cvae/generation/deg{DEG}_label{LABEL}/', exist_ok=True)
model.eval()
for i in tqdm(range(NUM_GENERATION)):
    z = torch.randn(ZDIM, device=DEVICE).unsqueeze(dim=0)
    label = torch.tensor([LABEL], device=DEVICE)
    deg = torch.tensor([DEG], device=DEVICE)
    with torch.no_grad():
        y = model.decode(z, to_onehot(label, CLASS_SIZE), to_onehot(deg, DEG_LABEL_SIZE))
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
    plt.savefig(f'img/cvae/generation/deg{DEG}_label{LABEL}/img{i + 1}')
    plt.close(fig) 
