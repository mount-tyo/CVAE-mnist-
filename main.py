import time

from config import *
from utils import to_onehot
from models import CVAE
import preprocess

# writer = SummaryWriter(log_dir="./logs")


def plot(img, label, deg=None):
    y = img[0].reshape(28,28).cpu().detach().numpy()

    # Plt setting
    fig, ax = plt.subplots()
    ax.imshow(y)
    ax.set_title(f'Generation(label={label}, deg={deg})')
    ax.tick_params(
        labelbottom=False,
        labelleft=False,        
        bottom=False,
        left=False,
    )

    # Plot
    plt.show()
    plt.clf()
    plt.close(fig) 



# Set seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)   
torch.cuda.manual_seed(SEED)




# Train
train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    transform=transforms.ToTensor(),
    download=True,
)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)



model = CVAE(ZDIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

model.train()
start = time.time()
for e in range(NUM_EPOCHS):
    train_loss = 0
    degs = []
    for i, (images, labels) in enumerate(train_loader):
        # Preprocess
        # print(labels.shape)
        # print(len(labels))
        # print(images.shape)
        # index = 255
        # plot(images[index],labels[index])
        for j in range(len(labels)):
            images[j], deg = preprocess.rand_rotate(images[j], labels[j])
            degs.append(deg)
        # print(images.shape)
        # plot(images[index], labels[index], degs[index])
        
        # print(f"type lables = {type(labels)}")
        # print(f"shape lables = {labels.shape}")
        # print(f"lables = {labels}")
        # print("---------------------")
        labels = to_onehot(labels, CLASS_SIZE)
        if i == 234:
            deg_labels = to_onehot(degs[256*i:], DEG_LABEL_SIZE)
        else:
            deg_labels = to_onehot(degs[256*i:256*(i+1)], DEG_LABEL_SIZE)
        # print(f"deg_lables = {len(labels)}")
        # print(f"type lables = {type(labels)}")
        # print(f"shape lables = {labels.shape}")
        # print(f"lables = {labels}")
        
        # Reconstruction images
        # Encode images
        x = images.view(-1, 28*28*1).to(DEVICE)
        mean, lnvar = model.encode(x, labels, deg_labels)
        std = lnvar.exp().sqrt()
        epsilon = torch.randn(ZDIM, device=DEVICE)

        # Decode latent variables
        z = mean + std * epsilon
        y = model.decode(z, labels, deg_labels)

        # Compute loss
        kld = 0.5 * (1 + lnvar - mean.pow(2) - lnvar.exp()).sum(axis=1)
        bce = F.binary_cross_entropy(y, x, reduction='none').sum(axis=1)
        loss = (-1 * kld + bce).mean()

        # Update model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * x.shape[0]

    print(f'epoch: {e + 1} epoch_loss: {train_loss/len(train_dataset)}')
    # writer.add_scalar("Epoch", e+1)
    # writer.add_scalar("Loss", train_loss/len(train_dataset))

# writer.close()
print(f"time = {time.time() - start}")
torch.save(model.state_dict(), f'model_e{NUM_EPOCHS}.pth')