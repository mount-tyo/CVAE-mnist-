import time

from config import *
from utils import to_onehot
from models import CVAE
import preprocess

# Set Tensorboard
writer = SummaryWriter(log_dir="./logs")

# Set seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)   
torch.cuda.manual_seed(SEED)


# Load Dataset
train_dataset = torchvision.datasets.MNIST(
    root=path_mnist_dataset,
    train=flag_mnist_dataset_train,
    transform=transforms.ToTensor(),
    download=flag_mnist_dataset_dl,
)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=flag_shuffle_trainloader_,
    num_workers=NUM_WORKERS
)
length_trainloader = len(train_loader)

# Set Model
model = CVAE(ZDIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATIO)

# Train Model
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
        if i == (length_trainloader-1):
            deg_labels = to_onehot(degs[BATCH_SIZE*i:], DEG_LABEL_SIZE)
        else:
            deg_labels = to_onehot(degs[BATCH_SIZE*i:BATCH_SIZE*(i+1)], DEG_LABEL_SIZE)
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
    writer.add_scalar("Loss(BCE)/Epoch", train_loss/len(train_dataset), e+1)


writer.close()
print(f"time = {time.time() - start}")
torch.save(model.state_dict(), f'model_e{NUM_EPOCHS}.pth')