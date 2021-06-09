from config import *


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


def to_onehot(label, SIZE):
    return torch.eye(SIZE, device=DEVICE, dtype=torch.float32)[label]



if __name__ == "__main__":
    pass