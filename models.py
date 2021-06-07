from config import *

class CVAE(nn.Module):
    def __init__(self, zdim):
        super().__init__()
        self._zdim = zdim
        self._in_units = 28 * 28
        hidden_units = 512
        self._encoder = nn.Sequential(
            nn.Linear(self._in_units + CLASS_SIZE, hidden_units),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(inplace=True),
        )
        self._to_mean = nn.Linear(hidden_units, zdim)
        self._to_lnvar = nn.Linear(hidden_units, zdim)
        self._decoder = nn.Sequential(
            nn.Linear(zdim + CLASS_SIZE, hidden_units),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_units, self._in_units),
            nn.Sigmoid()
        )

    def encode(self, x, labels):
        in_ = torch.empty((x.shape[0], self._in_units + CLASS_SIZE), device=DEVICE)
        in_[:, :self._in_units] = x
        in_[:, self._in_units:] = labels
        h = self._encoder(in_)
        mean = self._to_mean(h)
        lnvar = self._to_lnvar(h)
        return mean, lnvar

    def decode(self, z, labels):
        in_ = torch.empty((z.shape[0], self._zdim + CLASS_SIZE), device=DEVICE)
        in_[:, :self._zdim] = z
        in_[:, self._zdim:] = labels
        return self._decoder(in_)

