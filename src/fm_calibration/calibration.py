import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.isotonic import IsotonicRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import sys

from metrics import mc_brier_score


class IsotonicRegressionMulticlass:
    def fit(self, outputs, features, Y, **kwargs):
        n_classes = outputs.shape[-1]

        Y_onehot = np.arange(n_classes) == Y[:, None]
        Y_onehot = Y_onehot.flatten()

        self.scaler = IsotonicRegression(
            out_of_bounds="clip", y_min=np.log(1e-7), y_max=0
        )
        self.scaler.fit(np.log(1 - outputs).flatten(), np.log(Y_onehot + 1e-7))

    def forward(self, outputs, features, Y, **kwargs):
        n_classes = outputs.shape[-1]
        res = np.log(
            1
            - np.exp(
                self.scaler.transform(np.log(1 - outputs).flatten()).reshape(
                    (-1, n_classes)
                )
            )
        )
        res = torch.softmax(torch.Tensor(res), dim=-1).numpy()
        return res


def irm_fitter(confidences, trues_in):
    irm = IsotonicRegressionMulticlass()
    irm.fit(confidences, None, trues_in)
    return lambda confidences: irm.forward(confidences, None, None)


def temperature_scaling_grid(
    confidences, trues_in, loss=0.0, loss_center=0.0, weights=None, brier=False
):
    offset = np.random.random() * 6 / 100 - 3 / 100
    ts = np.logspace(-4 + offset, 2 + offset, 100, endpoint=False)

    best_t = None

    if len(trues_in.shape) == 1:
        trues_in = np.eye(confidences.shape[1])[trues_in].astype(float)

    scores = []

    for batch_ts in ts.reshape(-1, 100):
        y_val_calib = np.log(confidences)[None, :, :] * batch_ts[:, None, None]
        trues = trues_in[None, :, :] * np.ones_like(batch_ts)[:, None, None]
        y_val_calib = y_val_calib.reshape(-1, y_val_calib.shape[2])
        trues = trues.reshape(-1, trues.shape[2])
        if brier:
            score = mc_brier_score(
                F.softmax(torch.tensor(y_val_calib), dim=-1).numpy(),
                trues,
                reduce=False,
            )
            score = torch.tensor(score)
        else:
            score = F.cross_entropy(
                torch.tensor(y_val_calib), torch.tensor(trues), reduction="none"
            )

        if weights is not None:
            score = (score.reshape(len(batch_ts), -1) * weights).mean(axis=1)
        else:
            score = score.reshape(len(batch_ts), -1).mean(axis=1)

        score += loss * (torch.log(torch.tensor(batch_ts)) - loss_center) ** 2

        scores.append(score)

    scores = torch.concat(scores)
    best_t = ts[torch.argmin(scores).item()]

    return best_t


def temperature_scaling_fitter(confidences, trues_in):
    t = temperature_scaling_grid(confidences, trues_in)
    return lambda confidences: torch.softmax(
        t * torch.log(torch.Tensor(confidences)), -1
    ).numpy()


sys.path.append("/mnt/data_1/home_tsteam/malonso/group_calibration")
from methods.mix_calibration import calibrate as ets_calibrate


def ets_fitter(confidences, trues):
    return lambda conf_test: ets_calibrate(
        "ets",
        torch.log(torch.Tensor(confidences)),
        torch.Tensor(trues).long(),
        torch.log(torch.Tensor(conf_test)),
    )["prob"].numpy()


class ClusteredTemperature:
    def __init__(
        self, nb_ensembles=20, nb_clusters=7, dim_frac=0.1, pca_transform=False
    ):
        self.N = nb_ensembles
        self.K = nb_clusters
        self.dim_frac = dim_frac
        self.pca_transform = pca_transform

    def fit(
        self,
        x,
        y_proba,
        y_true,
        x_clustering=None,
        seed=1234,
        calib_fn_fitter=temperature_scaling_fitter,
    ):
        self._kmeans = []
        self._subsets = []
        self._calib_fns = []
        np.random.seed(seed)
        x_clustering = x if x_clustering is None else x_clustering

        if self.pca_transform:
            self._pca = PCA(n_components=x.shape[1]).fit(x_clustering)
            x_clustering = self._pca.transform(x_clustering)

        for n in range(self.N):
            # n_clusters = np.random.choice([1, 2, 3, 5, 8, 12, 20])
            n_clusters = self.K

            if self.dim_frac is None:
                dim_subset = np.arange(x.shape[1])
            elif self.pca_transform:
                dim_subset = np.random.multinomial(
                    int(x.shape[1] * self.dim_frac),
                    pvals=self._pca.explained_variance_ratio_,
                )
                dim_subset = np.array(
                    [i for i in range(len(dim_subset)) for j in range(dim_subset[i])]
                )
            else:
                if isinstance(self.dim_frac, int):
                    dim_subset = np.random.choice(x.shape[1], size=self.dim_frac)
                else:
                    dim_subset = np.random.choice(
                        x.shape[1], size=int(x.shape[1] * self.dim_frac)
                    )

            bootstrap = np.random.choice(len(x_clustering), size=20, replace=True)
            kmeans = KMeans(n_clusters=n_clusters, n_init="auto").fit(
                x_clustering[bootstrap][:, dim_subset]
            )

            if self.pca_transform:
                labels = kmeans.predict(self._pca.transform(x)[:, dim_subset])
            else:
                labels = kmeans.predict(x[:, dim_subset])

            for k in range(n_clusters):
                if sum(labels == k) == 0:
                    continue

                calib_fn = calib_fn_fitter(y_proba[labels == k], y_true[labels == k])

            self._subsets.append(dim_subset)
            self._kmeans.append(kmeans)
            self._calib_fns.append(calib_fn)

    def predict(self, x, y_proba, return_t=False):
        avg_probas = np.zeros_like(y_proba)
        for n, (kmeans, calib_fn, dim_subset) in enumerate(
            zip(self._kmeans, self._calib_fns, self._subsets)
        ):
            if self.pca_transform:
                labels = kmeans.predict(self._pca.transform(x)[:, dim_subset])
            else:
                labels = kmeans.predict(x[:, dim_subset])

            for k in range(kmeans.n_clusters):
                cal_proba = calib_fn(y_proba[labels == k])
                avg_probas[labels == k] = avg_probas[labels == k] * n / (
                    n + 1
                ) + cal_proba / (n + 1)

        return (
            None,
            avg_probas,
        )


class LocalCalib:
    def __init__(self, gamma=0.4, sigma=1, n_bins=15, dim_reduce=32):
        self.gamma = gamma
        self.sigma = sigma
        self.n_bins = n_bins
        self.bins = None
        self.dim_reduce = dim_reduce

    def binned(self, conf):
        # if self.bins is None:
        #     self.bins = np.quantile(conf, np.linspace(0.,1., self.n_bins + 1))

        # return np.digitize(conf, self.bins)

        return np.digitize(conf, np.linspace(0.0, 1.0, self.n_bins + 1))

        # conf = np.clip(conf, 1e-7, 1 - 1e-7)
        # return np.log(conf) - np.log(1-conf)

    def fit(
        self,
        x,
        y_proba,
        y_true,
        x_clustering=None,
        seed=1234,
        loss=0.3,
        loss_center=None,
    ):
        scaler_pca = Pipeline(
            [("scaler", StandardScaler()), ("pca", PCA(n_components=self.dim_reduce))]
        )
        self._pca = scaler_pca.fit(x_clustering)
        self._X = self._pca.transform(x)
        self._conf = self.binned(y_proba[:, 1:])
        self._acc = y_true[:, None]
        # self._acc = np.eye(y_proba.shape[1])[y_true]

    def predict(self, x, y_proba, return_t=False):
        x_pca = self._pca.transform(x)
        kernel_x = np.exp(
            -np.abs(x_pca[:, None, :] - self._X[None, :, :]).mean(axis=-1) / self.gamma
        )

        weights = (
            kernel_x[:, :, None, None]
            # * (self.binned(y_proba)[:, None, :, None] == self._conf[None, :, None, :])
            * np.exp(
                -np.abs(
                    self.binned(y_proba[:, 1:])[:, None, :, None]
                    - self._conf[None, :, None, :]
                )
                / self.sigma
            )
        )

        avg_probas = np.nan_to_num(
            (weights * self._acc[None, :, None, :]).sum(axis=(1, 3))
            / weights.sum(axis=(1, 3)),
            nan=0.0,
        )

        avg_probas = np.clip(avg_probas, 0, 1)
        avg_probas = np.concat([1 - avg_probas, avg_probas], axis=1)

        return (
            None,
            avg_probas,
        )
