import numpy as np
from scipy.stats import invgamma
import torch
import torch.nn.functional as F


def get_data_k(
    n=2000,
    d=128,
    seed=3,
    nb_clusters=None,
    alpha=2,
    nb_classes=2,
    std_groups=1,
    return_p=False,
    a_noise_decision=2,
    w_oracle=None,
    noise_decision=0.1,
    bias_decision=0.03,
    reduce=False,
    debias=False,
    pos=None,
):
    nb_classes = nb_classes
    alpha = alpha
    np.random.seed(seed)

    x = np.zeros((n, d))
    cluster = np.zeros(n)
    means = np.zeros((n, d))
    stds = np.zeros((n, d))

    if nb_clusters is None:
        for i in range(n):
            if np.random.rand() <= alpha / (i + alpha):
                mean = np.random.normal(0, 1, d) if pos is None else pos
                std = std_groups * invgamma.rvs(size=1, a=10)
                cluster[i] = i
                x[i] = np.random.normal(mean, std)
                means[i] = mean
                stds[i] = std
            else:
                j = np.random.randint(i)
                x[i] = np.random.normal(means[j], stds[j])
                cluster[i] = cluster[j]
                means[i] = means[j]
                stds[i] = stds[j]

    else:
        if alpha is not None:
            cluster_ps = np.random.dirichlet(alpha * np.ones(nb_clusters))
        else:
            cluster_ps = np.ones(nb_clusters) / nb_clusters

        for i in range(nb_clusters):
            mean = np.random.normal(0, 1, d) if pos is None else pos
            std = std_groups * invgamma.rvs(size=1, a=10)
            cluster[i] = i
            x[i] = np.random.normal(mean, std)
            means[i] = mean
            stds[i] = std

        for i in range(nb_clusters, n):
            j = 0 if nb_clusters == 1 else np.random.multinomial(1, cluster_ps).argmax()
            x[i] = np.random.normal(means[j], stds[j])
            cluster[i] = cluster[j]
            means[i] = means[j]
            stds[i] = stds[j]

    if w_oracle is None:
        w_oracle = np.random.randn(d, nb_classes)
        w_oracle /= np.linalg.norm(w_oracle)

    y = x @ w_oracle
    if debias:
        y = y - y.mean(axis=0)
    if reduce:
        y /= y.std(axis=0)

    noise_decision = noise_decision * invgamma.rvs(size=1, a=a_noise_decision)
    bias_decision = bias_decision * np.random.normal(0.0, 1.0, (y.shape[1],))

    ps = F.softmax((torch.tensor(y) + bias_decision[None, :]) / noise_decision).numpy()
    y = np.array([np.random.multinomial(1, p).argmax() for p in ps])

    if return_p:
        return x, y, ps
    else:
        return x, y


def get_data(
    K=1,
    n=2000,
    seed=3,
    global_w=False,
    w_reg=0.8,
    d=16,
    nb_classes=2,
    alpha_K=None,
    **kwargs
):
    """
    Get data for the experiment
    """
    w_oracle = np.random.randn(K, d, nb_classes)
    w_oracle /= np.linalg.norm(w_oracle)

    if isinstance(global_w, np.ndarray):
        w_oracle_mean = global_w
    else:
        w_oracle_mean = w_oracle.mean(axis=0)
    w_oracle_mean /= np.linalg.norm(w_oracle_mean)

    w_oracle = (1 - w_reg) * w_oracle + w_reg * w_oracle_mean[None, :, :]
    w_oracle /= np.linalg.norm(w_oracle)

    if isinstance(K, list):
        K, pos = len(K), K
    else:
        pos = [None for i in range(K)]

    if alpha_K is not None:
        p_K = np.random.dirichlet(alpha_K * np.ones(K))
    else:
        p_K = np.ones(K) / K
    n_K = np.random.multinomial(n, pvals=p_K)

    res = []
    for k, n_k in enumerate(n_K):
        resk = get_data_k(
            n=n_k,
            seed=seed + 10000 * k,
            d=d,
            nb_classes=nb_classes,
            w_oracle=w_oracle[k],
            pos=pos[k],
            **kwargs
        )
        res.append(resk)

    res = list(zip(*res))
    perm = np.arange(n)  # np.random.permutation(n)
    res = [np.concatenate(r)[perm] for r in res]

    return res
