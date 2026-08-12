"""SklearnAutoencoder: fit, transform, inverse_transform, reconstruction_error.

A neural-bottleneck featurizer built on MLPRegressor. The bottleneck is
the smallest hidden layer.
"""

# %%
import numpy as np

from jurebes.featurizers import SklearnAutoencoder

rng = np.random.default_rng(0)
X = rng.normal(size=(40, 12))

ae = SklearnAutoencoder(hidden_layer_sizes=(8, 3, 8), max_iter=200, random_state=0)
ae.fit(X)

Z = ae.transform(X)
Xr = ae.inverse_transform(Z)
err = ae.reconstruction_error(X)

print(f"bottleneck dim={Z.shape[1]}  reconstruction shape={Xr.shape}")
print(f"mean reconstruction error={err.mean():.4f}")
