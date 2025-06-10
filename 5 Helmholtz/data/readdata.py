#%%
import numpy as np
import matplotlib.pyplot as plt

N=128
data_f = np.empty((1, 1, N, N), dtype=np.complex64)
data_kappa = np.empty((1, 1, N, N), dtype=np.float32)
data_u = np.empty((1, 1, N, N), dtype=np.complex64)

for i in range(1,2):
    f = np.load(f"random/{N}/f{i}.npy")
    kappa = np.load(f"random/{N}/kappa{i}.npy")
    u = np.load(f"random/{N}/u{i}.npy")
    pic_has_nan = np.unique(np.where(np.isnan(f))[0])
    f = np.delete(f, pic_has_nan, axis=0)
    kappa = np.delete(kappa, pic_has_nan, axis=0)
    u = np.delete(u, pic_has_nan, axis=0)
    print(f.shape, kappa.shape, u.shape)
    data_f = np.concatenate((data_f, f), axis=0)
    data_kappa = np.concatenate((data_kappa, kappa), axis=0)
    data_u = np.concatenate((data_u, u), axis=0)
np.save(f"random/{N}/data_f{N}", data_f[1:, :, :, :])
np.save(f"random/{N}/data_kappa{N}", data_kappa[1:, :, :, :])
np.save(f"random/{N}/data_u{N}", data_u[1:, :, :, :])
# %%
