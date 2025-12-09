# %% [markdown]
# # Practical Homework #3
# 
# ## Riemannian optimization on fixed-rank TT manifolds for the tensor completion problem
# 
# In this assignment you will reconstruct a tensor from a given subset of its elements. More precisely, you need to find a tensor in the space of all tensors of a fixed TT-rank that is sufficiently close.
# 
# As the tensor, we will take a tensorized black-and-white image. This example, on the one hand, clearly demonstrates the idea of tensorization, and on the other hand shows typical issues that arise when reconstructing matrices or tensors from a small number of elements. You will also practice using Riemannian optimization methods and building algorithms for TT decomposition.

# %%
import numpy as np
!pip install git+https://github.com/oseledets/ttpy
import tt
import tt.riemannian.riemannian
import scipy as sp
from scipy import linalg as spla
from matplotlib import image, pyplot as plt
from tqdm import tqdm

# %%
from tqdm import tqdm

# %%
im = image.imread("cameraman.jpg")[1:,1:] # Fix strange size 513x513
plt.imshow(im, cmap="gray");

# %% [markdown]
# The tensor will have size 8 x 8 x 8 x 8 x 8 x 8, and during tensorization we will apply an index permutation trick to help the method take into account the local self-similarity typical for images.

# %%
m, n = im.shape
assert m == n == 512
base = 8
d = 3 * 2

# %% [markdown]
# ### 1. Tensorization, matricization and TT-SVD (**40 points**)
# 
# To begin with, let us look at a quasi-optimal approximation of rank $r = 20$ to our tensorized image. For this we will need 3 functions: `image_to_tensor`, `tt_svd`, and `tensor_to_image`.
# 
# a. (**5 points**) Write the function `image_to_tensor` that converts an image of size $b^{d/2} \times b^{d/2}$ into a tensor of size $b \times \dots \times b$:
# $$
#      \mathtt{tensor}_{i_1, j_1, \dots, i_{d/2}, j_{d/2}} = \mathtt{image}_{\overline{i_{d/2}\dots i_1}, \overline{j_{d/2}\dots j_{1}}}.
# $$
# Note that the indices $i_k, j_k$ could be merged into a single index, as is done for TT matrices. We will not do this, since we expect that separating these indices from each other does not lead to full rank for the image.

# %%
def image_to_tensor(image, base, d):
    """
    Input
        image: np array of shape base**(d//2) x base**(d//2)
        base: mode size of output tensor
        d: dimension of output tensor

    Output:
        tensor: d-dimensional np array of shape base x ... x base
    """
    half = d // 2
    size = base ** half

    tensor = np.zeros((base,) * d, dtype=image.dtype)

    flat_image = image.ravel(order='F')
    for idx, val in enumerate(flat_image):
        row = idx % size
        col = idx // size
        i = [(row // (base**k)) % base for k in range(half)]
        j = [(col // (base**k)) % base for k in range(half)]

        idx_tt = []

        for ik, jk in zip(i, j):
            idx_tt.extend((ik, jk))

        tensor[tuple(idx_tt)] = val

    return tensor

# %%
test_image = np.arange(16).reshape((4,4), order="F")
test_tensor = image_to_tensor(test_image, 2, 4)
assert np.all(test_tensor[:, 0, :, 0] == np.array([[0, 2], [1, 3]]))
assert np.all(test_tensor[:, 1, :, 0] == np.array([[4, 6], [5, 7]]))
assert np.all(test_tensor[:, 0, :, 1] == np.array([[8, 10], [9, 11]]))
assert np.all(test_tensor[:, 1, :, 1] == np.array([[12, 14], [13, 15]]))

# %% [markdown]
# b. (**5 points**) Write the function `tensor_to_image` that performs the inverse transformation.

# %%
def tensor_to_image(tensor):
    """
    Input
        tensor: d-dimensional np array of shape (base x ... x base)

    Output:
        image: np array of shape base**(d//2) x base**(d//2)
    """
    d = tensor.ndim

    base = tensor.shape[0]
    d_half = d // 2
    size = base ** d_half
    image = np.zeros((size, size), dtype=tensor.dtype)

    for idx_tt in np.ndindex(tensor.shape):
        row = col = 0
        for k in range(d_half):
            row += idx_tt[2*k] * (base**k)
            col += idx_tt[2*k+1] * (base**k)
        image[row, col] = tensor[idx_tt]

    return image

# %%
assert np.all(tensor_to_image(test_tensor) == test_image)

# %%
assert np.all(tensor_to_image(image_to_tensor(im, base, d)) == im)

# %% [markdown]
# c. (**12 points**)  Write the function `tt_svd` that approximates a given tensor by a TT decomposition (returning an object of type `tt.vector`) with the specified accuracy $\varepsilon$ and with TT-ranks not exceeding $r$ (the rank constraint should take effect if the specified accuracy cannot be achieved). Use the result about quasi-optimality of the approximation to choose the truncation tolerance for singular values for each core.
# 
# **Note:** to receive points for this part, you are forbidden from using the TT-SVD implemented in the `tt.vector` constructor, as well as the `.round()` method of the `tt.vector` object. To create a `tt.vector` object, use the function `tt.vector.from_list`. You may use the reference implementation given here later in the code if you want to skip this part.

# %%
def tt_svd(tensor, eps, max_rank):
    """
    Input
        tensor: np array
        eps: desired difference in frobenius norm between tensor and TT approximation
        max_rank: upper hard limit on each TT rank (it has priority over eps)

    Output
        tensor_tt: TT decomposition of tensor
    """
    A = tensor.copy()
    dims = A.shape
    d = len(dims)

    cores = []
    r_prev = 1

    for k in range(d - 1):
        n_k = dims[k]
        A = A.reshape((r_prev * n_k, -1))
        U, S, Vt = np.linalg.svd(A, full_matrices=False)

        S_sq = S**2
        total_sq = float(np.sum(S_sq))

        tol_loc_sq = (eps**2 / max(d - 1, 1)) * total_sq

        cumsum_sq = np.cumsum(S_sq)
        r_new = len(S)
        for r in range(1, len(S) + 1):
            tail_sq = total_sq - cumsum_sq[r - 1]
            if tail_sq <= tol_loc_sq:
                r_new = r
                break

        r_new = min(r_new, max_rank)

        U_r = U[:, :r_new]
        S_r = S[:r_new]
        VT_r = Vt[:r_new, :]

        cores.append(U_r.reshape((r_prev, n_k, r_new)))
        A = np.dot(np.diag(S_r), VT_r)
        r_prev = r_new

    cores.append(A.reshape((r_prev, dims[-1], 1)))

    return tt.vector.from_list(cores)

# %%
test_tensor_tt = tt_svd(test_tensor, 1e-12, 2)
assert test_tensor_tt.d == 4
assert np.all(test_tensor_tt.n == [2, 2, 2, 2])
assert np.all(test_tensor_tt.r == [1, 2, 2, 2, 1])
assert np.linalg.norm(test_tensor_tt.full() - test_tensor) < 1e-10

# %% [markdown]
# Now you can look at the quasi-optimal approximation for several values of TT-rank (10, 20, and 40). Pay attention to the structure of artifacts in these images.

# %%
fig, axs = plt.subplots(1, 3, figsize=(20,7))
tensor = image_to_tensor(im, base, d)
for ax, rank in zip(axs, [10, 20, 40]):
    tt_appr = tt_svd(tensor, 1e-12, rank)
    ax.imshow(tensor_to_image(tt_appr.full()).clip(0, 255), cmap="gray")

# %% [markdown]
# During optimization on the manifold of tensors of fixed TT-rank, we will need to round TT decompositions after arithmetic operations.
# 
# d. (**18 points**)  Write the function `tt_round`, which performs this rounding according to the algorithm described in the lecture.
# 
# **Note:** to receive points for this part, you are forbidden from using the `.round()` method of the `tt.vector` object. However, you may use the reference implementation given here later in the code if you want to skip this part.

# %%
def tt_round(T, eps, max_rank):
    """
    Input
        T: tt.vector
        eps: desired difference in frobenius norm between T and the rounded decomposition
        max_rank: upper hard limit on each TT rank of rounded decomposition.
                  It has priority over eps)

    Output
        rounded: rounded TT decomposition of T
    """
    cores = [c.copy() for c in tt.vector.to_list(T)]
    d = len(cores)

    for k in range(d - 1):
        G = cores[k]
        r1, n1, r2 = G.shape
        mat = G.reshape(r1 * n1, r2)
        Q, R = np.linalg.qr(mat, mode="reduced")
        r2_new = Q.shape[1]
        cores[k] = Q.reshape(r1, n1, r2_new)

        G_next = cores[k + 1]
        r2_old, n2, r3 = G_next.shape
        mat_next = G_next.reshape(r2_old, n2 * r3)
        cores[k + 1] = (R @ mat_next).reshape(r2_new, n2, r3)

    for k in range(d - 1, 0, -1):
        G_prev = cores[k - 1]
        G = cores[k]
        r1, n1, r2 = G_prev.shape
        r2p, n2, r3 = G.shape

        M = np.tensordot(G_prev, G, axes=(2, 0)).reshape(r1 * n1, n2 * r3)
        U, S, VT = np.linalg.svd(M, full_matrices=False)

        S_sq = S**2
        total_sq = np.sum(S_sq)

        tol_k_sq = (eps**2 / max(d - 1, 1)) * total_sq
        cumsum_sq = np.cumsum(S_sq)
        r_new = len(S)
        for r in range(1, len(S) + 1):
            tail_sq = total_sq - cumsum_sq[r - 1]
            if tail_sq <= tol_k_sq:
                r_new = r
                break
        r_new = max(1, min(r_new, max_rank))

        U_r = U[:, :r_new]
        S_r = S[:r_new]
        VT_r = VT[:r_new, :]

        cores[k - 1] = U_r.reshape(r1, n1, r_new)
