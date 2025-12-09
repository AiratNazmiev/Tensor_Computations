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
        cores[k] = (np.diag(S_r) @ VT_r).reshape(r_new, n2, r3)

    return tt.vector.from_list(cores)

# %%
test_tt_sum = test_tensor_tt + test_tensor_tt
test_tt_rounded = tt_round(test_tt_sum, 1e-12, 2)

assert test_tt_rounded.d == 4
assert np.all(test_tt_rounded.n == [2, 2, 2, 2])
assert np.all(test_tt_rounded.r == [1, 2, 2, 2, 1])
assert np.linalg.norm(test_tt_rounded.full() - 2 * test_tensor) < 1e-10

# %% [markdown]
# ## 2. Preparing data for tensor completion (**15 points**)
# 

# %% [markdown]
# Let us take some number of points randomly scattered across the image and look at the values at those points.

# %%
count = 60000

rng = np.random.default_rng(42)
rows = rng.integers(0, m, (count,))
cols = rng.integers(0, n, (count,))
vals = im[rows, cols]

known = np.zeros((m, n))
known[rows, cols] = vals
plt.imshow(known, cmap="gray");

# %% [markdown]
# a. (**7 points**) Write the function `image_to_tensor_inds`, which, given row and column indices of the matrix `image`, returns a list of indices in the $d$-dimensional tensor `tensor`. You are allowed, if necessary, to use the fact that `base == 8` (but you should write an `assert`).

# %%
def image_to_tensor_inds(rows, cols, base, d):
    """
    Input
        rows: list of row indices
        cols: list of column indices
        base: mode size of tensorization
        d: dimension of tensor

    Output
        inds: list of lists representing indices of tensor A
    """
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)

    d_half = d // 2
    size = base ** d_half
    pows = base ** np.arange(d_half, dtype=np.int64)

    i_digits = (rows[:, None] // pows) % base
    j_digits = (cols[:, None] // pows) % base

    count = rows.shape[0]
    inds = np.zeros((count, d), dtype=np.int64)
    inds[:, 0::2] = i_digits
    inds[:, 1::2] = j_digits

    return [list(ind) for ind in inds]

# %%
test_rows = [0, 15, 63]
test_cols = [0, 32, 0]
assert np.all(
    image_to_tensor_inds(test_rows, test_cols, 8, 4) == \
    np.array([
        [0, 0, 0, 0],
        [7, 0, 1, 4],
        [7, 0, 7, 0],
    ])
)

# %% [markdown]
# b. (**5 points**) Write the function `compute_at_points`, which takes a tensor in TT format and a list of indices into it, and returns an array of the values at these positions. **Hint:** use the function `tt.vector.to_list`.

# %%
def compute_at_points(tensor_tt, inds):
    """
    Input
        tensor_tt: tt.vector
        inds: list of n lists representing indices in tensor_tt

    Output
        vals: np.array of size n: values at inds
    """
    cores = list(tt.vector.to_list(tensor_tt))
    d = len(cores)
    N = len(inds)
    result = np.zeros(N, dtype=np.float64)

    for idx0, idx in enumerate(inds):
        curr = cores[0][0, idx[0], :]
        for k in range(1, d):
            Gk = cores[k]
            curr = curr @ Gk[:, idx[k], :]

        result[idx0] = curr.item()

    return result

# %%
test_inds = [[0,1,0,1], [1, 1, 0, 1], [0, 1, 0, 0]]

assert np.all(
    compute_at_points(test_tensor_tt, test_inds) == \
    test_tensor[tuple(I)] for I in test_inds
)

# %% [markdown]
# We want to solve the minimization problem
# $$
# \varphi(X) \equiv \|P_\Omega(A - X)\|_F \to \min_{X},
# $$
# where $A$ is the sought tensor, $X$ belongs to the manifold of tensors of a given TT-rank, $\Omega$ is the set of indices whose values we know, and $P_\Omega$ is an operator (an orthogonal projector) that sets elements outside $\Omega$ to zero.
# 
# c. (**3 points**) Write the function `compute_phi` to compute the functional $\varphi(X)$.

# %%
def compute_phi(inds, vals, X):
    """
    Input
        inds: list of N lists representing indices of tensor A
        vals: np.array of size N: values of A in indices inds
        X: tt.tensor

    Output
        phi: value of phi(X)
    """
    approx = compute_at_points(X, inds)

    return np.linalg.norm(vals - approx)

# %% [markdown]
# Let us now look at the error of the quasi-optimal TT approximation for the rank of interest. This value can be used as a baseline in subsequent optimization.

# %%
tt_appr = tt_svd(image_to_tensor(im, base, d), 1e-12, 20)
inds = image_to_tensor_inds(rows, cols, base, d)
compute_phi(inds, vals, tt_appr)

# %% [markdown]
# For convenience, we will optimize the functional $\psi(X)$, which has the same optimum as $\varphi(X)$, at zero:
# 
# $
# \varphi(X) = \bigl\|P_\Omega(A - X)\bigr\|_F,
# \qquad
# \psi(X) = \frac{1}{2} \varphi^2(X) = \frac{1}{2}\bigl\|P_\Omega(A - X)\bigr\|_F^2.
# $
# 

# %% [markdown]
# ## 3. Riemannian optimization on manifolds (**45 points**)

# %% [markdown]
# For optimization on manifolds, we need 3 ingredients: the gradient at a point $X$ on the manifold, the projector onto the tangent space, and a retraction.
# 
# a. (**4 points**)  Write down the formula for the Euclidean gradient of the given functional at a point $X$. Note that the gradient will be a tensor of the same size as $X$ (formally, you may assume that when computing the functional, a vectorization of $A-X$ is used, and the gradient vector is reshaped back into a tensor at the end).

# %% [markdown]
# Write $\varphi(X)$ in the following form:
# $$ \varphi(X) = \sqrt{ \sum_{i \in \Omega} (A_i - X_i)^2 } = \sqrt{\sum_{i\in\Omega} r_i^2 }. $$
# 
# For $\varphi(X) > 0$,  
# $$
# \nabla \varphi(X)
# = -\,\frac{P_\Omega(A - X)}{\|P_\Omega(A - X)\|_F}
# = -\,\frac{P_\Omega(r)}{\varphi(X)}.
# $$
# 
# Elementwise:
# $$
# (\nabla \varphi(X))_i =
# \begin{cases}
# \dfrac{X_i - A_i}{\varphi(X)}, & i \in \Omega,\\
# 0, & i \notin \Omega.
# \end{cases}
# $$
# 
# If $\varphi(X)=0$, we will set $\nabla\varphi(X)=0$.

# %% [markdown]
# Since $\psi(X)$ has the form:
# $$ \psi(X) = \frac{1}{2} \sum_{i\in\Omega} (A_i - X_i)^2 = \frac{1}{2} \sum_{i\in\Omega} r_i^2, $$  
# we obtain:
# $$
# \nabla \psi(X) = P_\Omega(X - A) = -\,P_\Omega(r).
# $$
# 
# Elementwise:
# $$
# (\nabla \psi(X))_i =
# \begin{cases}
# (X_i - A_i), & i \in \Omega,\\
# 0, & i \notin \Omega.
# \end{cases}
# $$
# 

# %% [markdown]
# b. (**10 points**)  Using the given function `project_onto_tangent_space`, write the function `compute_gradient_projection` that computes the projection of the gradient at a point $X$ onto the tangent space at the same point. Note that `project_onto_tangent_space` can compute the projection of a sum of tensors; in terms of efficiency, it is better to pass it a list of $n$ rank-1 tensors than a single tensor of rank $n$ (and they also use less memory).

# %%
def project_onto_tangent_space(X, what):
    """
    Input
        X: tensor from manifold. The tangent space corresponds to X
        what: list of tensors to project onto the tangent space

    Output
        proj: sum of projections of tensors from list `what` onto
              the tangent space
    """

    return tt.riemannian.riemannian.project(X, what, use_jit=True)

# %%
def compute_gradient_projection(inds, vals, X):
    """
    Input
        inds: list of N lists representing indices of tensor A
        vals: np.array of size N: values of A in indices inds
        X: tensor from manifold
    """
    x_vals = compute_at_points(X, inds)
    coeffs = x_vals - vals

    d = X.d
    sizes = X.n
    spikes = []

    for coef, idx in zip(coeffs, inds):
        cores = []
        for k, n_k in enumerate(sizes):
            core = np.zeros((1, n_k, 1), dtype=float)
            if k == 0:
                core[0, idx[k], 0] = coef
            else:
                core[0, idx[k], 0] = 1
            cores.append(core)

        spikes.append(tt.vector.from_list(cores))

    return project_onto_tangent_space(X, spikes)

# %% [markdown]
# c. (**5 points**)  Now that you know the direction in which you need to move, you need to compute the step size. Derive the formula for the optimal (ignoring retraction) step $\tau$, that is, the one for which
# $$
# \min_{\tau \in \mathbb{R}} \|P_\Omega(A - (X + \tau Y))\|_F
# $$
# is attained, where $Y$ is the gradient you have found.
# 
# 

# %% [markdown]
# Introduce $ X(\tau) = X + \tau Y $. Along this line:
# $$
# \psi(\tau)
# = \frac{1}{2}\sum_{i\in\Omega} (r_i - \tau y_i)^2,
# $$
# where $y_i = Y_i$. The function is convex in $\tau$.
# 
# Differentiate with respect to the parameter:
# $$
# \psi'(\tau)
# = -\sum_{i\in\Omega} r_i y_i
#   + \tau \sum_{i\in\Omega} y_i^2.
# $$
# 
# Setting the derivative to zero, $\psi'(\tau)=0$, we obtain the optimal step:
# $$
# \tau_\psi^*
# =
# \frac{\sum_{i\in\Omega} r_i y_i}
#      {\sum_{i\in\Omega} y_i^2}
# =
# \frac{\sum_{i\in\Omega} (A_i - X_i) Y_i}
#      {\sum_{i\in\Omega} Y_i^2}.
# $$
# 
# For the original function, $ \varphi(\tau) = \sqrt{\psi(\tau)} $, for $\varphi(\tau)>0$ we get:
# $$
# \varphi'(\tau) = \frac{\psi'(\tau)}{2\sqrt{\psi(\tau)}}.
# $$
# So $\varphi'(\tau)=0$ is equivalent to $\psi'(\tau)=0$.
# 
# Thus, for both functionals the optimal step values are the same:
# $$
# \tau_\varphi^* = \tau_\psi^*
# =
# \frac{\sum_{i\in\Omega} (A_i - X_i) Y_i}
#      {\sum_{i\in\Omega} Y_i^2}.
# $$

# %% [markdown]
# d. (**5 points**)  Write the function `get_optimal_tau` that computes $\tau$ using the formula you derived.

# %%
def get_optimal_tau(inds, vals, X, grad):
    """
    Input
        inds: list of N lists representing indices of tensor A
        vals: np.array of size N: values of A in indices inds
        X: tensor from manifold
        grad: gradient in X projected onto the tangent space

    Output
        tau: step minimizing functional along the direction grad
    """
    x_vals = compute_at_points(X, inds)
    g_vals = compute_at_points(grad, inds)

    res = vals - x_vals
    num = np.dot(res, g_vals)
    den = np.dot(g_vals, g_vals)

    if np.allclose(den, 0.):
        return 0.

    return num / den

# %% [markdown]
# e. (**3 points**) Write the function `retract` that computes the retraction onto the manifold.

# %%
def retract(X, rank, eps=1e-12):
    """
    Input
        X: tensor (possibly not from manifold)
        rank: rank of tensors in manifold

    Output
        Xr: retracted X
    """

    if isinstance(X, tt.vector):
        return tt_round(X, eps, rank)
    elif isinstance(X, np.ndarray):
        return tt_svd(X, eps, rank)
    else:
        raise TypeError("X must be tt.vector or np.ndarray")

# %% [markdown]
# f. (**15 points**) Write the function `optimize` that performs Riemannian optimization of our functional on the manifold of tensors of fixed TT-rank. If `X0` is `None`, take a random tensor with ranks `rank` as the initial approximation (for this you can use the function `tt.rand`). In this case, don’t forget to normalize the initial approximation: a tensor represented by a random TT decomposition may have very large elements. Also, in this implementation, for simplicity, you may always perform exactly `maxiter` iterations.
# 
# You will need the functions `compute_gradient_projection`, `get_optimal_tau`, and `retract`.

# %%
def optimize(inds, vals, rank, X0=None, base=None, d=None, maxiter=10):
    """
    Input
        inds: list of N lists representing indices of tensor A
        vals: np.array of size N: values of A in indices inds
        rank: target rank of approximation
        X0: initial approximation. If None, chosen randomly
        base, d: parameters for shape of tensor A: (base,) * d.
                 Use them for initialization if X0=None
        maxiter: number of iterations to perform

    Output
        Xk: approximation after maxiter iterations
        errs: values of functional on each step
    """
    if X0 is None:
        Xk = tt.rand(base, d, rank)
        norm = np.linalg.norm(Xk.full())
        if norm > 0:
            Xk = Xk * (1. / norm)
    else:
        Xk = X0

    errs = []

    for _ in tqdm(range(maxiter)):
        grad = compute_gradient_projection(inds, vals, Xk)
        tau  = get_optimal_tau(inds, vals, Xk, grad)
        Xk   = retract(Xk + grad * tau, rank)
        errs.append(compute_phi(inds, vals, Xk))

    return Xk, np.asarray(errs)

# %% [markdown]
# Perform the first 10 iterations of the optimization process. Make sure that the functional decreases monotonically. After 10 iterations, the functional will most likely be around 4000.

# %%
inds = image_to_tensor_inds(rows, cols, base, d)
X_10, errs_10 = optimize(inds, vals, rank=20, base=base, d=d, maxiter=10)

# %% [markdown]
# Let us output the current approximation as an image.

# %%
plt.imshow(tensor_to_image(X_10.full()).clip(0, 255), cmap="gray");

# %%
errs_10

# %% [markdown]
# g. (**3 points**) Run 90 more iterations of the method. Does the image become better in your opinion? Note that the value of the functional has become smaller than the baseline obtained using TT-SVD with the same rank. How do the obtained results agree with the quasi-optimality of TT-SVD? Does a smaller functional value lead to a better visual result? What is the reason for this: an insufficiently advanced optimization algorithm or an issue with the problem formulation?

# %%
X_100, errs_10_100 = optimize(inds, vals, rank=20, X0=X_10, maxiter=90)
errs_100 = np.concatenate([errs_10, errs_10_100])

# %%
errs_100[-1]

# %%
plt.imshow(tensor_to_image(X_100.full()).clip(0, 255), cmap="gray");

# %%
plt.figure()
plt.semilogy(errs_100)
plt.grid(True, 'major'); plt.grid(True, 'minor', lw=0.3, alpha=0.5)
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.tight_layout();
