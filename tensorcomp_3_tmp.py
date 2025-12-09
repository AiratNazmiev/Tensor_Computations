# %% [markdown]
# # Практическое домашнее задание № 3
# 
# ## Риманова оптимизация на ТТ многообразиях фиксированного ранга для задачи tensor completion
# 
# В этом задании вам предстоит восстановить тензор по заданному набору его элементов. Точнее, найти достаточно близкий в пространстве всех тензоров фиксированного ТТ-ранга.
# 
# В качестве тензора мы возьмём тензоризованную чёрно-белую картинку. Этот пример с одной наглядно продемонстрирует идею тензоризации, а с другой -- покажет типичные проблемы, возникающие при восстановлении матриц или тензоров по небольшому числу элементов. Так же вы попрактикуетесь в использовании методов римановой оптимизации и построении алгоритмов для ТТ разложения.

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
# Тензор будет иметь размер 8 x 8 x 8 x 8 x 8 x 8, причём при тензоризации мы применим трюк с перестановкой индексов, чтобы помочь методу учесть присущее картинкам локальное самоподобие.

# %%
m, n = im.shape
assert m == n == 512
base = 8
d = 3 * 2

# %% [markdown]
# ### 1. Тензоризация, матрицизация и ТТ-SVD (**40 баллов**)
# 
# Давайте для начала посмотрим на квазиоптимальное приближение ранга $r = 20$ к нашей тензоризованной картинке. Для этого нам понадобится 3 функции: `image_to_tensor`, `tt_svd` и `tensor_to_image`.
# 
# a. (**5 баллов**) Напишите функцию `image_to_tensor`, превращающую картинку размера $b^{d/2} \times b^{d/2}$ в тензор размеров $b \times \dots \times b$:
# $$
#      \mathtt{tensor}_{i_1, j_1, \dots, i_{d/2}, j_{d/2}} = \mathtt{image}_{\overline{i_{d/2}\dots i_1}, \overline{j_{d/2}\dots j_{1}}}.
# $$
# Обратите внимание, что индексы $i_k, j_k$ можно бы было объединить в один индекс, как это делается для ТТ матриц. Мы же этого делать не будем, так как ожидаем, что отделение этих индексов друг от друга не приводит к полному рангу для картинки.

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
# b. (**5 баллов**) Напишите функцию `tensor_to_image`, выполняющую обратное преобразование.

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
# c. (**12 баллов**)  Напишите функцию `tt_svd`, приближающую данный тензор TT-разложением (возвращающую объект типа `tt.vector`) с заданной точностью $\varepsilon$ и с рангами, не превосходящими $r$ (ограничение на ранги должно срабатывать, если указанной точности достичь не получается). Используйте результат о квазиоптимальности приближения для выбора точности отсечения сингулярных чисел для каждого ядра.
# 
# **Обратите внимание:** для получения баллов за этот пункт в функции запрещено использовать TT-SVD, реализованный в конструкторе `tt.vector`, а также метод `.round()` объекта `tt.vector`. Для создания объекта `tt.vector` используйте функцию `tt.vector.from_list`. Вы можете использовать данную здесь референсную реализацию дальше в коде, если хотите пропустить этот пункт.
# 

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
# Теперь вы можете посмотреть на квазиоптимальное приближение для нескольких значений ТТ-ранга (10, 20 и 40). Обратите внимание на структуру артефактов в этих картинках.

# %%
fig, axs = plt.subplots(1, 3, figsize=(20,7))
tensor = image_to_tensor(im, base, d)
for ax, rank in zip(axs, [10, 20, 40]):
    tt_appr = tt_svd(tensor, 1e-12, rank)
    ax.imshow(tensor_to_image(tt_appr.full()).clip(0, 255), cmap="gray")

# %% [markdown]
# В процессе оптимизации на многообразиях тензоров фиксированного ТТ-ранга нам понадобится округлять ТТ-разложения после арифметических операций.
# 
# d. (**18 баллов**)  Напишите функцию `tt_round`, которая будет делать это по алгоритму, описанному на лекции.
# 
# **Обратите внимание:** для получения баллов за этот пункт в функции запрещено использовать метод `.round()` объекта `tt.vector`. Однако вы можете использовать данную здесь референсную реализацию дальше в коде, если хотите пропустить этот пункт.

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
# ## 2. Подготовка данных для tensor completion (**15 баллов**)
# 
# 

# %% [markdown]
# Возьмём некоторое количество точек, случайно разбросанных по картинке, и посмотрим на значения в них.

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
# a. (**7 баллов**) Напишите функцию `image_to_tensor_inds`, которая по индексам строк и столбцов матрицы `image` вернёт список индексов в $d$-мерном тензоре `tensor`. Разрешается при необходимости пользоваться тем, что `base == 8` (но стоит написать `assert`).

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
# b. (**5 баллов**) Напишите функцию `compute_at_points`, которая принимает тензор в ТТ-формате и список индексов в него и возвращает массив значений в этих позициях. **Подсказка:** используйте функцию `tt.vector.to_list`.

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
# Мы хотим решить задачу минимизации
# $$
# \varphi(X) \equiv \|P_\Omega(A - X)\|_F \to \min_{X},
# $$
# где $A$ — искомый тензор, $X$ принадлежит многообразию тензоров заданного ТТ-ранга, $\Omega$ есть множество индексов, значения в которых нам известны, а $P_\Omega$ — оператор (являющийся ортопроектором), заменяющий элементы вне $\Omega$ на нули.
# 
# c. (**3 баллов**) Напишите функцию `compute_phi` вычисления функционала $\varphi(X)$.

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
# Давайте посмотрим на ошибку квазиотпимального ТТ-приближения для интересующего нас ранга. Эту величину можно использовать как бейзлайн в дальнейшей оптимизации.

# %%
tt_appr = tt_svd(image_to_tensor(im, base, d), 1e-12, 20)
inds = image_to_tensor_inds(rows, cols, base, d)
compute_phi(inds, vals, tt_appr)

# %% [markdown]
# Для удобства будем оптимизировать функционал $\psi(X)$, который имеет тот же оптимум, что и  $\varphi(X)$, в нуле:
# 
# $
# \varphi(X) = \bigl\|P_\Omega(A - X)\bigr\|_F,
# \qquad
# \psi(X) = \frac{1}{2} \varphi^2(X) = \frac{1}{2}\bigl\|P_\Omega(A - X)\bigr\|_F^2.
# $
# 

# %% [markdown]
# ## 3. Риманова оптимизация на многообразиях (**45 балла**)

# %% [markdown]
# Для оптимизации на многообразиях нужно 3 ингредиента: градиент в точке многообразия $X$, проектор на касательное пространство и ретракция.
# 
# a. (**4 балла**)  Напишите формулу для евклидового градиента указанного функционала в точке $X$. Обратите внимание, что градиент будет тензором того же размера, что и $X$ (формально можете считать, что при вычислении функционала используется векторизация тензора $A-X$, а градиент-вектор в конце решейпится обратно в тензор).

# %% [markdown]
# Запишем формулу для в $\varphi(X)$ в следующем виде:
# $$ \varphi(X) = \sqrt{ \sum_{i \in \Omega} (A_i - X_i)^2 } = \sqrt{\sum_{i\in\Omega} r_i^2 }. $$
# 
# Для $\varphi(X) > 0$,  
# $$
# \nabla \varphi(X)
# = -\,\frac{P_\Omega(A - X)}{\|P_\Omega(A - X)\|_F}
# = -\,\frac{P_\Omega(r)}{\varphi(X)}.
# $$
# 
# Поэлементно:
# $$
# (\nabla \varphi(X))_i =
# \begin{cases}
# \dfrac{X_i - A_i}{\varphi(X)}, & i \in \Omega,\\
# 0, & i \notin \Omega.
# \end{cases}
# $$
# 
# Если $\varphi(X)=0$, будем считать $\nabla\varphi(X)=0$.

# %% [markdown]
# Так как $\psi(X)$ имеет вид:
# $$ \psi(X) = \frac{1}{2} \sum_{i\in\Omega} (A_i - X_i)^2 = \frac{1}{2} \sum_{i\in\Omega} r_i^2, $$  
# получим:
# $$
# \nabla \psi(X) = P_\Omega(X - A) = -\,P_\Omega(r).
# $$
# 
# Поэлементная запись:
# $$
# (\nabla \psi(X))_i =
# \begin{cases}
# (X_i - A_i), & i \in \Omega,\\
# 0, & i \notin \Omega.
# \end{cases}
# $$
# 

# %% [markdown]
# b. (**10 баллов**)  Используя данную вам функцию `project_onto_tangent_space`, напишите функцию `compute_gradient_projection`, вычисляющую проекцию градиента в точке $X$ на касательную плоскость в этой же точке. Обратите внимание, что `project_onto_tangent_space` умеет считать проекцию суммы тензоров, причём с точки зрения эффективности лучше передать в неё список из $n$ тензоров ранга 1, чем 1 тензор ранга $n$ (да и памяти они меньше занимают).

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
# c. (**5 баллов**)  Теперь, когда вам известно направление, в котором нужно двигаться, нужно вычислить размер шага. Выведите формулу для оптимального (без учёта ретракции) шага $\tau$, то есть такого, для которого достигается
# $$
# \min_{\tau \in \mathbb{R}} \|P_\Omega(A - (X + \tau Y))\|_F,
# $$
# где $Y$ — найденный вами градиент.
# 
# 

# %% [markdown]
# 
# Введем $ X(\tau) = X + \tau Y $. Вдоль этой прямой:
# $$
# \psi(\tau)
# = \frac{1}{2}\sum_{i\in\Omega} (r_i - \tau y_i)^2,
# $$
# где $y_i = Y_i$. Функция выпуклая по $\tau$.
# 
# Продифференцируем по параметру:
# $$
# \psi'(\tau)
# = -\sum_{i\in\Omega} r_i y_i
#   + \tau \sum_{i\in\Omega} y_i^2.
# $$
# 
# Приравняв производную нулю, $\psi'(\tau)=0$, получим оптимальный шаг:
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
# Для исходной функции, $ \varphi(\tau) = \sqrt{\psi(\tau)} $, для $\varphi(\tau)>0$ получим:
# $$
# \varphi'(\tau) = \frac{\psi'(\tau)}{2\sqrt{\psi(\tau)}}.
# $$
# То есть $\varphi'(\tau)=0$ эквивалентно $\psi'(\tau)=0$.
# 
# Таким образом, для обоих функционалов значения оптимального шага равны:
# $$
# \tau_\varphi^* = \tau_\psi^*
# =
# \frac{\sum_{i\in\Omega} (A_i - X_i) Y_i}
#      {\sum_{i\in\Omega} Y_i^2}.
# $$

# %% [markdown]
# d. (**5 баллов**)  Напишите функцию `get_optimal_tau`, вычисляющую $\tau$ по найденной вами формуле.

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
# e. (**3 балла**) Напишите функцию `retract`, вычисляющую ретракцию на многообразие.

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
# f. (**15 баллов**) Напишите функцию `optimize`, выполняющую Риманову оптимизацию нашего функционала на многообразии тензоров фиксированного ТТ-ранга. Если `X0` — `None`, то возьмите в качестве начального приближения случайный тензор с рангами `rank` (для этого можно использовать функцию `tt.rand`). В этом случае не забудьте нормировать начальное приближение: у тензора, представленного случайным ТТ-разложением, могут оказаться очень большие элементы. Также в данной реализации можно для простоты всегда делать `maxiter` итераций.
# 
# Вам потребуются функции `compute_gradient_projection`, `get_optimal_tau` и `retract`.

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
# Выполните первые 10 итераций оптимизиационного процесса. Убедитесь, что функционал монотонно уменьшается. После 10 итераций функционал скорее всего будет иметь величину около 4000.

# %%
inds = image_to_tensor_inds(rows, cols, base, d)
X_10, errs_10 = optimize(inds, vals, rank=20, base=base, d=d, maxiter=10)

# %% [markdown]
# 
# Давайте выведем текущее приближение в виде картинки.

# %%
plt.imshow(tensor_to_image(X_10.full()).clip(0, 255), cmap="gray");

# %%
errs_10

# %% [markdown]
# g. (**3 балла**) Выполните ещё 90 итераций метода. Становится ли картинка лучше, на ваш взгляд? Обратите внимание, что значение функционала стало меньше, чем для бейзлайна, полученного с помощью ТТ-SVD того же ранга. Как полученные результаты согласуются с квазиоптимальностью TT-SVD? Приводит ли меньшее значение функционала к лучшему визуальному результату? Из-за чего это происходит: недостаточно продвинутый алгоритм оптимизации или проблема с постановкой задачи?

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

# %% [markdown]
# ## Бонусная часть. Сжатие картинок с помощью тензорной сети PEPS

# %% [markdown]
# В этой задаче вам предлагается сжать используемую в основной части картинку с помощью тензорной сети PEPS -- решетки, размера $2\times 3$ (так как исходный тензор имеет размерность $6=2\cdot3$) c заданным фиксированным значением ранга $r$ для всех ребер графа тензорной сети и изобразить полученные картинки из сжатого представления для нескольких значений $r$.
# 
# В этой задаче предполгается, что вы будете пользоваться пакетом ```TensorNetwork``` от google (https://github.com/google/TensorNetwork).
# 
# <img src="https://user-images.githubusercontent.com/8702042/67589472-5a1d0e80-f70d-11e9-8812-64647814ae96.png">
# 
# В этом пакете доступна удобная сборка произвольного графа тензорной сети, а также есть доступ к автоматическому дифференцированию по параметрам тензорной сети (смотри сборку в примере https://github.com/google/TensorNetwork/blob/master/examples/simple_mera/simple_mera.py, где собирается граф сети MERA, а также используется ```jax.grad```). Вы можете использовать/сами реализовать любой удобный оптимизационный алгоритм для приближения всей картинки (по умолчанию предполагается, что минимизироваться будет только $\|A-X\|_F$, но можете также использовать функционал $\|P_\Omega(A - X)\|_F $).
# 


