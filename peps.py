# %% [markdown]
# ## Bonus Section. Image Compression with the PEPS Tensor Network

# %% [markdown]
# In this problem, you are asked to compress the image used in the main section using the PEPS tensor network—a 2x3 lattice (since the original tensor has dimension 6=2x3) with a given fixed rank value r for all edges of the tensor network graph—and plot the resulting images from the compressed representation for several values ​​of r.
#
# This problem assumes that you will be using Google's ```TensorNetwork``` package (https://github.com/google/TensorNetwork).
#
# <img src="https://user-images.githubusercontent.com/8702042/67589472-5a1d0e80-f70d-11e9-8812-64647814ae96.png">
#
# This package provides convenient assembly of an arbitrary tensor network graph, as well as access to automatic differentiation by tensor network parameters (see the example assembly at https://github.com/google/TensorNetwork/blob/master/examples/simple_mera/simple_mera.py, which builds a MERA network graph and uses ```jax.grad```). You can use/implement any convenient optimization algorithm to approximate the entire image (by default, it is assumed that only $\|A-X\|_F$ will be minimized, but you can also use the $\|P_\Omega(A - X)\|_F $ functionality).
#
