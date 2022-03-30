from typing import Union

import numpy as np
import scipy
from datafold.pcfold.kernels import RadialBasisKernel, _apply_kernel_function_numexpr


class ThinPlateKernel(RadialBasisKernel):
    r"""Thin plate radial basis kernel.

    .. math::
        K = D \cdot \log(\sqrt{D} + \vardelta)

    where :math:`D` is the squared euclidean distance matrix.

    See also super classes :class:`RadialBasisKernel` and :class:`PCManifoldKernel`
    for more functionality and documentation.

    Parameters
    ----------
    delta
        Additional constant to make logarithm well-defined.
    """

    def __init__(self, delta: float = 1.0):
        self.delta = delta
        super(ThinPlateKernel, self).__init__(distance_metric="sqeuclidean")

    def eval(
        self, distance_matrix: Union[np.ndarray, scipy.sparse.csr_matrix]
    ) -> Union[np.ndarray, scipy.sparse.csr_matrix]:
        """Evaluate the kernel on pre-computed distance matrix.

        Parameters
        ----------
        distance_matrix
            Matrix of pairwise distances of shape `(n_samples_Y, n_samples_X)`.

        Returns
        -------
        Union[np.ndarray, scipy.sparse.csr_matrix]
            Kernel matrix of same shape and type as `distance_matrix`.
        """

        # Security copy, the distance matrix is maybe required again (for gradient,
        # or other computations...)

        self.delta = self._check_bandwidth_parameter(
            parameter=self.delta, name="delta"
        )

        kernel_matrix = _apply_kernel_function_numexpr(
            distance_matrix,
            expr="D * log(sqrt(D) + delta)",
            expr_dict={"delta": self.delta},
        )

        return kernel_matrix
