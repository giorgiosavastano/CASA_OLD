from pathlib import Path
from unittest import TestCase

import numpy as np
import pytest
from dask.distributed import Client

from CASA.distance_matrix import compute_distance_matrix

client = Client(n_workers=4)

path = Path(__file__)


class TestDistMatrix(TestCase):
    def setUp(self):
        pass

    @pytest.mark.unit
    def test_distance_matrix(self):
        matrix_arrays = np.random.random((1000, 10, 50))
        dist_matr = compute_distance_matrix(matrix_arrays, num_part=100)
        self.assertEqual(matrix_arrays.shape[0], dist_matr.shape[0])

    def tearDown(self):
        client.shutdown()
        pass
