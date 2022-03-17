from pathlib import Path
from unittest import TestCase

import numpy as np
import pytest

from CASA.distance_matrix import compute_distance_matrix

path = Path(__file__)


class TestDistMatrix(TestCase):
    def setUp(self):
        pass

    @pytest.mark.unit
    def test_distance_matrix(self):
        matrix_arrays = np.random.random((100, 10, 50))
        dist_matr = compute_distance_matrix(matrix_arrays, num_part=10)
        self.assertEqual(matrix_arrays.shape[0], dist_matr.shape[0])

    def tearDown(self):
        pass
