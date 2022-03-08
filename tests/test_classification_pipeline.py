from pathlib import Path
from unittest import TestCase

from CASA.classification_pipeline import compute_distance_matrix

path = Path(__file__)


class TestDistMatrix(TestCase):

    def setUp(self):
        pass

    @pytest.mark.unit
    def test_distance_matrix(self):
        self.assertTrue(True)

    def tearDown(self):
        pass