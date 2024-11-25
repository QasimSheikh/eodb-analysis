import unittest
import numpy as np
from pathlib import Path
from src.eodb_analysis_system import (
    GeometricBinner,
    StateMapper,
    TransitionMatrix,
    BayesianPredictor,
    StateFeatures
)

class TestGeometricBinner(unittest.TestCase):
    def setUp(self):
        self.binner = GeometricBinner()

    def test_eodb_binning(self):
        test_values = [0.05, 1.0, 50.0, 200.0]
        binned_values = [self.binner.bin_eodb(v) for v in test_values]
        self.assertEqual(len(binned_values), 4)
        self.assertTrue(all(isinstance(x, np.integer) for x in binned_values))

    def test_duration_binning(self):
        test_durations = [3, 10, 30, 100]
        binned_durations = [self.binner.bin_duration(d) for d in test_durations]
        self.assertTrue(all(isinstance(x, np.integer) for x in binned_durations))

class TestStateMapper(unittest.TestCase):
    def setUp(self):
        self.binner = GeometricBinner()
        self.mapper = StateMapper(self.binner)

    def test_state_mapping(self):
        features = StateFeatures(1, 1, 1, 1)
        state_id = self.mapper.get_state_id(features)
        self.assertEqual(state_id, 0)  # First state should be 0

    def test_state_consistency(self):
        features = StateFeatures(1, 1, 1, 1)
        state_id1 = self.mapper.get_state_id(features)
        state_id2 = self.mapper.get_state_id(features)
        self.assertEqual(state_id1, state_id2)

class TestTransitionMatrix(unittest.TestCase):
    def setUp(self):
        self.matrix = TransitionMatrix(n_states=3)

    def test_update(self):
        self.matrix.update(0, 1)
        self.assertEqual(self.matrix.counts[0, 1], 1)
        self.assertTrue(np.isclose(self.matrix.matrix[0, 1], 1.0))

    def test_save_load(self):
        self.matrix.update(0, 1)
        test_path = Path('test_matrix.npz')
        self.matrix.save(test_path)
        
        loaded_matrix = TransitionMatrix.load(test_path)
        np.testing.assert_array_equal(self.matrix.matrix, loaded_matrix.matrix)
        test_path.unlink()  # Cleanup

class TestBayesianPredictor(unittest.TestCase):
    def setUp(self):
        self.transition_matrix = TransitionMatrix(n_states=3)
        self.transition_matrix.update(0, 1)
        self.predictor = BayesianPredictor(self.transition_matrix)

    def test_prediction(self):
        prediction = self.predictor.predict_next_state(0)
        self.assertEqual(len(prediction), 3)
        self.assertTrue(np.isclose(sum(prediction), 1.0))

    def test_multi_step_prediction(self):
        prediction = self.predictor.predict_next_state(0, n_steps=2)
        self.assertEqual(len(prediction), 3)
        self.assertTrue(np.isclose(sum(prediction), 1.0))

def test_full_pipeline():
    # Integration test
    binner = GeometricBinner()
    mapper = StateMapper(binner)
    matrix = TransitionMatrix(n_states=10)
    predictor = BayesianPredictor(matrix)

    # Test data
    test_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    test_duration = 10
    test_macd = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

    # Process through pipeline
    state = mapper.map_segment_to_state(test_data, test_duration, test_macd)
    prediction = predictor.predict_next_state(state)

    assert isinstance(state, int)
    assert isinstance(prediction, np.ndarray)
    assert len(prediction) == 10

if __name__ == '__main__':
    unittest.main()
