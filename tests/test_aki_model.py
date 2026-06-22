# Unit tests for Inference_Model and Feature_Processor classes
# Tests focus on behaviour, not implementation details

import os
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from aki_model import Inference_Model, Feature_Processor

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'aki_model.pkl')

class TestInferenceModel(unittest.TestCase):
    """Tests for Inference_Model class behaviour"""
    @classmethod
    def setUpClass(cls):
        """Load the pretrained model once for all tests"""
        cls.model = Inference_Model(model_path=MODEL_PATH)

    # ---------------------------
    # Inference behaviour tests
    # ---------------------------

    def test_inference_returns_boolean(self):
        """Inference should return a boolean value"""
        # Feature order: Age, Sex, Creatinine_CV, Min_Creatinine, Creatinine_Max_Change,
        # Creatinine_Std, Creatinine_Max_Consec_Increase, Creatinine_Last_gt130_Count,
        # Creatinine_Increase_Count, Creatinine_Max_Consec_Decrease, Creatinine_Count_Since_Peak,
        # Creatinine_Change, Creatinine_gt1p5_Count
        patient_data = np.array([[50, 1, 0.05, 75, 5, 4, 3, 0, 2, 2, 1, 5, 0]])
        result = self.model.inference(patient_data)
        self.assertIsInstance(result, (bool, np.bool_))

    def test_inference_handles_normal_patient(self):
        """Inference should return False for a patient with normal creatinine levels"""
        # Normal patient: stable creatinine around 75-80, low CV, no significant changes
        # Feature order: Age, Sex, CV, Min, MaxChange, Std, MaxInc, Last>130, IncCount, MaxDec, SincePeak, Change, >1.5xCount
        normal_patient = np.array([[45, 0, 0.03, 75, 2, 2, 3, 0, 1, 2, 2, 3, 0]])
        result = self.model.inference(normal_patient)
        self.assertFalse(result)

    def test_inference_handles_high_risk_patient(self):
        """Inference should return True for a patient with AKI indicators"""
        # High risk: high CV, large change, above 130, multiple >1.5x readings
        # Feature order: Age, Sex, CV, Min, MaxChange, Std, MaxInc, Last>130, IncCount, MaxDec, SincePeak, Change, >1.5xCount
        high_risk_patient = np.array([[70, 1, 0.5, 60, 100, 50, 80, 1, 5, 10, 0, 150, 3]])
        result = self.model.inference(high_risk_patient)
        self.assertTrue(result)

    def test_inference_accepts_2d_array(self):
        """Inference should accept a 2D numpy array with shape (1, n_features)"""
        patient_data = np.array([[50, 1, 0.05, 75, 5, 4, 3, 0, 2, 2, 1, 5, 0]])
        self.assertEqual(patient_data.shape, (1, 13))
        # Should not raise an exception
        result = self.model.inference(patient_data)
        self.assertIsNotNone(result)

    # ---------------------------
    # Model loading behaviour tests
    # ---------------------------

    def test_model_loads_threshold(self):
        """Model should have a threshold value after loading"""
        self.assertIsNotNone(self.model.threshold)
        self.assertIsInstance(self.model.threshold, float)

    def test_model_has_predict_proba(self):
        """Loaded model should have predict_proba method"""
        self.assertTrue(hasattr(self.model.model, 'predict_proba'))


class TestFeatureProcessor(unittest.TestCase):
    """Tests for Feature_Processor class behaviour"""

    def setUp(self):
        """Create a fresh Feature_Processor for each test"""
        self.processor = Feature_Processor()

    # ---------------------------
    # process_features behaviour tests
    # ---------------------------

    def test_process_features_returns_tuple(self):
        """process_features should return a tuple of (dict, ndarray)"""
        row = self._create_patient_row(first=80, last=85, count=3)
        updated_row, features = self.processor.process_features(row, 90)

        self.assertIsInstance(updated_row, dict)
        self.assertIsInstance(features, np.ndarray)

    def test_process_features_returns_correct_shape(self):
        """Feature array should have shape (1, 13) for inference"""
        row = self._create_patient_row(first=80, last=85, count=3)
        _, features = self.processor.process_features(row, 90)

        self.assertEqual(features.shape, (1, 13))

    def test_process_features_updates_last_creatinine(self):
        """Last_Creatinine should be updated to the new result"""
        row = self._create_patient_row(first=80, last=85, count=3)
        updated_row, _ = self.processor.process_features(row, 100)

        self.assertEqual(updated_row["Last_Creatinine"], 100)

    def test_process_features_updates_count(self):
        """Creatinine_Count should increase by 1"""
        row = self._create_patient_row(first=80, last=85, count=3)
        updated_row, _ = self.processor.process_features(row, 90)

        self.assertEqual(updated_row["Creatinine_Count"], 4)

    def test_process_features_updates_min(self):
        """Min_Creatinine should update when new result is lower"""
        row = self._create_patient_row(first=80, last=85, min_val=75, count=3)
        updated_row, _ = self.processor.process_features(row, 70)

        self.assertEqual(updated_row["Min_Creatinine"], 70)

    def test_process_features_keeps_min_when_higher(self):
        """Min_Creatinine should stay the same when new result is higher"""
        row = self._create_patient_row(first=80, last=85, min_val=75, count=3)
        updated_row, _ = self.processor.process_features(row, 90)

        self.assertEqual(updated_row["Min_Creatinine"], 75)

    def test_process_features_updates_max(self):
        """Max_Creatinine should update when new result is higher"""
        row = self._create_patient_row(first=80, last=85, max_val=90, count=3)
        updated_row, _ = self.processor.process_features(row, 100)

        self.assertEqual(updated_row["Max_Creatinine"], 100)

    def test_process_features_calculates_change(self):
        """Creatinine_Change should be new_result - First_Creatinine"""
        row = self._create_patient_row(first=80, last=85, count=3)
        updated_row, _ = self.processor.process_features(row, 100)

        self.assertEqual(updated_row["Creatinine_Change"], 20)  # 100 - 80

    def test_process_features_detects_increase(self):
        """Creatinine_Increase_Count should increment when result increases"""
        row = self._create_patient_row(first=80, last=85, count=3, increase_count=2)
        updated_row, _ = self.processor.process_features(row, 90)  # 90 > 85

        self.assertEqual(updated_row["Creatinine_Increase_Count"], 3)

    def test_process_features_no_increase_when_decrease(self):
        """Creatinine_Increase_Count should not increment when result decreases"""
        row = self._create_patient_row(first=80, last=85, count=3, increase_count=2)
        updated_row, _ = self.processor.process_features(row, 80)  # 80 < 85

        self.assertEqual(updated_row["Creatinine_Increase_Count"], 2)

    def test_process_features_above_1_5x_threshold(self):
        """Creatinine_gt1p5_Count should increment when result > 1.5 * first"""
        row = self._create_patient_row(first=80, last=100, count=3, gt1p5_count=0)
        updated_row, _ = self.processor.process_features(row, 130)  # 130 > 80*1.5=120

        self.assertEqual(updated_row["Creatinine_gt1p5_Count"], 1)

    def test_process_features_below_1_5x_threshold(self):
        """Creatinine_gt1p5_Count should not increment when result <= 1.5 * first"""
        row = self._create_patient_row(first=80, last=100, count=3, gt1p5_count=0)
        updated_row, _ = self.processor.process_features(row, 110)  # 110 < 80*1.5=120

        self.assertEqual(updated_row["Creatinine_gt1p5_Count"], 0)

    def test_process_features_above_130(self):
        """Creatinine_Last_gt130_Count should be 1 when result >= 130"""
        row = self._create_patient_row(first=80, last=100, count=3)
        updated_row, _ = self.processor.process_features(row, 135)

        self.assertEqual(updated_row["Creatinine_Last_gt130_Count"], 1)

    def test_process_features_below_130(self):
        """Creatinine_Last_gt130_Count should be 0 when result < 130"""
        row = self._create_patient_row(first=80, last=100, count=3)
        updated_row, _ = self.processor.process_features(row, 125)

        self.assertEqual(updated_row["Creatinine_Last_gt130_Count"], 0)

    # ---------------------------
    # First reading (None handling) tests
    # ---------------------------

    def test_process_features_handles_first_reading(self):
        """Should initialize values when First_Creatinine is None (new patient)"""
        row = self._create_new_patient_row()
        updated_row, _ = self.processor.process_features(row, 85)

        self.assertEqual(updated_row["First_Creatinine"], 85)
        self.assertEqual(updated_row["Last_Creatinine"], 85)
        self.assertEqual(updated_row["Min_Creatinine"], 85)
        self.assertEqual(updated_row["Max_Creatinine"], 85)

    def test_process_features_handles_none_last_creatinine(self):
        """Should handle None Last_Creatinine without error"""
        row = self._create_new_patient_row()
        # Should not raise an exception
        updated_row, features = self.processor.process_features(row, 85)
        self.assertIsNotNone(updated_row)
        self.assertIsNotNone(features)

    # ---------------------------
    # process_history behaviour tests
    # ---------------------------

    def test_process_history_returns_list(self):
        """process_history should return a list of dictionaries"""
        # Create temp CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("mrn,creatinine_result_1,creatinine_result_2\n")
            f.write("123,80,85\n")
            f.write("456,90,95\n")
            temp_path = f.name

        try:
            result = self.processor.process_history(temp_path)
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 2)
            self.assertIsInstance(result[0], dict)
        finally:
            os.unlink(temp_path)

    def test_process_history_returns_empty_for_missing_file(self):
        """process_history should return empty list for missing file"""
        result = self.processor.process_history("nonexistent_file.csv")
        self.assertEqual(result, [])

    def test_process_history_contains_required_fields(self):
        """Each row should contain PID and creatinine features"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("mrn,creatinine_result_1,creatinine_result_2\n")
            f.write("123,80,85\n")
            temp_path = f.name

        try:
            result = self.processor.process_history(temp_path)
            row = result[0]

            # Check required fields exist
            self.assertIn("PID", row)
            self.assertIn("First_Creatinine", row)
            self.assertIn("Last_Creatinine", row)
            self.assertIn("Min_Creatinine", row)
            self.assertIn("Max_Creatinine", row)
            self.assertIn("Creatinine_Count", row)
        finally:
            os.unlink(temp_path)

    # ---------------------------
    # extract_creatinine_features behaviour tests
    # ---------------------------

    def test_extract_features_calculates_min_max(self):
        """Should correctly calculate min and max creatinine"""
        df = pd.DataFrame({
            "creatinine_result_1": [80, 90],
            "creatinine_result_2": [100, 70],
            "creatinine_result_3": [90, 80]
        })

        features = self.processor.extract_creatinine_features(df)

        # First row: min=80, max=100
        self.assertEqual(features["Min_Creatinine"][0], 80)
        self.assertEqual(features["Max_Creatinine"][0], 100)
        # Second row: min=70, max=90
        self.assertEqual(features["Min_Creatinine"][1], 70)
        self.assertEqual(features["Max_Creatinine"][1], 90)

    def test_extract_features_calculates_first_last(self):
        """Should correctly identify first and last creatinine"""
        df = pd.DataFrame({
            "creatinine_result_1": [80],
            "creatinine_result_2": [90],
            "creatinine_result_3": [100]
        })

        features = self.processor.extract_creatinine_features(df)

        self.assertEqual(features["First_Creatinine"][0], 80)
        self.assertEqual(features["Last_Creatinine"][0], 100)

    def test_extract_features_counts_readings(self):
        """Should correctly count number of readings"""
        df = pd.DataFrame({
            "creatinine_result_1": [80],
            "creatinine_result_2": [90],
            "creatinine_result_3": [100]
        })

        features = self.processor.extract_creatinine_features(df)

        self.assertEqual(features["Creatinine_Count"][0], 3)

    def test_extract_features_handles_nan(self):
        """Should handle NaN values in creatinine readings"""
        df = pd.DataFrame({
            "creatinine_result_1": [80],
            "creatinine_result_2": [np.nan],
            "creatinine_result_3": [100]
        })

        features = self.processor.extract_creatinine_features(df)

        # Count should be 2 (excluding NaN)
        self.assertEqual(features["Creatinine_Count"][0], 2)
        # Last should be forward-filled to 100
        self.assertEqual(features["Last_Creatinine"][0], 100)

    # ---------------------------
    # Helper methods
    # ---------------------------

    def _create_patient_row(self, first=80, last=85, min_val=None, max_val=None,
                            count=3, mean=None, std=0, increase_count=0,
                            gt1p5_count=0, age=50, sex=1):
        """Helper to create a patient row dictionary"""
        if min_val is None:
            min_val = min(first, last)
        if max_val is None:
            max_val = max(first, last)
        if mean is None:
            mean = (first + last) / 2

        return {
            "Age": age,
            "Sex": sex,
            "First_Creatinine": first,
            "Last_Creatinine": last,
            "Min_Creatinine": min_val,
            "Max_Creatinine": max_val,
            "Creatinine_Count": count,
            "Creatinine_Mean": mean,
            "Creatinine_Std": std,
            "Creatinine_Change": last - first,
            "Creatinine_CV": 0.1,
            "Creatinine_Max_Change": max_val - last,
            "Creatinine_Max_Consec_Increase": 0,
            "Creatinine_Max_Consec_Decrease": 0,
            "Creatinine_Increase_Count": increase_count,
            "Creatinine_Count_Since_Peak": 0,
            "Creatinine_gt1p5_Count": gt1p5_count,
            "Creatinine_Last_gt130_Count": 0
        }

    def _create_new_patient_row(self, age=50, sex=1):
        """Helper to create a new patient row with None values (no prior data)"""
        return {
            "Age": age,
            "Sex": sex,
            "First_Creatinine": None,
            "Last_Creatinine": None,
            "Min_Creatinine": 999999.0,
            "Max_Creatinine": 0.0,
            "Creatinine_Count": 0,
            "Creatinine_Mean": 0.0,
            "Creatinine_Std": 0.0,
            "Creatinine_Change": 0.0,
            "Creatinine_CV": 0.0,
            "Creatinine_Max_Change": 0.0,
            "Creatinine_Max_Consec_Increase": 0.0,
            "Creatinine_Max_Consec_Decrease": 0.0,
            "Creatinine_Increase_Count": 0,
            "Creatinine_Count_Since_Peak": 0,
            "Creatinine_gt1p5_Count": 0,
            "Creatinine_Last_gt130_Count": 0
        }


if __name__ == "__main__":
    unittest.main()
