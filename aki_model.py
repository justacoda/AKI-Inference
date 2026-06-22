# Relevant imports
import pickle
import logging
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score

class Inference_Model:
    def __init__(self, 
                 model_path:str=None,
                 training_filepath:str=None,
                 training_config:dict=None,
                 best_model_path:str=None):
        """
        Class to host the inference model

        Args:
            model_path (str, optional): Model weights path. Defaults to None if training is to be ran.
            training_filepath (str, optional): Training filepath. Defaults to None if no training.
            training_config (dict, optional): Dictionary of training config, format must follow the following:
                {
                'random_state': int of seed for train test split
                'max_iter': int of max iter in logistic regression
                'class_weight': dictionary containing class weights
                'thresholds': list containing all possible thresholds to test for model training.
                    threshold determines the probability of the output for a positive prediction
                    eg. threshold of 0.3 means all output >=0.3 will be positive
                }. 
            best_model_path (str, optional): filepath to save trained model. Defaults to None to not save the model.
        """

        self.logger = logging.getLogger('INFERENCE_MODEL')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        # Pretrained model exist
        if model_path != None:
            with open(model_path, 'rb') as f:
                saved = pickle.load(f)
                self.model = saved['model']
                self.threshold = saved['threshold']

        else:
            self.train(training_filepath, training_config)
            if best_model_path != None:
                self._save_model(best_model_path)


    def inference(self, patient_data:np.ndarray):
        """
        Function to run inference, postive class depends on self.threshold

        Returns:
            boolean, True for positive classification
        """
        prob = self.model.predict_proba(patient_data)
        has_aki = prob[0][1] >= self.threshold
        return has_aki
    

    def train(self, training_filepath:str, training_config:dict):
        """
        Function to train model
        """
        # Config file
        random_state = training_config.get('random_state', 42)
        max_iter = training_config.get('max_iter', 2000)
        class_weight = training_config.get('class_weight', {0:1, 1:10})
        thresholds = training_config.get('thresholds', [0.5])
        # Generate training file 
        train_df = self._load_dataframe(training_filepath)
        x, y = self._prepare_training_data(train_df)
        # Generate train & val data
        x_train, x_val, y_train, y_val = train_test_split(
            x, y, test_size=0.2, random_state=random_state, stratify=y
        )
        model = self._create_model(max_iter, class_weight)
        model.fit(x_train, y_train)
        # Extract output prob from model prediction 
        y_proba = model.predict_proba(x_val)[:,1]

        # Iterate and find best threshold based on f3 score
        best_threshold = None
        best_f3 = 0

        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            f3 = self.evaluate_f3(y_val, y_pred, beta=3)
            if f3 > best_f3:
                best_f3 = f3
                best_threshold = threshold

        self.threshold = best_threshold
        self.logger.info(f"Best model at threshold: {self.threshold} with f3 {best_f3}")
        
        # Now fully train model on x and y (no need for val)
        self.model = self._create_model(max_iter, class_weight)
        self.model.fit(x, y)
    
    def evaluate_f3(self, y_true: np.ndarray, y_pred: np.ndarray):
        return fbeta_score(y_true, y_pred, beta=3)

    def _save_model(self, filepath:str):
        """
        Function to save model and best threshold as a pickle file
        """
        try:
            with open(filepath, 'wb') as f:
                pickle.dump({'model': self.model, 'threshold': self.threshold}, f)
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")


    def _create_model(self, max_iter:int=2000, class_weight:dict={0: 1.0, 1: 10.0}):
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=max_iter, class_weight=class_weight)),
        ])
    
    # ------------------------------
    # Training Helper Functions
    # ------------------------------
    def _load_dataframe(self, filepath:str):
        """
        Function to load training data into a pd.Dataframe
        """
        try:
            df = pd.read_csv(filepath)
        except FileNotFoundError:
            raise FileNotFoundError(f"Training file not found: {filepath}")
        
        required_cols = ["age", "sex", "aki"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return df
    

    def _prepare_training_data(self, data:pd.DataFrame):
        """
        Function to preprocess dataframe to training data
    
        Returns (pd.Dataframe, pd.Series): x and y
        """
        # Format and extract age & sex
        age = data["age"].to_numpy()
        sex = data["sex"].map({"f": 0, "m": 1}).to_numpy()

        # Retrieve just the creatinine results data
        result_cols = [c for c in data.columns if c.startswith("creatinine_result_")]
        if not result_cols:
            raise ValueError("No creatinine result columns found")
        results_df = data[result_cols]

        # Extract features
        feature_processor = Feature_Processor()
        features = feature_processor.extract_creatinine_features(results_df)

        # Format x 
        x = np.column_stack([
            age,
            sex,
            features["Creatinine_CV"],
            features["Min_Creatinine"],
            features["Creatinine_Max_Change"],
            features["Creatinine_Std"],
            features["Creatinine_Max_Consec_Increase"],
            features["Creatinine_Last_gt130_Count"],
            features["Creatinine_Increase_Count"],
            features["Creatinine_Max_Consec_Decrease"],
            features["Creatinine_Count_Since_Peak"],
            features["Creatinine_Change"],
            features["Creatinine_gt1p5_Count"],
        ])

        # Format y
        aki_col = data["aki"]
        if aki_col.dtype == object:
            y = aki_col.map({"n": 0, "y": 1}).to_numpy()
        else:
            y = aki_col.to_numpy()

        return x, y

    

class Feature_Processor:
    def __init__(self):
        """
        System Engine that preprocess database results and new results to generate data for inference
        """
        self.inference_features_order = [
                                            "Age",
                                            "Sex",
                                            "Creatinine_CV",
                                            "Min_Creatinine",
                                            "Creatinine_Max_Change",
                                            "Creatinine_Std",
                                            "Creatinine_Max_Consec_Increase",
                                            "Creatinine_Last_gt130_Count",
                                            "Creatinine_Increase_Count",
                                            "Creatinine_Max_Consec_Decrease",
                                            "Creatinine_Count_Since_Peak",
                                            "Creatinine_Change",
                                            "Creatinine_gt1p5_Count",
                                        ]


    def process_history(self, history_filepath:str):
        """
        Function to process history.csv into required features for database

        Args:
            history_filepath (str): filepath to history.csv
        
        Returns:
            List[dict]: Preprocessed data as a list of dictionaries, 
                where each dictionary represents a single row.
                Empty list will be returned if there is no history_filepath
        """
        try:
            df = pd.read_csv(history_filepath)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            return []
        except Exception as e:
            print(f"ERROR: Failed to process history file {history_filepath}: {e}")
            raise
        
        result_cols = [c for c in df.columns if c.startswith("creatinine_result_")]
        results_df = df[result_cols]

        creatinine_features  = self.extract_creatinine_features(results_df)

        mrn = df.iloc[:,0].to_numpy()
        age = np.full(len(df), np.nan)
        sex = np.full(len(df), np.nan)
        df_features = pd.DataFrame({
            "PID": mrn,
            "Age": age,
            "Sex": sex,
            **creatinine_features
        })

        float_cols = df_features.select_dtypes(include=['float64']).columns
        df_features[float_cols] = df_features[float_cols].round(5)

        return df_features.to_dict(orient="records")

    
    def process_features(self, row:dict, new_result:float):
        """
        Function to update a database row with a new creatinine measurement.

        Args:
            row (dict): Dictionary representing a database row, with keys as column names and values as column values.
            new_reading (float): The new creatinine measurement to incorporate.

        Returns:
            Tuple[dict, np.ndarray]:
                - Updated row dictionary with recalculated statistics, ready to be written back to the database.
                - Feature array shaped (1, n_features), ordered according to self.inference_features_order,convenient for direct use in model inference.
        """
        # Update with newest creatinine result
        creatinine_second_last = row["Last_Creatinine"]
        if creatinine_second_last is None:
            creatinine_second_last = new_result  # No prior value, treat as no change
        row["Last_Creatinine"] = new_result

        # Handle first reading - initialize all None values
        if row["First_Creatinine"] is None:
            row["First_Creatinine"] = new_result
            row["Min_Creatinine"] = new_result
            row["Max_Creatinine"] = new_result
            row["Creatinine_Mean"] = new_result


        # Update Min / Max
        row["Min_Creatinine"] = new_result if new_result < row["Min_Creatinine"] else row["Min_Creatinine"]
        row["Max_Creatinine"] = new_result if new_result > row["Max_Creatinine"] else row["Max_Creatinine"]

        # Welford's algo for running mean and std
        row["Creatinine_Count"] += 1
        delta = new_result - row["Creatinine_Mean"]
        row["Creatinine_Mean"] += delta / row["Creatinine_Count"]

        if row["Creatinine_Count"] > 1:
            M2 = row["Creatinine_Std"]**2 * (row["Creatinine_Count"] - 2)
            M2 += delta * (new_result - row["Creatinine_Mean"])
            row["Creatinine_Std"] = (M2 / (row["Creatinine_Count"] - 1))**0.5
        else:
            row["Creatinine_Std"] = 0.0

        row["Creatinine_CV"] = (row["Creatinine_Std"] / (row["Creatinine_Mean"] + 1e-6))

        # Trend features
        row["Creatinine_Change"] = new_result - row["First_Creatinine"]
        row["Creatinine_Max_Change"] = row["Max_Creatinine"] - new_result

        if new_result - creatinine_second_last > row["Creatinine_Max_Consec_Increase"]:
            row["Creatinine_Max_Consec_Increase"] = new_result - creatinine_second_last
            
        if creatinine_second_last - new_result > row["Creatinine_Max_Consec_Decrease"]:
            row["Creatinine_Max_Consec_Decrease"] = creatinine_second_last - new_result

        if new_result > creatinine_second_last:
            row["Creatinine_Increase_Count"] += 1

        row["Creatinine_Count_Since_Peak"] = 0 if new_result == row["Max_Creatinine"] else row["Creatinine_Count_Since_Peak"] + 1

        if row["First_Creatinine"] * 1.5 < new_result:
            row["Creatinine_gt1p5_Count"] += 1
        
        row["Creatinine_Last_gt130_Count"] = int(new_result >= 130)

        # Extract features used in inference
        row_array = np.array([row[feat] for feat in self.inference_features_order]).reshape(1, -1)
        for key, value in row.items():
            if isinstance(value, float):
                row[key] = round(value, 5)

        return row, row_array


    # Helper methods for extract creatinine results for process history
    # Used for pandas dataframe 
    def _max_consecutive_increase(self, row):
        """Find maximum single-step increase in creatinine values."""
        vals = row.dropna().values
        if len(vals) < 2:
            return 0.0
        diffs = np.diff(vals)
        return max(diffs.max(), 0)


    def _max_consecutive_decrease(self, row):
        """Find maximum single-step decrease in creatinine values."""
        vals = row.dropna().values
        if len(vals) < 2:
            return 0.0
        diffs = np.diff(vals)
        return abs(min(diffs.min(), 0))


    def _count_increases(self, row):
        """Count the number of increases between consecutive readings."""
        vals = row.dropna().values
        if len(vals) < 2:
            return 0
        diffs = np.diff(vals)
        return (diffs > 0).sum()


    def _readings_since_peak(self, row):
        """Count number of readings since the peak value."""
        vals = row.dropna().values
        if len(vals) == 0:
            return 0
        max_idx = np.argmax(vals)
        return len(vals) - 1 - max_idx


    def extract_creatinine_features(self, results_df:pd.DataFrame):
        """
        Function that extracts all the creatinine features from history.csv

        Args:
            results_df (pd.DataFrame): dataframe containing just the creatinine results

        Returns:
            dictionary of creatinine results, each as a pd.Series
        """
        results_df_ffill = results_df.ffill(axis=1)

        # Basic statistics
        creatinine_last = results_df_ffill.iloc[:, -1].to_numpy()
        creatinine_first = results_df.iloc[:, 0].to_numpy()
        creatinine_min = results_df.min(axis=1, skipna=True).to_numpy()
        creatinine_max = results_df.max(axis=1, skipna=True).to_numpy()
        creatinine_mean = results_df.mean(axis=1, skipna=True).to_numpy()
        creatinine_count = results_df.count(axis=1).to_numpy()
        creatinine_std = results_df.std(axis=1, skipna=True).fillna(0).to_numpy()
        creatinine_change = creatinine_last - creatinine_first
        creatinine_cv = creatinine_std / (creatinine_mean + 1e-6)

        # Trend features
        distance_from_peak = creatinine_max - creatinine_last
        max_consec_increase = results_df_ffill.apply(self._max_consecutive_increase, axis=1).to_numpy()
        max_consec_decrease = results_df_ffill.apply(self._max_consecutive_decrease, axis=1).to_numpy()
        n_increases = results_df_ffill.apply(self._count_increases, axis=1).to_numpy()
        n_since_peak = results_df.apply(self._readings_since_peak, axis=1)

        # Threshold-based features
        n_above_1_5x_first = (results_df.gt(results_df.iloc[:, 0] * 1.5, axis=0) & results_df.notna()).sum(axis=1).to_numpy()
        last_above_130 = (creatinine_last >= 130).astype(int)

        return {
            "Last_Creatinine": creatinine_last,
            "First_Creatinine": creatinine_first,
            "Min_Creatinine": creatinine_min,
            "Max_Creatinine": creatinine_max,
            "Creatinine_Mean": creatinine_mean,
            "Creatinine_Count": creatinine_count,
            "Creatinine_Std": creatinine_std,
            "Creatinine_Change": creatinine_change,
            "Creatinine_CV": creatinine_cv,
            "Creatinine_Max_Change": distance_from_peak,
            "Creatinine_Max_Consec_Increase": max_consec_increase,
            "Creatinine_Max_Consec_Decrease": max_consec_decrease,
            "Creatinine_Increase_Count": n_increases,
            "Creatinine_Count_Since_Peak": n_since_peak,
            "Creatinine_gt1p5_Count": n_above_1_5x_first,
            "Creatinine_Last_gt130_Count": last_above_130,
        }
