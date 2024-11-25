import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, List

# Core Components
class GeometricBinner:
    def __init__(self):
        self.eodb_bins = np.geomspace(0.1, 150.0, num=50)
        self.duration_bins = np.geomspace(5, 90, num=20)
        self.slope_bins = np.linspace(-2.0, 2.0, num=30)
        self.curvature_bins = np.linspace(-0.5, 0.5, num=25)
        
    def bin_eodb(self, value):
        return np.digitize(value, self.eodb_bins)
        
    def bin_duration(self, days):
        return np.digitize(days, self.duration_bins)
        
    def bin_slope(self, slope):
        return np.digitize(slope, self.slope_bins)
        
    def bin_curvature(self, curvature):
        return np.digitize(curvature, self.curvature_bins)

@dataclass
class StateFeatures:
    eodb_level: int
    duration: int
    trend: int
    curvature: int

class StateMapper:
    def __init__(self, binner: GeometricBinner):
        self.binner = binner
        self.state_map: Dict[Tuple[int, int, int, int], int] = {}
        self.next_state_id = 0
        
    def get_state_id(self, features: StateFeatures) -> int:
        feature_tuple = (
            features.eodb_level,
            features.duration,
            features.trend,
            features.curvature
        )
        
        if feature_tuple not in self.state_map:
            self.state_map[feature_tuple] = self.next_state_id
            self.next_state_id += 1
            
        return self.state_map[feature_tuple]
        
    def map_segment_to_state(self, segment_data: np.ndarray, 
                            segment_duration: float,
                            macd_line: np.ndarray) -> int:
        eodb_level = self.binner.bin_eodb(np.mean(segment_data))
        duration = self.binner.bin_duration(segment_duration)
        trend = self.binner.bin_slope(np.polyfit(range(len(segment_data)), 
                                                segment_data, 1)[0])
        curvature = self.binner.bin_curvature(np.polyfit(range(len(macd_line)), 
                                                        macd_line, 2)[0])
        
        features = StateFeatures(eodb_level, duration, trend, curvature)
        return self.get_state_id(features)
        
    def decode_state(self, state_id: int) -> StateFeatures:
        for feature_tuple, sid in self.state_map.items():
            if sid == state_id:
                return StateFeatures(*feature_tuple)
        raise ValueError(f"Unknown state ID: {state_id}")

class TransitionMatrix:
    def __init__(self, n_states):
        self.matrix = np.zeros((n_states, n_states))
        self.counts = np.zeros((n_states, n_states))
        
    def update(self, from_state: int, to_state: int):
        self.counts[from_state, to_state] += 1
        self.matrix = self.counts / self.counts.sum(axis=1, keepdims=True)
        
    def save(self, filepath: Path):
        np.savez(filepath, 
                 matrix=self.matrix, 
                 counts=self.counts)
    
    @classmethod
    def load(cls, filepath: Path):
        data = np.load(filepath)
        tm = cls(len(data['matrix']))
        tm.matrix = data['matrix']
        tm.counts = data['counts']
        return tm

class BayesianPredictor:
    def __init__(self, transition_matrix: TransitionMatrix):
        self.tm = transition_matrix
        
    def predict_next_state(self, current_state: int, n_steps: int = 1) -> np.ndarray:
        current_dist = np.zeros(len(self.tm.matrix))
        current_dist[current_state] = 1.0
        
        for _ in range(n_steps):
            current_dist = current_dist @ self.tm.matrix
            
        return current_dist

class MACDValidator:
    def __init__(self, min_data_points=30, min_segment_length=3):
        self.min_data_points = min_data_points
        self.min_segment_length = min_segment_length
        
    def validate_input(self, data: np.ndarray) -> bool:
        return (len(data) >= self.min_data_points and 
                not np.any(np.isnan(data)) and 
                not np.any(np.isinf(data)))

def calculate_macd_robust(AN: np.ndarray, 
                         fast_period: int, 
                         slow_period: int, 
                         threshold: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    validator = MACDValidator()
    
    if not validator.validate_input(AN):
        raise ValueError("Input data validation failed")
    if fast_period >= slow_period:
        raise ValueError("Fast period must be less than slow period")
    if threshold <= 0:
        raise ValueError("Threshold must be positive")
        
    AN_std = np.std(AN)
    if AN_std == 0:
        raise ValueError("Zero standard deviation in input data")
    AN_normalized = (AN - np.mean(AN)) / AN_std
    
    adaptive_threshold = threshold * np.std(AN_normalized)
    
    try:
        fast_ema = pd.Series(AN_normalized).ewm(
            span=fast_period, 
            min_periods=fast_period,
            adjust=False
        ).mean()
    
        slow_ema = pd.Series(AN_normalized).ewm(
            span=slow_period, 
            min_periods=slow_period,
            adjust=False
        ).mean()
    
        macd_line = fast_ema - slow_ema
        
        above_threshold = macd_line > adaptive_threshold
        transitions = np.diff(above_threshold.astype(int))
        segment_starts = np.where(transitions == 1)[0]
        segment_ends = np.where(transitions == -1)[0]
        
        if len(segment_ends) < len(segment_starts):
            segment_ends = np.append(segment_ends, len(macd_line)-1)
        if len(segment_starts) < len(segment_ends):
            segment_starts = np.insert(segment_starts, 0, 0)
            
        valid_segments = []
        for start, end in zip(segment_starts, segment_ends):
            if end - start >= validator.min_segment_length:
                valid_segments.append((start, end))
                
        if valid_segments:
            segment_starts = np.array([s[0] for s in valid_segments])
            segment_ends = np.array([s[1] for s in valid_segments])
        else:
            segment_starts = np.array([])
            segment_ends = np.array([])
        
        return segment_starts, segment_ends, macd_line, fast_ema, slow_ema
        
    except Exception as e:
        raise RuntimeError(f"MACD calculation failed: {str(e)}")

def process_eodb_batch_with_transitions(eodb_files, state_mapper, transition_matrix):
    for eodb_file in eodb_files:
        eodb_data = pd.read_csv(eodb_file)
        segments = processor.process_segments(
            eodb_data['balance'],
            eodb_data['timestamps']
        )
        
        states = []
        for i in range(len(segments)-1):
            current_state = state_mapper.map_segment_to_state(
                segments[i].data,
                segments[i].duration,
                segments[i].macd_line
            )
            next_state = state_mapper.map_segment_to_state(
                segments[i+1].data,
                segments[i+1].duration,
                segments[i+1].macd_line
            )
            transition_matrix.update(current_state, next_state)
            states.append(current_state)
            
    return transition_matrix

# Main execution
if __name__ == "__main__":
    # Initialize components
    binner = GeometricBinner()
    state_mapper = StateMapper(binner)
    
    # Set up paths
    data_dir = Path('data/raw_eodb')
    output_dir = Path('data/processed_features')
    output_dir.mkdir(exist_ok=True)
    
    # Get list of EODB files
    eodb_files = list(data_dir.glob('*.csv'))
    
    # Initialize transition matrix
    transition_matrix = TransitionMatrix(n_states=1000)  # Initial size estimate
    
    # Process in batches
    batch_size = 1000
    for i in range(0, len(eodb_files), batch_size):
        batch = eodb_files[i:i + batch_size]
        transition_matrix = process_eodb_batch_with_transitions(
            batch, state_mapper, transition_matrix
        )
    
    # Save the trained transition matrix
    transition_matrix.save(output_dir / 'transition_matrix.npz')
    
    # Initialize predictor
    predictor = BayesianPredictor(transition_matrix)
