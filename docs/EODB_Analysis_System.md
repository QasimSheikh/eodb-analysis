
# EODB Analysis System Documentation

## 1. System Overview
The EODB (End of Day Balance) Analysis System is a robust tool designed for analyzing banking account behaviors through state transitions and Bayesian prediction. It processes time series data in stages to identify patterns and predict future states, aiding in behavioral insights and decision-making.

---

## 2. Component Documentation

### 2.1 **GeometricBinner**
- **Purpose:** Provides non-linear binning for continuous features.
- **Key Features:**
  - Bins EODB values (range: 0.1 to 150.0).
  - Bins duration periods (range: 5 to 90 days).
  - Handles slope and curvature binning for trends.
  - Uses geometric spacing to enhance resolution and analysis.

---

### 2.2 **StateMapper**
- **Purpose:** Maps features from data segments into discrete states.
- **Features:**
  - Supports dynamic state creation based on input features.
  - Manages feature combinations to ensure accurate state mapping.
  - Allows bidirectional mapping (features ↔ states).

---

### 2.3 **TransitionMatrix**
- **Purpose:** Tracks probabilities of state transitions within the dataset.
- **Features:**
  - Maintains a probability matrix for state transitions.
  - Tracks transition counts for statistical updates.
  - Includes persistence functionality for saving and loading matrices.

---

### 2.4 **BayesianPredictor**
- **Purpose:** Predicts future states based on historical data and state transitions.
- **Features:**
  - Performs multi-step predictions.
  - Outputs probability distributions for predicted states.
  - Utilizes Markov chain-based calculations for accuracy.

---

## 3. Usage Examples

### 3.1 **Basic Usage**
```python
# Initialize components
binner = GeometricBinner()
state_mapper = StateMapper(binner)
transition_matrix = TransitionMatrix(n_states=1000)

# Process raw EODB data
transition_matrix = process_eodb_batch_with_transitions(
    eodb_files, state_mapper, transition_matrix
)

# Make predictions
predictor = BayesianPredictor(transition_matrix)
next_states = predictor.predict_next_state(current_state, n_steps=3)
```

---

## 4. API Reference

### 4.1 **GeometricBinner**
- `bin_eodb(value)`: Bins EODB values into pre-defined ranges.
- `bin_duration(days)`: Bins durations into specified ranges.
- `bin_slope(slope)`: Bins slopes of trends in data segments.
- `bin_curvature(curvature)`: Bins curvatures for non-linear segment analysis.

### 4.2 **StateMapper**
- `get_state_id(features)`: Maps input features to a unique state ID.
- `map_segment_to_state(segment_data, duration, macd_line)`: Performs complete mapping for a data segment.
- `decode_state(state_id)`: Converts a state ID back into its feature representation.

### 4.3 **TransitionMatrix**
- `update(from_state, to_state)`: Updates the transition matrix with observed transitions.
- `save(filepath)`: Saves the matrix to disk for future use.
- `load(filepath)`: Loads a persisted transition matrix.

### 4.4 **BayesianPredictor**
- `predict_next_state(current_state, n_steps)`: Predicts the next state(s) and their probabilities.

---

## 5. Implementation Details

### 5.1 **Data Processing Pipeline**
1. **Data Ingestion:** Processes raw EODB data.
2. **Segment Identification:** Detects segments using indicators like MACD.
3. **Feature Extraction:** Extracts features for each segment.
4. **Feature Binning:** Bins features using `GeometricBinner`.
5. **State Mapping:** Maps binned features to discrete states.
6. **Matrix Updates:** Updates transition probabilities.
7. **Prediction:** Generates predictions with Bayesian inference.

---

### 5.2 **State Space Management**
- Combines features to form unique states.
- Dynamically creates new states as needed.
- Maintains an efficient mapping structure for fast lookups.

---

### 5.3 **Prediction Mechanism**
- Relies on Markov chain principles for state transitions.
- Supports multi-step predictions with confidence outputs.
- Provides probabilities for all potential next states.

---

## 6. Best Practices

### 6.1 **Data Preparation**
- Ensure at least 30 data points for meaningful analysis.
- Clean data by removing NaN/Inf values.
- Normalize the time series for consistency.

### 6.2 **Model Training**
- Use batch processing for large datasets.
- Regularly update the transition matrix with new data.
- Persist the matrix periodically to avoid data loss.

### 6.3 **Prediction Usage**
- Validate input states for consistency.
- Check prediction confidence before usage.
- Use appropriate step counts based on desired forecasting horizon.

---

## 7. Performance Considerations

### 7.1 **Memory Usage**
- Memory scales with the size of the state space.
- Use sparse transition matrices for efficient storage.

### 7.2 **Computational Efficiency**
- Geometric binning ensures quick lookups.
- Optimize matrix operations for sparse data.
- Utilize dictionary-based structures for state mapping.
