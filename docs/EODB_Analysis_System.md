# EODB Analysis System Documentation

## 1. System Overview
The EODB (End of Day Balance) Analysis System is a sophisticated pipeline for analyzing banking account behaviors using state transitions and Bayesian prediction. The system processes time series data through multiple stages to identify patterns and predict future states.

## 2. Component Documentation

### 2.1 GeometricBinner
- Purpose: Non-linear binning of continuous features
- Key Features:
  - EODB value binning (0.1 to 150.0)
  - Duration binning (5 to 90 days)
  - Slope and curvature binning
  - Geometric spacing for better resolution

### 2.2 StateMapper
- Purpose: Maps segment features to discrete states
- Features:
  - Dynamic state creation
  - Feature combination handling
  - Bidirectional mapping (features ↔ states)

### 2.3 TransitionMatrix
- Purpose: Tracks state transition probabilities
- Features:
  - Probability matrix maintenance
  - Count tracking
  - Persistence capabilities
  - Loading/saving functionality

### 2.4 BayesianPredictor
- Purpose: Generates future state predictions
- Features:
  - Multi-step prediction
  - Probability distribution output
  - Markov chain-based calculations

## 3. Usage Examples

### 3.1 Basic Usage
```python
# Initialize components
binner = GeometricBinner()
state_mapper = StateMapper(binner)
transition_matrix = TransitionMatrix(n_states=1000)

# Process data
transition_matrix = process_eodb_batch_with_transitions(
    eodb_files, state_mapper, transition_matrix
)

# Make predictions
predictor = BayesianPredictor(transition_matrix)
next_states = predictor.predict_next_state(current_state, n_steps=3)
