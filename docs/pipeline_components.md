# EODB Analysis Pipeline Components

## 1. Data Generation (generate_eodb)
- Creates synthetic EODB time series
- Implements random walk with drift
- Generates multiple series for analysis
- Output: Raw EODB time series data

## 2. MACD Segmentation (segment_macd)
- Applies MACD indicator to identify trend changes
- Segments EODB series at MACD crossover points
- Calculates segment properties
- Output: Segmented EODB data with properties

## 3. Geometric Binning (geometric_binning)
- Bins segment features into discrete states
- Features: EODB level, Duration, Slope, Curvature
- Uses geometric spacing for bin edges
- Output: Binned feature data

## 4. Transition Matrix (transition_matrix)
- Builds state transition probability matrix
- Maps sequences of binned states
- Calculates transition probabilities
- Output: State transition matrix

## 5. Bayesian Predictor (predict)
- Uses transition matrix for state predictions
- Implements n-step ahead predictions
- Calculates prediction probabilities
- Output: State predictions and confidence levels

## Data Flow
Raw Data → Segments → Binned States → Transition Matrix → Predictions

## Key Features
- Modular design
- Consistent data structures
- Probabilistic approach
- Visualization capabilities
