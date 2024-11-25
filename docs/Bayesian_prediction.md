# Mathematical Framework for EODB State Prediction

## State Space Definition
Let S = {s₁, s₂, ..., sₙ} be the set of all possible states
where each state sᵢ represents a unique combination of:
- Binned EODB level
- Binned Duration
- Binned Slope
- Binned Curvature

## Transition Matrix
P(i,j) = P(sⱼ|sᵢ) : probability of transitioning from state sᵢ to state sⱼ

The transition matrix P contains these probabilities for all possible state pairs.

## Current State Vector
π₀ = [0,...,1,...,0] where 1 is at position i if current state is sᵢ

This vector represents our current knowledge of the system state.

## k-step Prediction
π₍ₖ₎ = π₀ Pᵏ

where:
- π₍ₖ₎ is the probability distribution over states after k steps
- Pᵏ represents the transition matrix raised to power k

## Posterior Probability
For any future state sⱼ:
P(sⱼ|s₍ₜ₎=sᵢ) = [π₀ Pᵏ]ⱼ

This gives us the probability of being in state sⱼ after k steps, given that we started in state sᵢ.
