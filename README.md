# Multi-Modal Perception
<img width="1921" height="976" alt="test_result" src="https://github.com/user-attachments/assets/a172d3b0-f5f1-46a6-8b2b-712604f69291" />

A Python-based robot perception system combining multi-modal tactile sensing with force/brake control. This project demonstrates perception model training and constraint-aware force control for robotic applications.

## Overview

This repository contains two complementary perception models:

1. **Multi-Modal Tactile Perception** - A neural network that fuses tactile images and pressure matrices to infer contact state parameters (normal force, contact area, displacement, position)
2. **Force Brake Control** - A constraint-aware model that ensures force outputs respect friction constraints during robot manipulation

## Project Structure

```
multi_modal_perception/
├── multi_modal/
│   ├── model.py                 # Tactile perception NN with training & test
│   ├── explaination.md          # Model risks and considerations
│   ├── test_result.png          # Visualization of inference results
│   └── train.txt                # Training logs
│
├── force_brake/
│   ├── model.py                 # Force brake constraint model with test
│   ├── explaination.md          # Numerical stability handling
│   └── [model outputs]
│
└── README.md
```

## Models

### 1. Multi-Modal Tactile Perception (`multi_modal/model.py`)

A dual-stream convolutional neural network that processes tactile data:

**Input:**
- `tactile_img`: Tactile displacement field (2×64×64)
- `pressure_mat`: Pressure distribution (16×16)

**Output:**
- 5-dimensional state vector: `[F_n, A, delta, x_c, y_c]`
  - `F_n`: Normal force (N)
  - `A`: Contact area (mm²)
  - `delta`: Displacement (mm)
  - `x_c, y_c`: Contact center position (pixels)

**Architecture:**
- Tactile stream: Conv2d → ReLU → MaxPool → Conv2d → ReLU → MaxPool → Conv2d → AdaptiveAvgPool (64 features)
- Pressure stream: Conv2d → ReLU → Conv2d → ReLU → AdaptiveAvgPool (32 features)
- Fusion: Concatenate features → FC layers → 5D output

**Running the test:**
```bash
cd multi_modal
python model.py
```

The test generates synthetic tactile data, trains the model for 2000 epochs, and visualizes inference results comparing ground truth vs predictions on sample inputs.

### 2. Force Brake Control (`force_brake/model.py`)

A constraint enforcement model for robot force control:

**Input:**
- `force`: 3D force vector (batch × 3) → `[n, fx, fy]`
  - `n`: Normal force
  - `fx, fy`: Tangential force components

**Output:**
- Constrained force vector respecting friction limits

**Key Features:**
- Friction constraint: `|f_tangential| ≤ μ × n_normal`
- Numerical stability with epsilon clamping
- Gradient-safe scaling for differentiable optimization

**Running the test:**
```bash
cd force_brake
python model.py
```

The test validates:
- ✓ Constraint satisfaction (tangential force ≤ friction limit)
- ✓ Numerical stability (no NaN/Inf gradients)
- ✓ Backward pass correctness through 1000 test iterations

## Requirements

- Python 3.8+
- PyTorch ≥ 1.9
- NumPy
- Matplotlib
- CUDA (optional, for GPU acceleration)

## Installation

```bash
# Clone the repository
git clone https://github.com/whe128/multi_modal_perception.git
cd multi_modal_perception

# Install dependencies
pip install torch numpy matplotlib
```

## Quick Start

### Run Tactile Perception Model
```bash
cd multi_modal
python model.py
```
This will:
1. Generate 5120 synthetic tactile samples
2. Train the PressPerceptionNN for 2000 epochs
3. Evaluate accuracy on held-out test set
4. Display inference visualizations

### Run Force Brake Model
```bash
cd force_brake
python model.py
```
This will:
1. Run 1000 test iterations with random force inputs
2. Validate friction constraints
3. Check gradient stability
4. Print pass/fail results

## Model Details

### Multi-Modal Tactile Perception

**Dataset Generation:**
- Synthetic data with randomized contact parameters
- Realistic tactile image simulation with Gaussian displacement fields
- Pressure matrix generation with physical constraints

**Training:**
- MSE loss between predicted and ground truth state
- Adam optimizer with learning rate scheduling
- Batch size: 64, Epochs: 2000

**Evaluation:**
- Per-dimension accuracy (5% error threshold)
- Per-sample visualization comparing tactile/pressure inputs with predictions

**Known Limitations (see `multi_modal/explaination.md`):**
- Sim-to-real distribution gap (idealized Gaussian noise)
- Calibration sensitivity in real systems
- Limited to single-point contact assumption

### Force Brake Control

**Physics Constraint:**
The model enforces the Coulomb friction constraint:
```
f_tangential_magnitude ≤ μ × n_normal
```

**Numerical Stability:**
- Clamped normal force: `n = max(n, ε)` prevents division by zero
- Epsilon in denominator: `f_tan + ε` for stable scaling
- Hard constraint check ensures constraint satisfaction

**See `force_brake/explaination.md` for detailed numerical stability analysis.**

## Results

### Multi-Modal Perception
- Test visualization saved as `multi_modal/test_result.png`
- Shows 5 tactile images with corresponding pressure matrices
- Displays ground truth vs. network inference values

### Force Brake
- All constraint tests pass (1000/1000 iterations)
- Gradient computation stable throughout
- Suitable for real-time robotic control

## Contributing

Contributions are welcome! Please:
1. Test both models before submitting changes
2. Update `explaination.md` with any model modifications
3. Ensure no constraint violations in force brake model
4. Run both `python model.py` tests successfully

## License

This project is licensed under the MIT License.

## Author

- **whe128** - Initial development

## Acknowledgments

- Multi-modal sensor fusion for robotic perception
- Physics-informed neural network constraints for safe robot control

---

*Last updated: June 2026*
