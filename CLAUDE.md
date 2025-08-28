# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a neuromorphic cortical column implementation that simulates the six-layer structure of neocortical columns using analog circuit principles. The project translates biological neural circuits into computational models suitable for neuromorphic computing applications.

## Core Architecture

### Main Implementation
- `cortical_column.py` - Main implementation with all layer classes and simulation logic
- Six cortical layers (L1, L2/3, L4, L5, L6) each with specialized functionality:
  - **Layer 1**: Context modulation with phase sensitivity and delay lines
  - **Layer 2/3**: Associative processing with Hebbian learning and sparse coding
  - **Layer 4**: Sensory input processing with bandpass filtering and edge detection
  - **Layer 5**: Motor output generation with burst detection mechanisms
  - **Layer 6**: Feedback control with delay-locked loops and timing regulation

### Key Classes
- `CorticalColumn` - Main orchestrator class that manages all layers
- `AnalogLayer` - Abstract base class for all cortical layers
- `FieldCoupling` - Manages electromagnetic field interactions between layers
- `LayerConfig` - Configuration dataclass for layer parameters

## Development Commands

### Testing
```bash
# Run all tests
python -m pytest test_cortical_column.py -v

# Run specific test class
python -m pytest test_cortical_column.py::TestLayer23 -v

# Run specific test method
python -m pytest test_cortical_column.py::TestLayer23::test_hebbian_learning -v
```

### Running Simulations
```bash
# Main simulation with visualization
python cortical_column.py

# Comprehensive demonstration suite (5 demos with analysis)
python demo_cortical_column.py

# Simple step response example
python simple_example.py
```

### Demo Scenarios Available
- **Basic Response**: Step input analysis with frequency spectrum
- **Frequency Response**: Multi-frequency testing (1-100 Hz)
- **Learning Demo**: Hebbian plasticity with pattern discrimination
- **Field Coupling**: Inter-column electromagnetic interactions
- **Noise Robustness**: Performance under various noise conditions

### Dependencies (2025)
```bash
# Install dependencies (Python 3.13.5 compatible)
pip install -r requirements.txt

# Core packages (2025 latest):
# numpy>=2.3.2, scipy>=1.16.1, matplotlib>=3.10.5, pytest>=8.4.1

# Optional neuromorphic hardware SDKs:
# Intel Loihi 3: pip install intel-nxsdk
# BrainChip Akida 2: pip install akida
# SynSense tools: pip install synsense-sdk
```

## Code Structure and Patterns

### Layer Implementation Pattern
- All layers inherit from `AnalogLayer` abstract base class
- Each layer implements `dynamics()` method defining differential equations
- Layers have configurable parameters (tau, threshold, gain, coupling_strength, noise_level)
- State updates use Euler integration for real-time performance

### Simulation Flow
1. **Step-based simulation**: `CorticalColumn.step()` processes one time step; `dt` is in seconds and layer `tau` values are in milliseconds (internally converted)
2. **Layer updating order**: L4 → L2/3 → L5 → L6 → L1
3. **Field coupling**: Computed between all layers using `FieldCoupling` class
4. **Modulation**: L1 applies top-down modulation to L2/3 and L5

### Configuration
- Default time step: 1 ms (`dt = 0.001` seconds)
- Default column size: 64 neurons per layer
- Layer-specific time constants range from 10 ms (L4) to 50 ms (L1)
- `integration` block centralizes pathway gains and layer behaviors (e.g., `l23_from_l4_gain`, `l5_integration_rate`, `l1_modulation_gain`)

## Key Implementation Details

### Biologically-Inspired Features
- **Hebbian Learning**: Implemented in Layer 2/3 with spike-timing dependent plasticity; learning rate and decay are configurable
- **Sparse Coding**: Selective activation using thresholds in Layer 2/3 (configurable)
- **Burst Firing & PWM**: Layer 5 generates burst patterns and a time-based PWM motor signal with leaky integration
- **Field Coupling**: Electromagnetic-like interactions between layers using distance-based decay
- **Feedback Control**: Layer 6 provides timing regulation via an oscillator advanced by `dt` and configurable gains

### Performance Characteristics
- Real-time capable simulation (runs faster than biological time)
- Scalable from 8 to 256+ neurons per layer
- Memory usage: ~100 MB for 64-neuron column
- Computation: ~1000 steps/second on modern CPU
- Demonstration suite provides comprehensive validation across 5 test scenarios

### Visualization Outputs
All simulation scripts generate PNG visualizations:
- `cortical_column_simulation.png`: Main simulation results
- `simple_example_output.png`: Basic step response analysis
- `*_demo.png`: Comprehensive analysis plots from demo suite

### File Dependencies
- Main implementation requires: numpy, scipy, matplotlib
- Testing requires: pytest
- Visualization outputs: PNG files for simulation results

## Testing Structure

The test suite (`test_cortical_column.py`) covers:
- Individual layer functionality and dynamics
- Inter-layer connections and field coupling
- Learning behavior (Hebbian plasticity)
- Noise robustness and integration stability
- Complete cortical column integration

All tests should pass (29/29 test cases) for a healthy codebase.

## Research Context

This implementation bridges biological cortical research with analog/field-based neuromorphic engineering. Key research foundations:

### Biological Basis
- **Cortical Column Theory**: Six-layer laminar organization (research-nmf.md)
- **Canonical Microcircuit**: Standardized connectivity patterns across cortical areas
- **Field Coupling**: Electromagnetic interactions between layers and columns

### Implementation Strategy (2025 Current)
- **Analog Translation**: Biological functions → analog circuit components → field interactions
- **FPAA Target**: Field Programmable Analog Array implementation (10,000x power efficiency vs digital)
- **Hybrid Architecture**: Analog processing with digital AI integration
- **Current Platforms**: Intel Loihi 3, BrainChip Akida 2, SynSense Speck for hardware acceleration
- **Python 3.13.5**: Latest stable release with JIT compiler and free-threaded mode support

### Applications and Extensions
- **Immediate**: Sensory processing, edge AI, adaptive robotics, brain-computer interfaces
- **Advanced**: Multi-column arrays, quantum-enhanced processing, consciousness research
- **Hardware**: FPGA prototyping → custom neuromorphic chips

### Related Documentation
- `research-nmf.md`: Core biological-to-analog translation theory
- `neuromorphic-cortical-column-design.md`: Comprehensive hardware implementation guide
- `FUTURE_RESEARCH_ROADMAP.md`: Consciousness research and AI integration roadmap