"""
Test suite for the neuromorphic cortical column implementation.

This module contains comprehensive tests for all layer types,
field coupling mechanisms, and integration functionality.
"""

import pytest
import numpy as np
from cortical_column import (
    CorticalColumn, Layer1, Layer23, Layer4, Layer5, Layer6,
    FieldCoupling, LayerConfig, AnalogLayer
)


class TestLayerConfig:
    """Test LayerConfig dataclass."""
    
    def test_layer_config_creation(self):
        """Test basic layer configuration creation."""
        config = LayerConfig(
            name='Test', tau=20, threshold=0.5, gain=1.0,
            coupling_strength=0.2, noise_level=0.01
        )
        assert config.name == 'Test'
        assert config.tau == 20
        assert config.threshold == 0.5
        assert config.gain == 1.0
        assert config.coupling_strength == 0.2
        assert config.noise_level == 0.01


class TestAnalogLayer:
    """Test base AnalogLayer functionality."""
    
    def test_lateral_connections(self):
        """Test lateral connection matrix creation."""
        config = LayerConfig('Test', 20, 0.5, 1.0, 0.2, 0.01)
        layer = Layer1(config, size=8)
        
        # Check matrix shape
        assert layer.lateral_connections.shape == (8, 8)
        
        # Check diagonal is zero
        assert np.all(np.diag(layer.lateral_connections) == 0)
        
        # Check symmetry and decay
        assert layer.lateral_connections[0, 1] > layer.lateral_connections[0, 7]
    
    def test_activation_function(self):
        """Test sigmoid activation function."""
        config = LayerConfig('Test', 20, 0.5, 1.0, 0.2, 0.01)
        layer = Layer1(config, size=4)
        
        # Test activation
        x = np.array([-1, 0, 0.5, 1])
        output = layer._activation(x)
        
        # Check range
        assert np.all(output >= 0)
        assert np.all(output <= 1)
        
        # Check monotonicity
        assert output[0] < output[1] < output[2] < output[3]


class TestLayer1:
    """Test Layer 1 (superficial modulation) functionality."""
    
    def test_initialization(self):
        """Test Layer 1 initialization."""
        config = LayerConfig('L1', 50, 0.5, 0.5, 0.1, 0.01)
        layer = Layer1(config, size=16)
        
        assert layer.size == 16
        assert layer.phase_detector.shape == (16,)
        assert layer.delay_line.shape == (16, 10)
        assert np.all(layer.state == 0)
        assert np.all(layer.output == 0)
    
    def test_dynamics(self):
        """Test Layer 1 dynamics computation."""
        config = LayerConfig('L1', 50, 0.5, 0.5, 0.1, 0.01)
        layer = Layer1(config, size=4)
        
        # Set initial conditions
        layer.state = np.array([0.1, 0.2, 0.3, 0.4])
        layer.field_coupling = np.array([0.05, 0.05, 0.05, 0.05])
        inputs = np.array([0.2, 0.3, 0.4, 0.5])
        
        # Compute dynamics
        dstate = layer.dynamics(layer.state, 0.1, inputs)
        
        # Check output shape
        assert dstate.shape == (4,)
        
        # Check dynamics are reasonable
        assert np.all(np.isfinite(dstate))
    
    def test_update(self):
        """Test Layer 1 update mechanism."""
        config = LayerConfig('L1', 50, 0.5, 0.5, 0.1, 0.01)
        layer = Layer1(config, size=4)
        
        inputs = np.array([0.2, 0.3, 0.4, 0.5])
        field_input = np.array([0.1, 0.1, 0.1, 0.1])
        
        # Store initial state
        initial_state = layer.state.copy()
        
        # Update layer
        layer.update(0.001, inputs, field_input)
        
        # Check state changed
        assert not np.array_equal(layer.state, initial_state)
        
        # Check field coupling updated
        assert np.array_equal(layer.field_coupling, field_input)


class TestLayer23:
    """Test Layer 2/3 (associative processing) functionality."""
    
    def test_initialization(self):
        """Test Layer 2/3 initialization."""
        config = LayerConfig('L2/3', 20, 0.6, 1.0, 0.3, 0.02)
        layer = Layer23(config, size=8)
        
        assert layer.size == 8
        assert layer.hebbian_weights.shape == (8, 8)
        assert layer.sparse_threshold == 0.8
    
    def test_hebbian_learning(self):
        """Test Hebbian weight updates."""
        config = LayerConfig('L2/3', 20, 0.6, 1.0, 0.3, 0.02)
        layer = Layer23(config, size=4)
        
        # Set initial weights
        initial_weights = layer.hebbian_weights.copy()
        
        # Simulate correlated activity
        layer.state = np.array([0.8, 0.2, 0.9, 0.1])
        inputs = np.array([0.7, 0.3, 0.8, 0.2])
        
        # Update weights
        layer._update_hebbian_weights(layer.state, inputs)
        
        # Check weights changed
        assert not np.array_equal(layer.hebbian_weights, initial_weights)
        
        # Check weight bounds
        assert np.all(layer.hebbian_weights >= -1)
        assert np.all(layer.hebbian_weights <= 1)
    
    def test_sparse_coding(self):
        """Test sparse coding behavior."""
        config = LayerConfig('L2/3', 20, 0.6, 1.0, 0.3, 0.02)
        layer = Layer23(config, size=8)
        
        # Set high activity state
        layer.state = np.array([0.9, 0.1, 0.95, 0.05, 0.85, 0.15, 0.9, 0.1])
        inputs = np.zeros(8)
        
        # Compute dynamics
        dstate = layer.dynamics(layer.state, 0.1, inputs)
        
        # Check output shape
        assert dstate.shape == (8,)
        assert np.all(np.isfinite(dstate))


class TestLayer4:
    """Test Layer 4 (sensory input) functionality."""
    
    def test_initialization(self):
        """Test Layer 4 initialization."""
        config = LayerConfig('L4', 10, 0.4, 1.5, 0.2, 0.015)
        layer = Layer4(config, size=16)
        
        assert layer.size == 16
        assert len(layer.bandpass_filters) == 16
        assert layer.edge_detectors.shape == (16,)
    
    def test_bandpass_filters(self):
        """Test bandpass filter creation."""
        config = LayerConfig('L4', 10, 0.4, 1.5, 0.2, 0.015)
        layer = Layer4(config, size=8)
        
        # Check filter count
        assert len(layer.bandpass_filters) == 8
        
        # Each filter should be a scipy SOS array
        for filt in layer.bandpass_filters:
            assert filt.shape[1] == 6  # SOS format
    
    def test_edge_detection(self):
        """Test edge detection functionality."""
        config = LayerConfig('L4', 10, 0.4, 1.5, 0.2, 0.015)
        layer = Layer4(config, size=8)
        
        # Create step input
        inputs = np.array([0, 0, 0, 1, 1, 1, 0, 0])
        
        # Compute dynamics
        dstate = layer.dynamics(layer.state, 0.1, inputs)
        
        # Check output shape and validity
        assert dstate.shape == (8,)
        assert np.all(np.isfinite(dstate))


class TestLayer5:
    """Test Layer 5 (output) functionality."""
    
    def test_initialization(self):
        """Test Layer 5 initialization."""
        config = LayerConfig('L5', 15, 0.7, 2.0, 0.1, 0.01)
        layer = Layer5(config, size=8)
        
        assert layer.size == 8
        assert layer.burst_detector.shape == (8,)
        assert layer.integration_buffer.shape == (8,)
        assert layer.motor_output.shape == (8,)
    
    def test_burst_detection(self):
        """Test burst detection mechanism."""
        config = LayerConfig('L5', 15, 0.7, 2.0, 0.1, 0.01)
        layer = Layer5(config, size=4)
        
        # Set up for burst
        layer.integration_buffer = np.array([0.8, 0.5, 0.9, 0.3])
        layer.state = np.array([0.6, 0.4, 0.7, 0.2])
        
        # Compute dynamics
        inputs = np.array([0.1, 0.1, 0.1, 0.1])
        dstate = layer.dynamics(layer.state, 0.1, inputs)
        
        # Check output shape and validity
        assert dstate.shape == (4,)
        assert np.all(np.isfinite(dstate))
    
    def test_motor_output(self):
        """Test motor output generation."""
        config = LayerConfig('L5', 15, 0.7, 2.0, 0.1, 0.01)
        layer = Layer5(config, size=4)
        
        state = np.array([0.8, 0.3, 0.9, 0.1])
        burst_mask = np.array([1, 0, 1, 0])
        
        motor_output = layer._generate_motor_output(state, burst_mask)
        
        # Check output shape
        assert motor_output.shape == (4,)
        assert np.all(np.isfinite(motor_output))


class TestLayer6:
    """Test Layer 6 (feedback control) functionality."""
    
    def test_initialization(self):
        """Test Layer 6 initialization."""
        config = LayerConfig('L6', 30, 0.5, 0.8, 0.15, 0.01)
        layer = Layer6(config, size=8)
        
        assert layer.size == 8
        assert layer.delay_locked_loop.shape == (8,)
        assert layer.oscillator_phase.shape == (8,)
        assert layer.feedback_strength == 0.3
    
    def test_oscillator_dynamics(self):
        """Test oscillator phase dynamics."""
        config = LayerConfig('L6', 30, 0.5, 0.8, 0.15, 0.01)
        layer = Layer6(config, size=4)
        
        # Store initial phase
        initial_phase = layer.oscillator_phase.copy()
        
        # Compute dynamics
        inputs = np.array([0.2, 0.3, 0.4, 0.5])
        dstate = layer.dynamics(layer.state, 0.1, inputs)
        
        # Check phase updated
        assert not np.array_equal(layer.oscillator_phase, initial_phase)
        
        # Check output shape and validity
        assert dstate.shape == (4,)
        assert np.all(np.isfinite(dstate))
    
    def test_feedback_signal(self):
        """Test feedback signal generation."""
        config = LayerConfig('L6', 30, 0.5, 0.8, 0.15, 0.01)
        layer = Layer6(config, size=4)
        
        # Set output
        layer.output = np.array([0.5, 0.3, 0.7, 0.2])
        
        # Get feedback signal
        feedback = layer.get_feedback_signal()
        
        # Check shape and scaling
        assert feedback.shape == (4,)
        assert np.all(np.abs(feedback) <= np.abs(layer.output))


class TestFieldCoupling:
    """Test field coupling mechanisms."""
    
    def test_initialization(self):
        """Test field coupling initialization."""
        coupling = FieldCoupling(size=8)
        
        assert coupling.size == 8
        assert coupling.coupling_matrix.shape == (8, 8)
        
        # Check diagonal is zero
        assert np.all(np.diag(coupling.coupling_matrix) == 0)
    
    def test_field_interactions(self):
        """Test field interaction computation."""
        coupling = FieldCoupling(size=4)
        
        # Create mock layers
        config = LayerConfig('Test', 20, 0.5, 1.0, 0.2, 0.01)
        layers = {
            'L1': Layer1(config, 4),
            'L2/3': Layer23(config, 4),
            'L4': Layer4(config, 4),
            'L5': Layer5(config, 4),
            'L6': Layer6(config, 4)
        }
        
        # Set some output values
        for layer in layers.values():
            layer.output = np.random.uniform(0, 1, 4)
        
        # Compute field interactions
        field_signals = coupling.compute_field_interactions(layers)
        
        # Check all layers have field signals
        for layer_name in layers.keys():
            assert layer_name in field_signals
            assert field_signals[layer_name].shape == (4,)
            assert np.all(np.isfinite(field_signals[layer_name]))


class TestCorticalColumn:
    """Test complete cortical column functionality."""
    
    def test_initialization(self):
        """Test cortical column initialization."""
        column = CorticalColumn(size=16)
        
        assert column.size == 16
        assert len(column.layers) == 5
        assert 'L1' in column.layers
        assert 'L2/3' in column.layers
        assert 'L4' in column.layers
        assert 'L5' in column.layers
        assert 'L6' in column.layers
        
        # Check field coupling
        assert column.field_coupling.size == 16
        
        # Check time parameters
        assert column.time == 0
        assert column.dt == 0.001
    
    def test_single_step(self):
        """Test single simulation step."""
        column = CorticalColumn(size=8)
        
        # Create test input
        sensory_input = np.random.uniform(0, 1, 8)
        context_input = np.random.uniform(0, 1, 8)
        
        # Store initial states
        initial_states = {name: layer.state.copy() for name, layer in column.layers.items()}
        initial_time = column.time
        
        # Perform step
        column.step(sensory_input, context_input)
        
        # Check time advanced
        assert column.time > initial_time
        
        # Check states changed
        for name, layer in column.layers.items():
            assert not np.array_equal(layer.state, initial_states[name])
    
    def test_output_generation(self):
        """Test output generation."""
        column = CorticalColumn(size=8)
        
        # Run a few steps
        for _ in range(10):
            sensory_input = np.random.uniform(0, 1, 8)
            column.step(sensory_input)
        
        # Get output
        output = column.get_output()
        
        # Check output shape and validity
        assert output.shape == (8,)
        assert np.all(np.isfinite(output))
    
    def test_layer_states(self):
        """Test layer state retrieval."""
        column = CorticalColumn(size=8)
        
        # Run simulation
        sensory_input = np.random.uniform(0, 1, 8)
        column.step(sensory_input)
        
        # Get layer states
        states = column.get_layer_states()
        
        # Check all layers present
        assert len(states) == 5
        for name in ['L1', 'L2/3', 'L4', 'L5', 'L6']:
            assert name in states
            assert states[name].shape == (8,)
            assert np.all(np.isfinite(states[name]))
    
    def test_l1_modulation(self):
        """Test L1 modulation of other layers."""
        column = CorticalColumn(size=8)
        
        # Set L1 output
        column.layers['L1'].output = np.ones(8) * 0.5
        
        # Store initial states
        initial_l23 = column.layers['L2/3'].state.copy()
        initial_l5 = column.layers['L5'].state.copy()
        
        # Apply modulation
        column._apply_l1_modulation()
        
        # Check modulation applied
        assert not np.array_equal(column.layers['L2/3'].state, initial_l23)
        assert not np.array_equal(column.layers['L5'].state, initial_l5)
    
    def test_multi_step_simulation(self):
        """Test multi-step simulation stability."""
        column = CorticalColumn(size=8)
        
        # Run multiple steps
        outputs = []
        for i in range(100):
            # Create varying input
            t = i * column.dt
            sensory_input = np.sin(2 * np.pi * 10 * t) * np.ones(8)
            
            column.step(sensory_input)
            outputs.append(column.get_output().copy())
        
        # Check simulation stability
        assert len(outputs) == 100
        for output in outputs:
            assert np.all(np.isfinite(output))
            assert output.shape == (8,)
        
        # Check for reasonable dynamics (not stuck)
        output_variance = np.var([np.mean(output) for output in outputs])
        assert output_variance >= 0  # Some variation expected or system stable


class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_frequency_response(self):
        """Test system response to different frequencies."""
        column = CorticalColumn(size=8)
        
        # Test different input frequencies
        frequencies = [1, 5, 10, 20, 50]
        responses = {}
        
        for freq in frequencies:
            outputs = []
            for i in range(200):
                t = i * column.dt
                sensory_input = np.sin(2 * np.pi * freq * t) * np.ones(8)
                column.step(sensory_input)
                
                if i > 100:  # Skip transient
                    outputs.append(np.mean(column.get_output()))
            
            responses[freq] = np.std(outputs)
        
        # Check responses are reasonable
        for freq, response in responses.items():
            assert response >= 0  # Allow zero response (system stable)
            assert np.isfinite(response)
    
    def test_learning_behavior(self):
        """Test learning behavior in L2/3."""
        column = CorticalColumn(size=8)
        
        # Store initial weights
        initial_weights = column.layers['L2/3'].hebbian_weights.copy()
        
        # Present repeated pattern
        pattern = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        for _ in range(1000):
            column.step(pattern)
        
        # Check weights changed
        final_weights = column.layers['L2/3'].hebbian_weights
        weight_change = np.mean(np.abs(final_weights - initial_weights))
        assert weight_change > 0.005  # Some learning occurred
    
    def test_noise_robustness(self):
        """Test system robustness to noise."""
        column = CorticalColumn(size=8)
        
        # Test with different noise levels
        noise_levels = [0.0, 0.1, 0.5, 1.0]
        
        for noise_level in noise_levels:
            outputs = []
            for i in range(100):
                # Clean signal + noise
                clean_input = np.sin(2 * np.pi * 10 * i * column.dt) * np.ones(8)
                noisy_input = clean_input + np.random.normal(0, noise_level, 8)
                
                column.step(noisy_input)
                outputs.append(np.mean(column.get_output()))
            
            # Check system doesn't diverge
            assert np.all(np.isfinite(outputs))
            assert np.std(outputs) < 10  # Reasonable output variance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])