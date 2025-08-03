"""
Configuration system for neuromorphic cortical column.

Centralizes all configuration parameters to eliminate hardcoded values.
"""

from dataclasses import dataclass
from typing import Dict, Any
import numpy as np


@dataclass
class SimulationConfig:
    """Global simulation configuration."""
    dt: float = 0.001  # Time step in seconds
    noise_seed: int = 42  # Random seed for reproducibility
    

@dataclass  
class LayerParameters:
    """Physical and biological parameters for each layer."""
    # Timing constants
    tau: float  # Time constant (ms)
    threshold: float  # Activation threshold
    gain: float  # Amplification factor
    coupling_strength: float  # Lateral coupling strength
    noise_level: float  # Background noise level
    
    # Layer-specific parameters
    frequency_range: tuple = (1.0, 100.0)  # Hz range for frequency tuning
    learning_rate: float = 0.001  # Hebbian learning rate
    sparse_threshold: float = 0.8  # Sparse coding threshold
    burst_threshold: float = 0.1  # Burst detection threshold
    

@dataclass
class FieldCouplingConfig:
    """Configuration for electromagnetic field coupling."""
    coupling_decay_constant: float = 0.1  # Distance decay for lateral connections
    field_strength_l1: float = 0.1  # L1 field coupling strength
    field_strength_l23: float = 0.2  # L2/3 field coupling strength
    field_strength_l5: float = 0.15  # L5 field coupling strength
    field_strength_l6: float = 0.1  # L6 field coupling strength
    

@dataclass
class NetworkConfig:
    """Multi-column network configuration."""
    default_column_size: int = 64
    inter_column_coupling: float = 0.05
    attention_strength: float = 0.3
    

class CorticalConfig:
    """Main configuration class with validated parameters."""
    
    def __init__(self):
        self.simulation = SimulationConfig()
        self.field_coupling = FieldCouplingConfig()
        self.network = NetworkConfig()
        
        # Layer configurations based on biological data and engineering constraints
        self.layers = {
            'L1': LayerParameters(
                tau=50, threshold=0.3, gain=1.0, coupling_strength=0.2, 
                noise_level=0.01, learning_rate=0.0005
            ),
            'L2/3': LayerParameters(
                tau=20, threshold=0.4, gain=2.0, coupling_strength=0.4,
                noise_level=0.02, learning_rate=0.001, sparse_threshold=0.8
            ),
            'L4': LayerParameters(
                tau=10, threshold=0.2, gain=3.0, coupling_strength=0.3,
                noise_level=0.015, frequency_range=(1.0, 100.0)
            ),
            'L5': LayerParameters(
                tau=15, threshold=0.3, gain=4.0, coupling_strength=0.2,
                noise_level=0.01, burst_threshold=0.1
            ),
            'L6': LayerParameters(
                tau=30, threshold=0.3, gain=1.5, coupling_strength=0.25,
                noise_level=0.01, learning_rate=0.0008
            )
        }
        
        # Oscillation frequencies for different layers (Hz)
        self.oscillations = {
            'L1_phase_freq': 0.1,  # Slow phase modulation
            'L23_resonance_freq': 0.05,  # Field resonance
            'L6_oscillator_freq': 0.1,  # Base oscillator
            'L6_resonance_freq': 0.02,  # Resonance coupling
            'L5_carrier_freq': 50.0  # PWM carrier frequency
        }
        
        # Integration and coupling parameters
        self.integration = {
            'l1_delay_steps': 10,  # Delay line length
            'l5_integration_rate': 0.5,  # Integration buffer rate
            'l6_feedback_strength': 0.3,  # Feedback modulation
            'edge_detection_gain': 0.5,  # Edge detection scaling
            'activation_sharpness': 0.1  # Sigmoid activation steepness
        }
        
    def validate(self) -> bool:
        """Validate configuration parameters."""
        # Check that all required parameters are positive
        for layer_name, params in self.layers.items():
            if params.tau <= 0 or params.gain <= 0:
                raise ValueError(f"Layer {layer_name}: tau and gain must be positive")
            if not 0 <= params.threshold <= 1:
                raise ValueError(f"Layer {layer_name}: threshold must be between 0 and 1")
                
        # Check simulation parameters
        if self.simulation.dt <= 0 or self.simulation.dt > 0.01:
            raise ValueError("Time step must be between 0 and 0.01 seconds")
            
        return True
        
    def update_layer_config(self, layer_name: str, **kwargs):
        """Update configuration for a specific layer."""
        if layer_name not in self.layers:
            raise ValueError(f"Unknown layer: {layer_name}")
            
        layer_config = self.layers[layer_name]
        for key, value in kwargs.items():
            if hasattr(layer_config, key):
                setattr(layer_config, key, value)
            else:
                raise ValueError(f"Unknown parameter {key} for layer {layer_name}")
                
        self.validate()
        
    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary for serialization."""
        return {
            'simulation': self.simulation.__dict__,
            'field_coupling': self.field_coupling.__dict__,
            'network': self.network.__dict__,
            'layers': {name: params.__dict__ for name, params in self.layers.items()},
            'oscillations': self.oscillations,
            'integration': self.integration
        }


# Global configuration instance
DEFAULT_CONFIG = CorticalConfig()