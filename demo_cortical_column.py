"""
Neuromorphic Cortical Column Demonstration

This script demonstrates the capabilities of the neuromorphic cortical column
implementation with various input patterns and analysis tools.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from cortical_column import CorticalColumn
import time


def demonstrate_basic_response():
    """Demonstrate basic column response to simple inputs."""
    print("=== Basic Response Demonstration ===")
    
    column = CorticalColumn(size=32)
    
    # Simulation parameters
    duration = 0.5  # seconds
    steps = int(duration / column.dt)
    
    # Storage arrays
    time_points = np.linspace(0, duration, steps)
    inputs = np.zeros(steps)
    outputs = np.zeros(steps)
    layer_activities = {name: np.zeros(steps) for name in column.layers.keys()}
    
    # Run simulation
    print(f"Running simulation for {duration} seconds...")
    start_time = time.time()
    
    for i in range(steps):
        t = i * column.dt
        
        # Step input at t=0.1s
        if t > 0.1:
            sensory_input = np.ones(32) * 0.5
        else:
            sensory_input = np.zeros(32)
        
        inputs[i] = np.mean(sensory_input)
        
        # Step simulation
        column.step(sensory_input)
        
        # Record data
        outputs[i] = np.mean(column.get_output())
        states = column.get_layer_states()
        for name, activity in states.items():
            layer_activities[name][i] = np.mean(activity)
    
    sim_time = time.time() - start_time
    print(f"Simulation completed in {sim_time:.2f} seconds")
    print(f"Real-time factor: {duration/sim_time:.1f}x")
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Input signal
    plt.subplot(3, 3, 1)
    plt.plot(time_points, inputs, 'k-', linewidth=2)
    plt.title('Input Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    # Layer activities
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, (name, activity) in enumerate(layer_activities.items()):
        plt.subplot(3, 3, i+2)
        plt.plot(time_points, activity, color=colors[i], linewidth=2)
        plt.title(f'{name} Activity')
        plt.xlabel('Time (s)')
        plt.ylabel('Activity')
        plt.grid(True)
    
    # Output signal
    plt.subplot(3, 3, 7)
    plt.plot(time_points, outputs, 'r-', linewidth=2)
    plt.title('Column Output')
    plt.xlabel('Time (s)')
    plt.ylabel('Output')
    plt.grid(True)
    
    # Frequency analysis
    plt.subplot(3, 3, 8)
    freqs, psd = signal.welch(outputs, fs=1/column.dt, nperseg=256)
    plt.semilogy(freqs, psd)
    plt.title('Output Power Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.grid(True)
    
    # Phase space plot
    plt.subplot(3, 3, 9)
    plt.plot(layer_activities['L2/3'], layer_activities['L5'], 'b-', alpha=0.7)
    plt.xlabel('L2/3 Activity')
    plt.ylabel('L5 Activity')
    plt.title('L2/3 vs L5 Phase Space')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('basic_response_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Basic response demonstration completed.")


def demonstrate_frequency_response():
    """Demonstrate column response to different frequencies."""
    print("\n=== Frequency Response Demonstration ===")
    
    column = CorticalColumn(size=32)
    
    # Test frequencies
    test_freqs = [1, 5, 10, 20, 50, 100]
    responses = {}
    
    for freq in test_freqs:
        print(f"Testing frequency: {freq} Hz")
        
        # Reset column
        column = CorticalColumn(size=32)
        
        # Run simulation
        duration = 0.5
        steps = int(duration / column.dt)
        outputs = []
        
        for i in range(steps):
            t = i * column.dt
            
            # Sinusoidal input
            sensory_input = np.sin(2 * np.pi * freq * t) * np.ones(32) * 0.5
            
            column.step(sensory_input)
            
            # Record output after settling
            if i > steps // 2:
                outputs.append(np.mean(column.get_output()))
        
        # Calculate response metrics
        output_std = np.std(outputs)
        output_mean = np.mean(outputs)
        
        responses[freq] = {
            'std': output_std,
            'mean': output_mean,
            'snr': output_std / (np.mean(np.abs(outputs)) + 1e-10)
        }
    
    # Plot frequency response
    plt.figure(figsize=(12, 8))
    
    freqs = list(responses.keys())
    stds = [responses[f]['std'] for f in freqs]
    means = [responses[f]['mean'] for f in freqs]
    snrs = [responses[f]['snr'] for f in freqs]
    
    plt.subplot(2, 2, 1)
    plt.semilogx(freqs, stds, 'bo-', linewidth=2, markersize=8)
    plt.title('Response Variability vs Frequency')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Output Std Dev')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.semilogx(freqs, means, 'ro-', linewidth=2, markersize=8)
    plt.title('Mean Response vs Frequency')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Mean Output')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.semilogx(freqs, snrs, 'go-', linewidth=2, markersize=8)
    plt.title('Signal-to-Noise Ratio vs Frequency')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('SNR')
    plt.grid(True)
    
    # Bar chart comparison
    plt.subplot(2, 2, 4)
    plt.bar(range(len(freqs)), stds, alpha=0.7)
    plt.xticks(range(len(freqs)), freqs)
    plt.title('Response Variability Comparison')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Output Std Dev')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('frequency_response_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Frequency response demonstration completed.")


def demonstrate_learning():
    """Demonstrate learning behavior in L2/3."""
    print("\n=== Learning Demonstration ===")
    
    column = CorticalColumn(size=16)
    
    # Define training patterns
    patterns = {
        'pattern_A': np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]),
        'pattern_B': np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
        'pattern_C': np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0])
    }
    
    # Training parameters
    training_epochs = 100
    pattern_duration = 0.1  # seconds per pattern
    steps_per_pattern = int(pattern_duration / column.dt)
    
    # Storage for learning metrics
    weight_changes = []
    response_strengths = {'pattern_A': [], 'pattern_B': [], 'pattern_C': []}
    
    print(f"Training for {training_epochs} epochs...")
    
    # Initial weights
    initial_weights = column.layers['L2/3'].hebbian_weights.copy()
    
    for epoch in range(training_epochs):
        epoch_responses = {'pattern_A': [], 'pattern_B': [], 'pattern_C': []}
        
        # Present each pattern
        for pattern_name, pattern in patterns.items():
            for _ in range(steps_per_pattern):
                column.step(pattern)
                
                # Record response
                l23_activity = np.mean(column.layers['L2/3'].output)
                epoch_responses[pattern_name].append(l23_activity)
        
        # Calculate mean responses
        for pattern_name in patterns.keys():
            mean_response = np.mean(epoch_responses[pattern_name])
            response_strengths[pattern_name].append(mean_response)
        
        # Calculate weight change
        current_weights = column.layers['L2/3'].hebbian_weights
        weight_change = np.mean(np.abs(current_weights - initial_weights))
        weight_changes.append(weight_change)
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Weight change = {weight_change:.4f}")
    
    # Plot learning results
    plt.figure(figsize=(15, 10))
    
    # Weight changes over time
    plt.subplot(2, 3, 1)
    plt.plot(weight_changes, 'b-', linewidth=2)
    plt.title('Hebbian Weight Changes')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Weight Change')
    plt.grid(True)
    
    # Response strengths over time
    plt.subplot(2, 3, 2)
    colors = ['red', 'blue', 'green']
    for i, (pattern_name, responses) in enumerate(response_strengths.items()):
        plt.plot(responses, color=colors[i], linewidth=2, label=pattern_name)
    plt.title('Response Strengths During Learning')
    plt.xlabel('Epoch')
    plt.ylabel('Mean L2/3 Activity')
    plt.legend()
    plt.grid(True)
    
    # Final weight matrix
    plt.subplot(2, 3, 3)
    final_weights = column.layers['L2/3'].hebbian_weights
    plt.imshow(final_weights, cmap='RdBu', aspect='auto')
    plt.colorbar()
    plt.title('Final Hebbian Weights')
    plt.xlabel('Input Neuron')
    plt.ylabel('Output Neuron')
    
    # Weight distribution
    plt.subplot(2, 3, 4)
    plt.hist(final_weights.flatten(), bins=30, alpha=0.7, edgecolor='black')
    plt.title('Weight Distribution')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    # Pattern discrimination
    plt.subplot(2, 3, 5)
    final_responses = [response_strengths[p][-1] for p in patterns.keys()]
    plt.bar(range(len(patterns)), final_responses, alpha=0.7)
    plt.xticks(range(len(patterns)), patterns.keys())
    plt.title('Final Pattern Responses')
    plt.xlabel('Pattern')
    plt.ylabel('Mean Response')
    plt.grid(True)
    
    # Learning curve
    plt.subplot(2, 3, 6)
    # Calculate learning efficiency
    learning_efficiency = np.gradient(weight_changes)
    plt.plot(learning_efficiency, 'g-', linewidth=2)
    plt.title('Learning Rate Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Efficiency')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('learning_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Learning demonstration completed.")


def demonstrate_field_coupling():
    """Demonstrate field coupling effects."""
    print("\n=== Field Coupling Demonstration ===")
    
    # Create two columns to show coupling
    column1 = CorticalColumn(size=16)
    column2 = CorticalColumn(size=16)
    
    # Simulation parameters
    duration = 0.3
    steps = int(duration / column1.dt)
    
    # Storage arrays
    time_points = np.linspace(0, duration, steps)
    col1_outputs = np.zeros(steps)
    col2_outputs = np.zeros(steps)
    field_strengths = np.zeros(steps)
    
    print(f"Running coupled simulation for {duration} seconds...")
    
    for i in range(steps):
        t = i * column1.dt
        
        # Different inputs to each column
        input1 = np.sin(2 * np.pi * 10 * t) * np.ones(16) * 0.5
        input2 = np.sin(2 * np.pi * 15 * t) * np.ones(16) * 0.3
        
        # Simulate field coupling between columns
        # Use L2/3 activity of each column to influence the other
        field_coupling_1 = np.mean(column2.layers['L2/3'].output) * 0.1
        field_coupling_2 = np.mean(column1.layers['L2/3'].output) * 0.1
        
        # Add field coupling to L1 layers
        field_input_1 = np.ones(16) * field_coupling_1
        field_input_2 = np.ones(16) * field_coupling_2
        
        # Update columns
        column1.step(input1)
        column2.step(input2)
        
        # Apply cross-coupling
        column1.layers['L1'].field_coupling += field_input_1
        column2.layers['L1'].field_coupling += field_input_2
        
        # Record data
        col1_outputs[i] = np.mean(column1.get_output())
        col2_outputs[i] = np.mean(column2.get_output())
        field_strengths[i] = abs(field_coupling_1 + field_coupling_2)
    
    # Plot field coupling results
    plt.figure(figsize=(12, 8))
    
    # Column outputs
    plt.subplot(2, 2, 1)
    plt.plot(time_points, col1_outputs, 'b-', linewidth=2, label='Column 1')
    plt.plot(time_points, col2_outputs, 'r-', linewidth=2, label='Column 2')
    plt.title('Column Outputs')
    plt.xlabel('Time (s)')
    plt.ylabel('Output')
    plt.legend()
    plt.grid(True)
    
    # Field coupling strength
    plt.subplot(2, 2, 2)
    plt.plot(time_points, field_strengths, 'g-', linewidth=2)
    plt.title('Field Coupling Strength')
    plt.xlabel('Time (s)')
    plt.ylabel('Coupling Strength')
    plt.grid(True)
    
    # Cross-correlation
    plt.subplot(2, 2, 3)
    correlation = np.correlate(col1_outputs, col2_outputs, mode='full')
    lags = np.arange(-len(col1_outputs)+1, len(col1_outputs))
    plt.plot(lags * column1.dt, correlation, 'purple', linewidth=2)
    plt.title('Cross-Correlation')
    plt.xlabel('Lag (s)')
    plt.ylabel('Correlation')
    plt.grid(True)
    
    # Phase space
    plt.subplot(2, 2, 4)
    plt.plot(col1_outputs, col2_outputs, 'orange', alpha=0.7)
    plt.xlabel('Column 1 Output')
    plt.ylabel('Column 2 Output')
    plt.title('Phase Space (Column 1 vs Column 2)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('field_coupling_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Field coupling demonstration completed.")


def demonstrate_noise_robustness():
    """Demonstrate system robustness to noise."""
    print("\n=== Noise Robustness Demonstration ===")
    
    # Test different noise levels
    noise_levels = [0.0, 0.1, 0.3, 0.5, 1.0]
    
    results = {}
    
    for noise_level in noise_levels:
        print(f"Testing noise level: {noise_level}")
        
        column = CorticalColumn(size=16)
        
        # Simulation parameters
        duration = 0.3
        steps = int(duration / column.dt)
        
        outputs = []
        
        for i in range(steps):
            t = i * column.dt
            
            # Clean signal
            clean_input = np.sin(2 * np.pi * 10 * t) * np.ones(16) * 0.5
            
            # Add noise
            noise = np.random.normal(0, noise_level, 16)
            noisy_input = clean_input + noise
            
            column.step(noisy_input)
            outputs.append(np.mean(column.get_output()))
        
        # Calculate metrics
        output_std = np.std(outputs)
        output_mean = np.mean(outputs)
        snr = output_std / (output_mean + 1e-10)
        
        results[noise_level] = {
            'outputs': outputs,
            'std': output_std,
            'mean': output_mean,
            'snr': snr
        }
    
    # Plot noise robustness results
    plt.figure(figsize=(12, 8))
    
    # Output traces for different noise levels
    plt.subplot(2, 2, 1)
    time_points = np.linspace(0, duration, steps)
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    
    for i, noise_level in enumerate(noise_levels):
        plt.plot(time_points, results[noise_level]['outputs'], 
                color=colors[i], alpha=0.7, linewidth=1, 
                label=f'Noise={noise_level}')
    
    plt.title('Output Under Different Noise Levels')
    plt.xlabel('Time (s)')
    plt.ylabel('Output')
    plt.legend()
    plt.grid(True)
    
    # Noise vs performance metrics
    plt.subplot(2, 2, 2)
    stds = [results[n]['std'] for n in noise_levels]
    plt.plot(noise_levels, stds, 'bo-', linewidth=2, markersize=8)
    plt.title('Output Variability vs Noise Level')
    plt.xlabel('Noise Level')
    plt.ylabel('Output Std Dev')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    means = [results[n]['mean'] for n in noise_levels]
    plt.plot(noise_levels, means, 'ro-', linewidth=2, markersize=8)
    plt.title('Mean Output vs Noise Level')
    plt.xlabel('Noise Level')
    plt.ylabel('Mean Output')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    snrs = [results[n]['snr'] for n in noise_levels]
    plt.plot(noise_levels, snrs, 'go-', linewidth=2, markersize=8)
    plt.title('Signal-to-Noise Ratio vs Noise Level')
    plt.xlabel('Noise Level')
    plt.ylabel('SNR')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('noise_robustness_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Noise robustness demonstration completed.")


def main():
    """Run all demonstrations."""
    print("Neuromorphic Cortical Column Demonstration Suite")
    print("=" * 50)
    
    # Set matplotlib backend for headless operation if needed
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    
    try:
        # Run all demonstrations
        demonstrate_basic_response()
        demonstrate_frequency_response()
        demonstrate_learning()
        demonstrate_field_coupling()
        demonstrate_noise_robustness()
        
        print("\n" + "=" * 50)
        print("All demonstrations completed successfully!")
        print("Check the generated PNG files for visualizations.")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()