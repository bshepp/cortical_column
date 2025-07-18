#!/usr/bin/env python3
"""
Simple Example: Neuromorphic Cortical Column

This script demonstrates basic usage of the neuromorphic cortical column
implementation with a simple sensory input pattern.
"""

import numpy as np
import matplotlib.pyplot as plt
from cortical_column import CorticalColumn

def main():
    print("Neuromorphic Cortical Column - Simple Example")
    print("=" * 50)
    
    # Create a cortical column with 32 neurons per layer
    column = CorticalColumn(size=32)
    print(f"Created cortical column with {column.size} neurons per layer")
    
    # Simulation parameters
    duration = 0.2  # 200ms simulation
    steps = int(duration / column.dt)
    print(f"Running simulation for {duration}s ({steps} steps)")
    
    # Storage for results
    time_points = []
    layer_activities = {name: [] for name in column.layers.keys()}
    outputs = []
    
    # Run simulation
    for i in range(steps):
        t = i * column.dt
        
        # Create a simple step input at t=0.1s
        if t > 0.1:
            # Step input: half neurons active
            sensory_input = np.concatenate([
                np.ones(16) * 0.8,   # First half active
                np.zeros(16)         # Second half inactive
            ])
        else:
            # No input initially
            sensory_input = np.zeros(32)
        
        # Step the simulation
        column.step(sensory_input)
        
        # Record data every 10 steps (10ms intervals)
        if i % 10 == 0:
            time_points.append(t)
            
            # Get layer activities
            states = column.get_layer_states()
            for name, activity in states.items():
                layer_activities[name].append(np.mean(activity))
            
            # Get output
            outputs.append(np.mean(column.get_output()))
    
    # Print results
    print(f"\nSimulation completed!")
    print(f"Final layer activities:")
    for name, activity in layer_activities.items():
        final_activity = activity[-1]
        print(f"  {name}: {final_activity:.3f}")
    
    print(f"Final output: {outputs[-1]:.3f}")
    print(f"Output range: [{min(outputs):.3f}, {max(outputs):.3f}]")
    
    # Simple visualization
    plt.figure(figsize=(12, 6))
    
    # Plot layer activities
    plt.subplot(1, 2, 1)
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, (name, activity) in enumerate(layer_activities.items()):
        plt.plot(time_points, activity, 
                color=colors[i], linewidth=2, label=name)
    
    plt.axvline(x=0.1, color='black', linestyle='--', alpha=0.5, label='Input ON')
    plt.title('Layer Activities')
    plt.xlabel('Time (s)')
    plt.ylabel('Mean Activity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot output
    plt.subplot(1, 2, 2)
    plt.plot(time_points, outputs, 'red', linewidth=3, label='Column Output')
    plt.axvline(x=0.1, color='black', linestyle='--', alpha=0.5, label='Input ON')
    plt.title('Column Output')
    plt.xlabel('Time (s)')
    plt.ylabel('Output')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('simple_example_output.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved as 'simple_example_output.png'")
    
    # Show some interesting properties
    print(f"\nSystem Properties:")
    print(f"Response latency: ~{(np.argmax(np.array(outputs) > 0.01) * 10):.0f}ms")
    print(f"Steady-state reached: ~{(len(outputs) * 10):.0f}ms")
    
    # Layer-specific information
    print(f"\nLayer-specific behavior:")
    l4_response = max(layer_activities['L4']) - min(layer_activities['L4'])
    l23_response = max(layer_activities['L2/3']) - min(layer_activities['L2/3'])
    l5_response = max(layer_activities['L5']) - min(layer_activities['L5'])
    
    print(f"L4 (sensory) response magnitude: {l4_response:.3f}")
    print(f"L2/3 (associative) response magnitude: {l23_response:.3f}")
    print(f"L5 (output) response magnitude: {l5_response:.3f}")
    
    if l4_response > l23_response:
        print("→ Direct sensory processing dominates")
    else:
        print("→ Associative processing dominates")
    
    print(f"\nExample completed successfully!")

if __name__ == "__main__":
    main()