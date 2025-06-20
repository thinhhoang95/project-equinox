#!/usr/bin/env python3
"""
Standalone script to visualize CostRev3 piecewise linear functions and coefficients.

This script provides the same functionality as the Jupyter notebook but as a 
standalone Python script that can be run directly.

Usage:
    python src/equinox/cost/visualize_cost_rev3.py [--case-dir path] [--checkpoint name]
"""

import sys
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.equinox.cost.cost_rev3 import CostRev3, DEFAULT_KNOTS_AC_DIST, DEFAULT_KNOTS_WIND


def load_trained_model(case_dir: str = "data/cases/LEMD_EGLL", 
                       checkpoint_name: Optional[str] = None,
                       device: str = "cuda") -> Tuple[CostRev3, Dict[str, Any]]:
    """Load a trained CostRev3 model from checkpoint."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    results_dir = Path(case_dir) / "batch_sgd_results"
    
    if results_dir.exists():
        checkpoints = list(results_dir.glob("checkpoint_iter_*.pt"))
        
        if checkpoints:
            if checkpoint_name:
                checkpoint_path = results_dir / checkpoint_name
            else:
                # Find latest checkpoint
                checkpoint_nums = []
                for cp in checkpoints:
                    try:
                        num = int(cp.stem.split('_')[-1])
                        checkpoint_nums.append((num, cp))
                    except ValueError:
                        continue
                
                if checkpoint_nums:
                    checkpoint_path = max(checkpoint_nums, key=lambda x: x[0])[1]
                    print(f"Loading latest checkpoint: {checkpoint_path.name}")
                else:
                    checkpoint_path = None
            
            if checkpoint_path and checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location=device)
                
                model = CostRev3(
                    beta0=0.0, beta1=1e-2, beta2=0.0, beta3=1.0,
                    num_waypoints=1000,
                    alpha_pref_reg=1.0,
                    device=device
                )
                
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                
                print(f"✓ Loaded trained model from {checkpoint_path.name}")
                print(f"  Training iteration: {checkpoint['iteration']}")
                
                return model, checkpoint
    
    # Create default model if no checkpoint found
    print("No trained model found. Creating new model with default parameters.")
    model = CostRev3(
        beta0=0.1, beta1=1e-2, beta2=-0.1, beta3=1.0,
        num_waypoints=1000,
        alpha_pref_reg=1.0,
        device=device
    )
    model.eval()
    
    return model, {}


def print_model_coefficients(model: CostRev3, title: str = "Model Coefficients"):
    """Print all model coefficients in a formatted way."""
    print(f"\n{'='*50}")
    print(f"{title:^50}")
    print(f"{'='*50}")
    
    # Beta coefficients
    print(f"Beta Coefficients:")
    print(f"  β₀ (intercept):     {model.beta0.item():>12.6f}")
    print(f"  β₁ (AC*dist coeff): {model.beta1.item():>12.6f}")  
    print(f"  β₂ (wind coeff):    {model.beta2.item():>12.6f}")
    
    # Regularization parameter
    print(f"\nRegularization:")
    print(f"  α (pref reg):       {model.alpha_pref_reg.item():>12.6f}")
    
    # Piecewise linear model parameters
    print(f"\nAirspace Charge PLM (monotonic {model.plm_ac_dist.monotonic_type}):")
    print(f"  Knot points:        {model.plm_ac_dist.knot_points.tolist()}")
    print(f"  Initial intercept:  {model.plm_ac_dist.initial_intercept.item():>12.6f}")
    print(f"  First slope:        {model.plm_ac_dist.first_slope.item():>12.6f}")
    if model.plm_ac_dist.knot_points.numel() > 0:
        print(f"  Slope increments:   {model.plm_ac_dist.unconstrained_slope_increments.data.tolist()}")
        slopes = model.plm_ac_dist.get_slopes()
        print(f"  Actual slopes:      {slopes.tolist()}")
    
    print(f"\nWind PLM (monotonic {model.plm_wind.monotonic_type}):")
    print(f"  Knot points:        {model.plm_wind.knot_points.tolist()}")
    print(f"  Initial intercept:  {model.plm_wind.initial_intercept.item():>12.6f}")
    print(f"  First slope:        {model.plm_wind.first_slope.item():>12.6f}")
    if model.plm_wind.knot_points.numel() > 0:
        print(f"  Slope increments:   {model.plm_wind.unconstrained_slope_increments.data.tolist()}")
        slopes = model.plm_wind.get_slopes()
        print(f"  Actual slopes:      {slopes.tolist()}")
    
    print(f"{'='*50}\n")


def plot_piecewise_linear_function(plm, x_range: Tuple[float, float], 
                                   title: str, xlabel: str, ylabel: str,
                                   num_points: int = 1000, save_path: Optional[str] = None):
    """Plot a piecewise linear monotonic function."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Generate x values
    x_min, x_max = x_range
    x_vals = torch.linspace(x_min, x_max, num_points)
    
    # Evaluate the function
    with torch.no_grad():
        y_vals = plm(x_vals)
    
    # Convert to numpy for plotting
    x_np = x_vals.cpu().numpy()
    y_np = y_vals.cpu().numpy()
    
    # Plot 1: Function values
    ax1.plot(x_np, y_np, 'b-', linewidth=2, label='PLM Function')
    
    # Mark knot points
    if plm.knot_points.numel() > 0:
        knot_x = plm.knot_points.cpu().numpy()
        knot_y = plm(plm.knot_points).cpu().numpy()
        ax1.scatter(knot_x, knot_y, c='red', s=100, zorder=5, 
                   label=f'Knot Points ({len(knot_x)})')
        
        # Add vertical lines at knots
        for kx in knot_x:
            ax1.axvline(x=kx, color='red', alpha=0.3, linestyle='--')
    
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title(f'{title} - Function Values')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Slopes
    if plm.knot_points.numel() > 0:
        slopes = plm.get_slopes().cpu().numpy()
        knot_points = plm.knot_points.cpu().numpy()
        
        # Create x positions for slopes
        x_positions = [x_min] + knot_points.tolist() + [x_max]
        
        # Plot slopes as step function
        for i, slope in enumerate(slopes):
            x_start = x_positions[i]
            x_end = x_positions[i + 1]
            ax2.hlines(slope, x_start, x_end, colors='green', linewidth=3,
                      label=f'Slope {i+1}: {slope:.4f}' if i < 3 else None)
            
            # Mark slope changes
            if i < len(slopes) - 1:
                ax2.axvline(x=x_end, color='red', alpha=0.5, linestyle='--')
        
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel('Slope')
        ax2.set_title(f'{title} - Slopes by Segment')
        ax2.grid(True, alpha=0.3)
        if len(slopes) <= 3:
            ax2.legend()
    else:
        # Single slope case
        slope = plm.get_slopes().item()
        ax2.axhline(y=slope, color='green', linewidth=3, 
                   label=f'Constant Slope: {slope:.4f}')
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel('Slope')
        ax2.set_title(f'{title} - Constant Slope')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to {save_path}")
    
    return fig


def compare_cost_formulations(model: CostRev3, ac_dist_val: float = 50.0, wind_val: float = 10.0):
    """Compare different cost formulations to understand the CostRev3 implementation."""
    print(f"\n{'='*70}")
    print("COST FORMULATION COMPARISON")
    print(f"{'='*70}")
    print(f"Test point: AC×Distance = {ac_dist_val}, Wind = {wind_val} knots")
    print()
    
    with torch.no_grad():
        ac_dist_tensor = torch.tensor([ac_dist_val], dtype=torch.float32)
        wind_tensor = torch.tensor([wind_val], dtype=torch.float32)
        
        # Get PLM outputs
        plm_ac_out = model.plm_ac_dist(ac_dist_tensor)
        plm_wind_out = model.plm_wind(wind_tensor)
        
        print("PLM Outputs:")
        print(f"  PLM_AC({ac_dist_val}) = {plm_ac_out.item():.6f}")
        print(f"  PLM_Wind({wind_val}) = {plm_wind_out.item():.6f}")
        print()
        
        # CostRev3 actual formulation (with special AC component)
        ac_component_special = ac_dist_tensor / (450.0 + plm_wind_out)
        cost_rev3_actual = (model.beta0 + 
                           model.beta1 * ac_component_special + 
                           model.beta2 * plm_wind_out)
        
        print("CostRev3 Actual Formulation (with special AC component):")
        print(f"  AC_component = AC×Distance / (450.0 + PLM_Wind)")
        print(f"  AC_component = {ac_dist_val} / (450.0 + {plm_wind_out.item():.6f}) = {ac_component_special.item():.6f}")
        print(f"  Cost = β₀ + β₁×AC_component + β₂×PLM_Wind")
        print(f"  Cost = {model.beta0.item():.6f} + {model.beta1.item():.6f}×{ac_component_special.item():.6f} + {model.beta2.item():.6f}×{plm_wind_out.item():.6f}")
        print(f"  Cost = {cost_rev3_actual.item():.6f}")
        print()
        
        print("Interpretation:")
        print(f"  The special formulation AC×Distance/(450+PLM_Wind) creates a")
        print(f"  time-like cost where higher tailwinds reduce the effective")
        print(f"  cost of distance/charges by increasing ground speed.")
        print(f"  The 450 likely represents a baseline true airspeed in knots.")


def main():
    """Main function for the visualization script."""
    parser = argparse.ArgumentParser(description="Visualize CostRev3 piecewise linear functions")
    parser.add_argument("--case-dir", default="data/cases/LEMD_EGLL",
                       help="Directory containing case data")
    parser.add_argument("--checkpoint", help="Specific checkpoint file name")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--save-plots", action="store_true", help="Save plots to files")
    parser.add_argument("--no-display", action="store_true", help="Don't display plots (useful for headless)")
    
    args = parser.parse_args()
    
    print("CostRev3 Piecewise Linear Function Visualization")
    print("=" * 50)
    
    # Load model
    try:
        cost_model, checkpoint_info = load_trained_model(
            args.case_dir, args.checkpoint, args.device
        )
        print(f"Model device: {cost_model.device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1
    
    # Print coefficients
    print_model_coefficients(cost_model, "CostRev3 Model Coefficients")
    
    # Plot functions
    print("Plotting Airspace Charge Piecewise Linear Function...")
    fig_ac = plot_piecewise_linear_function(
        cost_model.plm_ac_dist,
        x_range=(0, 300),
        title="Airspace Charge Component",
        xlabel="AC × Distance (hundreds of euros × nm)",
        ylabel="PLM Output",
        save_path="ac_plm_plot.png" if args.save_plots else None
    )
    
    print("Plotting Wind Piecewise Linear Function...")
    fig_wind = plot_piecewise_linear_function(
        cost_model.plm_wind,
        x_range=(-100, 100),
        title="Wind Component",
        xlabel="Tailwind (knots)",
        ylabel="PLM Output",
        save_path="wind_plm_plot.png" if args.save_plots else None
    )
    
    # Compare formulations
    compare_cost_formulations(cost_model, ac_dist_val=50.0, wind_val=10.0)
    
    # Display plots if not in headless mode
    if not args.no_display:
        plt.show()
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("MODEL STATISTICS")
    print(f"{'='*60}")
    print(f"Total parameters: {sum(p.numel() for p in cost_model.parameters())}")
    print(f"Trainable parameters: {sum(p.numel() for p in cost_model.parameters() if p.requires_grad)}")
    print(f"AC PLM knot points: {cost_model.plm_ac_dist.knot_points.numel()}")
    print(f"Wind PLM knot points: {cost_model.plm_wind.knot_points.numel()}")
    
    if checkpoint_info:
        print(f"\nCHECKPOINT INFO:")
        if 'iteration' in checkpoint_info:
            print(f"Training iteration: {checkpoint_info['iteration']}")
        if 'training_history' in checkpoint_info:
            history = checkpoint_info['training_history']
            if 'avg_log_likelihood' in history and history['avg_log_likelihood']:
                print(f"Final avg log likelihood: {history['avg_log_likelihood'][-1]:.6f}")
            if 'gradient_norms' in history and history['gradient_norms']:
                print(f"Final gradient norm: {history['gradient_norms'][-1]:.6f}")
    
    print("\n✓ Visualization complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 