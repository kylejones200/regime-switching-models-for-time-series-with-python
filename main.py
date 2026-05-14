#!/usr/bin/env python3
"""
Regime Switching Models for Time Series
Markov switching models for time series with structural breaks.
"""

import sys
from pathlib import Path

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# Add src to path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import consolidated utilities (signalplot already applied in src/__init__.py)
from src import (
    load_config,
    load_time_series,
    ensure_output_dir,
    get_output_dir,
    save_plot,
)

from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression


def main(plot: bool = False):
    """Main execution function."""
    script_dir = Path(__file__).parent
    
    # Load configuration using consolidated loader
    config = load_config()
    
    # Load data using consolidated loader
    series = load_time_series(
        config["data"]["input_file"],
        date_column=config["data"].get("date_col", "date"),
        value_column=config["data"].get("value_col", "value")
    )
    
    logger.info(f"Loaded {len(series)} data points")
    
    data = series.values
    
    # Fit Markov switching model
    logger.info("\nFitting Markov switching model...")
    model = MarkovRegression(
        data,
        k_regimes=config["model"]["k_regimes"],
        trend=config["model"]["trend"],
        switching_variance=config["model"]["switching_variance"],
    )
    
    result = model.fit()
    logger.info("\nMarkov Switching Model Results:")
    logger.info(result.summary())
    
    logger.info("\nTransition Matrix:")
    logger.info(result.regime_transition)
    
    smoothed_probs = result.smoothed_marginal_probabilities
    
    # Create visualization
    logger.info("\nCreating visualization...")
    if plot:
        fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
        axes[0].plot(
            series.index,
            data,
            "k-",
            linewidth=config.get("plotting", {}).get("linewidth", 1.5),
            alpha=config.get("plotting", {}).get("alpha", 0.8),
            label="Time Series",
        )
        axes[0].set_title(config.get("plot_titles", {}).get("regime_switching", "Regime Switching Model"))
        axes[0].set_ylabel("Value")
        axes[0].legend(loc="best")
        axes[0].grid(True, alpha=0.3)
    
        for regime in range(config["model"]["k_regimes"]):
            axes[1].plot(
                series.index,
                smoothed_probs[:, regime],
                linewidth=config.get("plotting", {}).get("linewidth", 1.5),
                alpha=config.get("plotting", {}).get("alpha", 0.8),
                label=f"Regime {regime + 1} Probability",
            )
        axes[1].set_title("Smoothed Regime Probabilities")
        axes[1].set_ylabel("Probability")
        axes[1].set_xlabel("Date")
        axes[1].legend(loc="best")
        axes[1].grid(True, alpha=0.3)
    
        plt.tight_layout()
        output_dir = ensure_output_dir(get_output_dir(config, script_dir))
        save_plot(fig, output_dir / "regime_switching.png", dpi=300)
        logger.info(f"Plot saved to: {output_dir / 'regime_switching.png'}")
    
        logger.info("\n Regime switching analysis complete")
    
        if config.get("plotting", {}).get("show_plot", True):
            plt.show()
        else:
            plt.close(fig)


if __name__ == "__main__":
    main()
