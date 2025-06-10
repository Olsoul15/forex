"""
Correlation analysis module for the Forex AI Trading System.

This module provides functionality to analyze correlations between
currency pairs to help identify relationships and avoid overexposure.
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


class CorrelationAnalyzer:
    """
    Analyzer for currency pair correlations.

    This class provides methods to calculate and visualize correlations
    between multiple currency pairs over different time periods.
    """

    def __init__(self):
        """Initialize the correlation analyzer."""
        self.correlation_cache = {}

    def calculate_correlation(
        self,
        price_data: Dict[str, pd.DataFrame],
        method: str = "pearson",
        lookback_periods: int = 30,
        use_returns: bool = True,
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix between multiple currency pairs.

        Args:
            price_data: Dictionary mapping currency pairs to their OHLCV DataFrames
            method: Correlation method ('pearson', 'spearman', or 'kendall')
            lookback_periods: Number of periods to include in correlation calculation
            use_returns: Whether to use returns instead of price levels

        Returns:
            DataFrame with correlation matrix
        """
        # Check valid inputs
        if not price_data:
            logger.warning("No price data provided for correlation calculation")
            return pd.DataFrame()

        # Extract pairs and check for price data
        pairs = list(price_data.keys())

        # Prepare DataFrame for correlation calculation
        price_series = {}
        for pair, data in price_data.items():
            if data.empty:
                logger.warning(f"Empty data for {pair}, skipping")
                continue

            # Use 'close' price or first column if not available
            if "close" in data.columns:
                series = data["close"]
            else:
                series = data.iloc[:, 0]

            # Limit to lookback period
            if len(series) > lookback_periods:
                series = series.tail(lookback_periods)

            # Calculate returns if requested
            if use_returns:
                series = series.pct_change().dropna()

            price_series[pair] = series

        # Check if we have valid data after processing
        if not price_series:
            logger.warning("No valid price series after processing")
            return pd.DataFrame()

        # Create a DataFrame with all price/return series
        df = pd.DataFrame(price_series)

        # Calculate correlation matrix
        correlation = df.corr(method=method)

        # Cache the result
        cache_key = f"{','.join(pairs)}_{method}_{lookback_periods}_{use_returns}"
        self.correlation_cache[cache_key] = {
            "matrix": correlation,
            "timestamp": datetime.now(),
            "parameters": {
                "method": method,
                "lookback_periods": lookback_periods,
                "use_returns": use_returns,
            },
        }

        return correlation

    def create_correlation_heatmap(
        self,
        correlation_matrix: pd.DataFrame,
        title: str = "Currency Pair Correlation Matrix",
        height: int = 600,
        width: Optional[int] = None,
        colorscale: str = "RdBu_r",
        show_scale: bool = True,
        text_auto: bool = True,
        dark_mode: bool = True,
    ) -> go.Figure:
        """
        Create a heatmap visualization of correlation matrix.

        Args:
            correlation_matrix: DataFrame with correlation data
            title: Title for the heatmap
            height: Height of the figure in pixels
            width: Width of the figure in pixels (None for auto)
            colorscale: Colorscale for the heatmap
            show_scale: Whether to show the color scale
            text_auto: Whether to show correlation values in cells
            dark_mode: Whether to use dark mode theme

        Returns:
            Plotly Figure with correlation heatmap
        """
        if correlation_matrix.empty:
            logger.warning("Empty correlation matrix, creating empty chart")
            fig = go.Figure(go.Heatmap(z=[[0]]))
            fig.update_layout(
                title="No correlation data available", height=height, width=width
            )
            return fig

        # Set theme colors based on mode
        bg_color = "#1E1E1E" if dark_mode else "#FFFFFF"
        text_color = "#FFFFFF" if dark_mode else "#333333"
        grid_color = "#333333" if dark_mode else "#DDDDDD"

        # Create heatmap
        fig = go.Figure(
            go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.index,
                colorscale=colorscale,
                zmid=0,  # Center colorscale at 0
                zmin=-1,
                zmax=1,
                showscale=show_scale,
                text=correlation_matrix.values if text_auto else None,
                texttemplate="%{text:.2f}" if text_auto else None,
                colorbar=dict(
                    title="Correlation",
                    titlefont=dict(color=text_color),
                    tickfont=dict(color=text_color),
                ),
                hovertemplate="<b>%{y} vs %{x}</b><br>Correlation: %{z:.4f}<extra></extra>",
            )
        )

        # Update layout
        fig.update_layout(
            title=dict(text=title, font=dict(size=16, color=text_color)),
            height=height,
            width=width,
            plot_bgcolor=bg_color,
            paper_bgcolor=bg_color,
            font=dict(color=text_color),
            margin=dict(l=80, r=40, t=60, b=80),
            xaxis=dict(
                title="", tickangle=-45, tickfont=dict(size=12), gridcolor=grid_color
            ),
            yaxis=dict(title="", tickfont=dict(size=12), gridcolor=grid_color),
        )

        return fig

    def create_correlation_network(
        self,
        correlation_matrix: pd.DataFrame,
        threshold: float = 0.7,
        title: str = "Currency Pair Correlation Network",
        height: int = 700,
        width: Optional[int] = None,
        show_all_pairs: bool = False,
        dark_mode: bool = True,
    ) -> go.Figure:
        """
        Create a network visualization of correlations above threshold.

        Args:
            correlation_matrix: DataFrame with correlation data
            threshold: Minimum absolute correlation to include in the network
            title: Title for the network
            height: Height of the figure in pixels
            width: Width of the figure in pixels (None for auto)
            show_all_pairs: Whether to show all pairs or only those above threshold
            dark_mode: Whether to use dark mode theme

        Returns:
            Plotly Figure with correlation network
        """
        if correlation_matrix.empty:
            logger.warning("Empty correlation matrix, creating empty chart")
            fig = go.Figure()
            fig.update_layout(
                title="No correlation data available", height=height, width=width
            )
            return fig

        # Set theme colors based on mode
        bg_color = "#1E1E1E" if dark_mode else "#FFFFFF"
        text_color = "#FFFFFF" if dark_mode else "#333333"

        # Create pairs lists
        pairs = list(correlation_matrix.columns)

        # Create node positions (circular layout)
        num_nodes = len(pairs)
        angle_step = 2 * np.pi / num_nodes
        radius = 1

        node_positions = {}
        for i, pair in enumerate(pairs):
            angle = i * angle_step
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            node_positions[pair] = (x, y)

        # Create nodes
        node_x = [pos[0] for pos in node_positions.values()]
        node_y = [pos[1] for pos in node_positions.values()]

        # Create edges (connections between pairs with high correlation)
        edge_x = []
        edge_y = []
        edge_colors = []
        edge_widths = []
        hover_texts = []

        for i, pair1 in enumerate(pairs):
            for j, pair2 in enumerate(pairs):
                if i != j:  # Don't connect a pair with itself
                    correlation = correlation_matrix.loc[pair1, pair2]
                    abs_corr = abs(correlation)

                    # Only show edges above threshold
                    if abs_corr >= threshold or show_all_pairs:
                        # Add edge
                        x0, y0 = node_positions[pair1]
                        x1, y1 = node_positions[pair2]

                        edge_x.extend([x0, x1, None])
                        edge_y.extend([y0, y1, None])

                        # Set edge color based on correlation sign
                        if correlation > 0:
                            edge_colors.append(
                                "rgba(76, 175, 80, {opacity})".format(opacity=abs_corr)
                            )
                        else:
                            edge_colors.append(
                                "rgba(244, 67, 54, {opacity})".format(opacity=abs_corr)
                            )

                        # Set edge width based on correlation strength
                        edge_widths.append(abs_corr * 3)

                        # Hover text
                        hover_texts.append(f"{pair1} vs {pair2}: {correlation:.4f}")

        # Create figure
        fig = go.Figure()

        # Add edges
        for i in range(0, len(edge_x), 3):
            fig.add_trace(
                go.Scatter(
                    x=edge_x[i : i + 3],
                    y=edge_y[i : i + 3],
                    mode="lines",
                    line=dict(width=edge_widths[i // 3], color=edge_colors[i // 3]),
                    hoverinfo="text",
                    hovertext=hover_texts[i // 3],
                    showlegend=False,
                )
            )

        # Add nodes
        fig.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                marker=dict(
                    size=15,
                    color="rgba(255, 255, 255, 0.8)",
                    line=dict(width=1, color="rgba(0, 0, 0, 0.5)"),
                ),
                text=pairs,
                textposition="bottom center",
                hoverinfo="text",
                hovertext=pairs,
                showlegend=False,
            )
        )

        # Update layout
        fig.update_layout(
            title=dict(text=title, font=dict(size=16, color=text_color)),
            height=height,
            width=width,
            plot_bgcolor=bg_color,
            paper_bgcolor=bg_color,
            font=dict(color=text_color),
            margin=dict(l=20, r=20, t=60, b=20),
            xaxis=dict(
                showgrid=False, zeroline=False, showticklabels=False, range=[-1.2, 1.2]
            ),
            yaxis=dict(
                showgrid=False, zeroline=False, showticklabels=False, range=[-1.2, 1.2]
            ),
            legend=dict(x=0, y=1, font=dict(color=text_color)),
            annotations=[
                dict(
                    x=0.5,
                    y=-0.1,
                    xref="paper",
                    yref="paper",
                    text=(
                        f"Threshold: {threshold:.2f}"
                        if not show_all_pairs
                        else "All correlations shown"
                    ),
                    showarrow=False,
                    font=dict(color=text_color, size=12),
                )
            ],
        )

        # Add legend for correlation colors
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=10, color="rgba(76, 175, 80, 0.8)"),
                name="Positive Correlation",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=10, color="rgba(244, 67, 54, 0.8)"),
                name="Negative Correlation",
            )
        )

        return fig

    def create_correlation_time_series(
        self,
        price_data: Dict[str, pd.DataFrame],
        pair1: str,
        pair2: str,
        window: int = 20,
        method: str = "pearson",
        use_returns: bool = True,
        title: Optional[str] = None,
        height: int = 500,
        width: Optional[int] = None,
        dark_mode: bool = True,
    ) -> go.Figure:
        """
        Create a time series of rolling correlation between two pairs.

        Args:
            price_data: Dictionary mapping currency pairs to their OHLCV DataFrames
            pair1: First currency pair
            pair2: Second currency pair
            window: Window size for rolling correlation
            method: Correlation method ('pearson', 'spearman', or 'kendall')
            use_returns: Whether to use returns instead of price levels
            title: Title for the chart (auto-generated if None)
            height: Height of the figure in pixels
            width: Width of the figure in pixels (None for auto)
            dark_mode: Whether to use dark mode theme

        Returns:
            Plotly Figure with rolling correlation time series
        """
        # Check valid inputs
        if pair1 not in price_data or pair2 not in price_data:
            logger.warning(f"Missing price data for {pair1} or {pair2}")
            fig = go.Figure()
            fig.update_layout(
                title=f"No data available for {pair1} vs {pair2}",
                height=height,
                width=width,
            )
            return fig

        # Get price data
        df1 = price_data[pair1]
        df2 = price_data[pair2]

        # Check for empty data
        if df1.empty or df2.empty:
            logger.warning(f"Empty data for {pair1} or {pair2}")
            fig = go.Figure()
            fig.update_layout(
                title=f"No data available for {pair1} vs {pair2}",
                height=height,
                width=width,
            )
            return fig

        # Use 'close' price or first column if not available
        series1 = df1["close"] if "close" in df1.columns else df1.iloc[:, 0]
        series2 = df2["close"] if "close" in df2.columns else df2.iloc[:, 0]

        # Calculate returns if requested
        if use_returns:
            series1 = series1.pct_change().dropna()
            series2 = series2.pct_change().dropna()

        # Create a DataFrame with both series
        combined = pd.DataFrame({pair1: series1, pair2: series2})

        # Align data (ensure same dates)
        combined = combined.dropna()

        # Calculate rolling correlation
        rolling_corr = combined[pair1].rolling(window=window).corr(combined[pair2])

        # Set theme colors based on mode
        bg_color = "#1E1E1E" if dark_mode else "#FFFFFF"
        text_color = "#FFFFFF" if dark_mode else "#333333"
        grid_color = "#333333" if dark_mode else "#DDDDDD"

        # Generate title if not provided
        if title is None:
            title = f"Rolling {window}-Period Correlation: {pair1} vs {pair2}"

        # Create figure with subplots: prices on top, correlation on bottom
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=["Price Movements", "Correlation"],
        )

        # Add price series
        fig.add_trace(
            go.Scatter(
                x=series1.index,
                y=series1.values,
                mode="lines",
                name=pair1,
                line=dict(width=1.5, color="#4CAF50"),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=series2.index,
                y=series2.values,
                mode="lines",
                name=pair2,
                line=dict(width=1.5, color="#2196F3"),
                yaxis="y2",
            ),
            row=1,
            col=1,
        )

        # Add correlation series
        fig.add_trace(
            go.Scatter(
                x=rolling_corr.index,
                y=rolling_corr.values,
                mode="lines",
                name=f"{window}-Period Correlation",
                line=dict(width=2, color="#FF9800"),
            ),
            row=2,
            col=1,
        )

        # Add reference lines at correlation levels
        fig.add_shape(
            type="line",
            x0=rolling_corr.index[0],
            x1=rolling_corr.index[-1],
            y0=0.7,
            y1=0.7,
            line=dict(color="rgba(76, 175, 80, 0.5)", width=1, dash="dash"),
            row=2,
            col=1,
        )

        fig.add_shape(
            type="line",
            x0=rolling_corr.index[0],
            x1=rolling_corr.index[-1],
            y0=-0.7,
            y1=-0.7,
            line=dict(color="rgba(244, 67, 54, 0.5)", width=1, dash="dash"),
            row=2,
            col=1,
        )

        fig.add_shape(
            type="line",
            x0=rolling_corr.index[0],
            x1=rolling_corr.index[-1],
            y0=0,
            y1=0,
            line=dict(color="rgba(255, 255, 255, 0.5)", width=1, dash="dot"),
            row=2,
            col=1,
        )

        # Update layout
        fig.update_layout(
            title=dict(text=title, font=dict(size=16, color=text_color)),
            height=height,
            width=width,
            plot_bgcolor=bg_color,
            paper_bgcolor=bg_color,
            font=dict(color=text_color),
            margin=dict(l=60, r=60, t=60, b=40),
            xaxis=dict(
                showgrid=True,
                gridcolor=grid_color,
                zeroline=False,
                tickfont=dict(size=10),
            ),
            yaxis=dict(
                title=pair1,
                showgrid=True,
                gridcolor=grid_color,
                zeroline=False,
                tickfont=dict(size=10),
                titlefont=dict(color="#4CAF50"),
            ),
            yaxis2=dict(
                title=pair2,
                overlaying="y",
                side="right",
                showgrid=False,
                zeroline=False,
                tickfont=dict(size=10),
                titlefont=dict(color="#2196F3"),
            ),
            xaxis2=dict(
                showgrid=True,
                gridcolor=grid_color,
                zeroline=False,
                tickfont=dict(size=10),
            ),
            yaxis3=dict(
                title="Correlation",
                range=[-1.1, 1.1],
                showgrid=True,
                gridcolor=grid_color,
                zeroline=False,
                tickfont=dict(size=10),
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(0,0,0,0)",
            ),
            hovermode="x unified",
        )

        # Add annotations for correlation levels
        fig.add_annotation(
            x=rolling_corr.index[-1],
            y=0.7,
            text="Strong +ve",
            showarrow=False,
            font=dict(size=10, color="rgba(76, 175, 80, 0.8)"),
            xshift=40,
            row=2,
            col=1,
        )

        fig.add_annotation(
            x=rolling_corr.index[-1],
            y=-0.7,
            text="Strong -ve",
            showarrow=False,
            font=dict(size=10, color="rgba(244, 67, 54, 0.8)"),
            xshift=40,
            row=2,
            col=1,
        )

        # Update subplot y-axis titles
        fig.update_yaxes(title_text=pair1, row=1, col=1)
        fig.update_yaxes(title_text="Correlation", row=2, col=1)

        return fig

    def get_currency_exposure(
        self, positions: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate exposure to individual currencies across all positions.

        Args:
            positions: List of position dictionaries with instrument and size

        Returns:
            Dictionary mapping currencies to their net exposure
        """
        # Initialize currency exposures
        exposures = {}

        for position in positions:
            instrument = position.get("instrument", "")
            size = float(position.get("size", 0))

            # Skip invalid positions
            if not instrument or size == 0:
                continue

            # Split instrument into base and quote currencies
            parts = instrument.replace("_", "/").split("/")
            if len(parts) != 2:
                logger.warning(f"Invalid instrument format: {instrument}")
                continue

            base, quote = parts

            # Add exposure to base currency (positive for long, negative for short)
            if base not in exposures:
                exposures[base] = 0
            exposures[base] += size

            # Add exposure to quote currency (negative for long, positive for short)
            if quote not in exposures:
                exposures[quote] = 0
            exposures[quote] -= size

        return exposures

    def analyze_portfolio_correlation(
        self, positions: List[Dict[str, Any]], correlation_matrix: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Analyze portfolio for correlation-based risks.

        Args:
            positions: List of position dictionaries
            correlation_matrix: DataFrame with correlation data

        Returns:
            Dictionary with correlation analysis results
        """
        # Check for valid inputs
        if not positions or correlation_matrix.empty:
            logger.warning("Missing positions or correlation data")
            return {
                "correlated_positions": [],
                "opposing_positions": [],
                "exposure_recommendations": [],
            }

        # Extract instruments from positions
        position_instruments = [
            p.get("instrument", "").replace("_", "/")
            for p in positions
            if p.get("instrument")
        ]

        # Find all pairs of positions
        correlated_positions = []
        opposing_positions = []

        for i, instr1 in enumerate(position_instruments):
            for j, instr2 in enumerate(position_instruments):
                if i < j:  # Only check each pair once
                    # Check if these instruments are in the correlation matrix
                    if (
                        instr1 in correlation_matrix.index
                        and instr2 in correlation_matrix.columns
                    ):
                        correlation = correlation_matrix.loc[instr1, instr2]

                        # Check for high positive correlation
                        if correlation > 0.7:
                            correlated_positions.append(
                                {
                                    "pair1": instr1,
                                    "pair2": instr2,
                                    "correlation": correlation,
                                }
                            )

                        # Check for high negative correlation
                        elif correlation < -0.7:
                            opposing_positions.append(
                                {
                                    "pair1": instr1,
                                    "pair2": instr2,
                                    "correlation": correlation,
                                }
                            )

        # Calculate currency exposures
        exposures = self.get_currency_exposure(positions)

        # Generate recommendations based on exposures and correlations
        recommendations = []

        # Check for overexposure to correlated pairs
        if correlated_positions:
            recommendations.append(
                {
                    "type": "warning",
                    "message": f"You have {len(correlated_positions)} pairs of positions with high positive correlation (>0.7)",
                    "details": "These positions may expose you to similar market risks",
                }
            )

        # Check for opposing positions that might offset each other
        if opposing_positions:
            recommendations.append(
                {
                    "type": "info",
                    "message": f"You have {len(opposing_positions)} pairs of positions with high negative correlation (<-0.7)",
                    "details": "These positions may offset each other's movements",
                }
            )

        # Check for excessive currency exposure
        for currency, exposure in exposures.items():
            if abs(exposure) > 100000:  # Arbitrary threshold, adjust as needed
                recommendations.append(
                    {
                        "type": "warning",
                        "message": f"High exposure to {currency}: {exposure:,.0f} units",
                        "details": "Consider reducing exposure to avoid currency-specific risks",
                    }
                )

        return {
            "correlated_positions": correlated_positions,
            "opposing_positions": opposing_positions,
            "currency_exposures": exposures,
            "recommendations": recommendations,
        }

    def get_correlation_stats(self, correlation_matrix: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics from a correlation matrix.

        Args:
            correlation_matrix: DataFrame with correlation data

        Returns:
            Dictionary with correlation statistics
        """
        if correlation_matrix.empty:
            return {
                "num_pairs": 0,
                "avg_correlation": 0,
                "max_correlation": None,
                "min_correlation": None,
                "strongly_correlated": [],
                "strongly_negatively_correlated": [],
            }

        # Get the lower triangle of the correlation matrix (exclude diagonal)
        mask = np.tril(np.ones(correlation_matrix.shape), k=-1).astype(bool)
        triangle = correlation_matrix.values[mask]

        # Calculate statistics
        avg_correlation = np.mean(triangle)
        max_correlation = np.max(triangle)
        min_correlation = np.min(triangle)

        # Find strongly correlated pairs
        pairs = list(correlation_matrix.columns)
        strongly_correlated = []
        strongly_negatively_correlated = []

        for i, pair1 in enumerate(pairs):
            for j, pair2 in enumerate(pairs):
                if i < j:  # Check each pair once
                    correlation = correlation_matrix.loc[pair1, pair2]

                    if correlation > 0.7:
                        strongly_correlated.append(
                            {"pair1": pair1, "pair2": pair2, "correlation": correlation}
                        )
                    elif correlation < -0.7:
                        strongly_negatively_correlated.append(
                            {"pair1": pair1, "pair2": pair2, "correlation": correlation}
                        )

        return {
            "num_pairs": len(pairs),
            "avg_correlation": avg_correlation,
            "max_correlation": max_correlation,
            "min_correlation": min_correlation,
            "strongly_correlated": strongly_correlated,
            "strongly_negatively_correlated": strongly_negatively_correlated,
        }
