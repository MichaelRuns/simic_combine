"""Clinical visualizations for thyroid dosing analysis."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

from simic_combine.data import load_data, get_treatment_data, get_controlled_cases, CONTROL_STATUS
from simic_combine.model import fit_allometric_model, AllometricModel


# Color scheme for control status
CONTROL_COLORS = {
    1: "#2ecc71",  # Green - Controlled
    2: "#3498db",  # Blue - Undertreated
    3: "#e74c3c",  # Red - Overtreated
    4: "#9b59b6",  # Purple - Elevated TT4 and TSH
    5: "#f39c12",  # Orange - Normal TT4, elevated TSH
}

CONTROL_LABELS = {
    1: "Controlled",
    2: "Undertreated",
    3: "Overtreated",
    4: "High TT4 & TSH",
    5: "Normal TT4, High TSH",
}


def setup_style():
    """Configure matplotlib for publication-quality figures."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def plot_dose_vs_weight(
    df: pd.DataFrame,
    model: AllometricModel,
    ax: Optional[plt.Axes] = None,
    show_ci: bool = True,
) -> plt.Axes:
    """
    Plot total daily dose vs weight with fitted curve.

    Graph 1: Shows dose-weight relationship colored by control status.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))

    # Plot scatter points colored by control status
    for control, color in CONTROL_COLORS.items():
        subset = df[df["Control"] == control]
        if len(subset) > 0:
            ax.scatter(
                subset["Weight"],
                subset["daily_dose_mcg"],
                c=color,
                label=CONTROL_LABELS[control],
                alpha=0.7,
                edgecolors="white",
                linewidth=0.5,
                s=60,
            )

    # Plot fitted curve
    w_range = np.linspace(df["Weight"].min() * 0.9, df["Weight"].max() * 1.1, 100)
    pred, lower, upper = model.predict_with_ci(w_range)

    ax.plot(w_range, pred, color="black", linewidth=2, label="Optimal fit")

    if show_ci:
        ax.fill_between(w_range, lower, upper, alpha=0.2, color="black", label="95% CI")

    # Add equation text
    eq_text = f"Dose = {model.a:.1f} × W^{model.b:.2f}\nR² = {model.r_squared:.3f}"
    ax.text(
        0.05, 0.95, eq_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax.set_xlabel("Weight (kg)")
    ax.set_ylabel("Total Daily Dose (mcg)")
    ax.set_title("T4 Dose vs Weight by Control Status")
    ax.legend(loc="lower right")

    return ax


def plot_dose_per_kg_vs_weight(
    df: pd.DataFrame,
    model: AllometricModel,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot dose per kg vs weight - the key clinical insight.

    Graph 2: Shows smaller kittens need MORE mcg/kg.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))

    # Plot scatter points colored by control status
    for control, color in CONTROL_COLORS.items():
        subset = df[df["Control"] == control]
        if len(subset) > 0:
            ax.scatter(
                subset["Weight"],
                subset["dose_per_kg"],
                c=color,
                label=CONTROL_LABELS[control],
                alpha=0.7,
                edgecolors="white",
                linewidth=0.5,
                s=60,
            )

    # Plot fitted curve (dose per kg)
    w_range = np.linspace(df["Weight"].min() * 0.9, df["Weight"].max() * 1.1, 100)
    dose_per_kg = model.predict_per_kg(w_range)

    ax.plot(w_range, dose_per_kg, color="black", linewidth=2, label="Optimal curve")

    # Add equation text
    eq_text = f"Dose/kg = {model.a:.1f} × W^{model.b - 1:.2f}"
    ax.text(
        0.95, 0.95, eq_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax.set_xlabel("Weight (kg)")
    ax.set_ylabel("Dose per kg (mcg/kg/day)")
    ax.set_title("Dose per kg vs Weight: Smaller Kittens Need Higher mcg/kg")
    ax.legend(loc="upper right")

    return ax


def plot_dose_per_kg_vs_age(
    df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot dose per kg vs age.

    Graph 3: Shows temporal progression of dose requirements.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))

    # Filter to rows with valid age
    plot_df = df.dropna(subset=["Age"])

    # Plot scatter points colored by control status
    for control, color in CONTROL_COLORS.items():
        subset = plot_df[plot_df["Control"] == control]
        if len(subset) > 0:
            ax.scatter(
                subset["Age"],
                subset["dose_per_kg"],
                c=color,
                label=CONTROL_LABELS[control],
                alpha=0.7,
                edgecolors="white",
                linewidth=0.5,
                s=60,
            )

    ax.set_xlabel("Age (days)")
    ax.set_ylabel("Dose per kg (mcg/kg/day)")
    ax.set_title("Dose per kg vs Age by Control Status")
    ax.legend(loc="upper right")

    return ax


def plot_individual_trajectories(
    df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    highlight_controlled: bool = True,
) -> plt.Axes:
    """
    Plot individual kitten dose trajectories over time.

    Graph 4: Shows longitudinal treatment patterns.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 7))

    # Get unique kitten identifiers (use first column which is Animal ID)
    id_col = df.columns[0] if "Animal ID" not in df.columns else "Animal ID"

    # Filter to kittens with multiple observations and valid age
    plot_df = df.dropna(subset=["Age"])
    kitten_counts = plot_df[id_col].value_counts()
    multi_visit_kittens = kitten_counts[kitten_counts >= 3].index

    cmap = plt.cm.tab20
    for i, kitten_id in enumerate(multi_visit_kittens[:15]):  # Limit to 15 for readability
        kitten_data = plot_df[plot_df[id_col] == kitten_id].sort_values("Age")

        # Use different style for controlled vs not
        if highlight_controlled:
            controlled_mask = kitten_data["Control"] == 1
            # Plot controlled points with marker
            ax.plot(
                kitten_data["Age"],
                kitten_data["dose_per_kg"],
                color=cmap(i % 20),
                alpha=0.6,
                linewidth=1.5,
            )
            # Mark controlled observations
            controlled_data = kitten_data[controlled_mask]
            ax.scatter(
                controlled_data["Age"],
                controlled_data["dose_per_kg"],
                color=cmap(i % 20),
                marker="o",
                s=80,
                edgecolors="black",
                linewidth=1,
                zorder=5,
            )
        else:
            ax.plot(
                kitten_data["Age"],
                kitten_data["dose_per_kg"],
                color=cmap(i % 20),
                alpha=0.7,
                linewidth=1.5,
                marker="o",
                markersize=4,
            )

    # Add legend for controlled marker
    if highlight_controlled:
        controlled_marker = plt.Line2D(
            [], [],
            color="gray",
            marker="o",
            linestyle="None",
            markersize=8,
            markeredgecolor="black",
            label="Controlled (status=1)",
        )
        ax.legend(handles=[controlled_marker], loc="upper right")

    ax.set_xlabel("Age (days)")
    ax.set_ylabel("Dose per kg (mcg/kg/day)")
    ax.set_title("Individual Kitten Dosing Trajectories Over Time")

    return ax


def plot_control_distribution(
    df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot distribution of control status by dose range.

    Graph 5: Shows the dosing "sweet spot" for control.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))

    # Create dose bins
    df = df.copy()
    df["dose_bin"] = pd.cut(
        df["dose_per_kg"],
        bins=[0, 15, 25, 35, 45, 60, 200],
        labels=["<15", "15-25", "25-35", "35-45", "45-60", ">60"],
    )

    # Calculate proportions
    pivot = df.groupby(["dose_bin", "Control"]).size().unstack(fill_value=0)
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    # Plot stacked bar
    bottom = np.zeros(len(pivot_pct))
    for control in sorted(CONTROL_COLORS.keys()):
        if control in pivot_pct.columns:
            ax.bar(
                pivot_pct.index,
                pivot_pct[control],
                bottom=bottom,
                color=CONTROL_COLORS[control],
                label=CONTROL_LABELS[control],
                edgecolor="white",
                linewidth=0.5,
            )
            bottom += pivot_pct[control].values

    ax.set_xlabel("Dose Range (mcg/kg/day)")
    ax.set_ylabel("Percentage of Observations")
    ax.set_title("Control Status Distribution by Dose Range")
    ax.legend(loc="upper right", title="Control Status")
    ax.set_ylim(0, 100)

    return ax


def generate_all_figures(
    output_dir: Optional[Path] = None,
    show: bool = True,
) -> None:
    """
    Generate all clinical visualization figures.

    Args:
        output_dir: Directory to save figures. If None, only displays.
        show: Whether to display figures interactively.
    """
    setup_style()

    # Load and prepare data
    df = load_data()
    treatment = get_treatment_data(df)
    controlled = get_controlled_cases(treatment)

    # Fit model
    model = fit_allometric_model(controlled["Weight"], controlled["daily_dose_mcg"])

    # Print model summary
    print("=" * 60)
    print("ALLOMETRIC MODEL FIT RESULTS")
    print("=" * 60)
    print(f"\n{model.formula_string()}")
    print(f"{model.formula_per_kg_string()}")
    print(f"\nR² = {model.r_squared:.3f}")
    print(f"n = {model.n_observations} controlled observations")
    print(f"Parameters: a = {model.a:.2f} ± {model.a_se:.2f}, b = {model.b:.3f} ± {model.b_se:.3f}")
    print("\nDose Reference Table:")
    print(model.dose_table().to_string(index=False))
    print("=" * 60)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Dose vs Weight
    fig1, ax1 = plt.subplots(figsize=(10, 7))
    plot_dose_vs_weight(treatment, model, ax=ax1)
    if output_dir:
        fig1.savefig(output_dir / "fig1_dose_vs_weight.png")
        fig1.savefig(output_dir / "fig1_dose_vs_weight.pdf")

    # Figure 2: Dose/kg vs Weight
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    plot_dose_per_kg_vs_weight(treatment, model, ax=ax2)
    if output_dir:
        fig2.savefig(output_dir / "fig2_dose_per_kg_vs_weight.png")
        fig2.savefig(output_dir / "fig2_dose_per_kg_vs_weight.pdf")

    # Figure 3: Dose/kg vs Age
    fig3, ax3 = plt.subplots(figsize=(10, 7))
    plot_dose_per_kg_vs_age(treatment, ax=ax3)
    if output_dir:
        fig3.savefig(output_dir / "fig3_dose_per_kg_vs_age.png")
        fig3.savefig(output_dir / "fig3_dose_per_kg_vs_age.pdf")

    # Figure 4: Individual Trajectories
    fig4, ax4 = plt.subplots(figsize=(12, 7))
    plot_individual_trajectories(treatment, ax=ax4)
    if output_dir:
        fig4.savefig(output_dir / "fig4_individual_trajectories.png")
        fig4.savefig(output_dir / "fig4_individual_trajectories.pdf")

    # Figure 5: Control Distribution
    fig5, ax5 = plt.subplots(figsize=(10, 7))
    plot_control_distribution(treatment, ax=ax5)
    if output_dir:
        fig5.savefig(output_dir / "fig5_control_distribution.png")
        fig5.savefig(output_dir / "fig5_control_distribution.pdf")

    # Combined summary figure
    fig_summary, axes = plt.subplots(2, 2, figsize=(14, 12))
    plot_dose_vs_weight(treatment, model, ax=axes[0, 0], show_ci=False)
    plot_dose_per_kg_vs_weight(treatment, model, ax=axes[0, 1])
    plot_dose_per_kg_vs_age(treatment, ax=axes[1, 0])
    plot_control_distribution(treatment, ax=axes[1, 1])
    fig_summary.suptitle("Hypothyroid Kitten T4 Dosing Analysis", fontsize=16, fontweight="bold")
    plt.tight_layout()
    if output_dir:
        fig_summary.savefig(output_dir / "summary_figure.png")
        fig_summary.savefig(output_dir / "summary_figure.pdf")

    if show:
        plt.show()


def create_interactive_plot(
    output_path: Optional[Path] = None,
    auto_open: bool = False,
) -> "plotly.graph_objects.Figure":
    """
    Create an interactive Plotly scatter plot with hover information.

    Features:
    - All treatment observations color-coded by control status
    - Fitted allometric curve from controlled cases
    - Hover info: Animal ID, weight, age, dose, control status
    - Interactive legend to toggle groups

    Args:
        output_path: Path to save HTML file. If None, returns figure only.
        auto_open: Whether to open the HTML file in browser.

    Returns:
        Plotly Figure object.
    """
    import plotly.graph_objects as go

    # Load data
    df = load_data()
    treatment = get_treatment_data(df)
    controlled = get_controlled_cases(treatment)

    # Fit model
    model = fit_allometric_model(controlled["Weight"], controlled["daily_dose_mcg"])

    # Get animal ID column
    id_col = df.columns[0] if "Animal ID" not in df.columns else "Animal ID"

    # Create figure
    fig = go.Figure()

    # Add scatter traces for each control status
    control_order = [1, 2, 3, 4, 5]  # Order for legend
    plotly_colors = {
        1: "#2ecc71",  # Green - Controlled
        2: "#3498db",  # Blue - Undertreated
        3: "#e74c3c",  # Red - Overtreated
        4: "#9b59b6",  # Purple - Elevated TT4 and TSH
        5: "#f39c12",  # Orange - Normal TT4, elevated TSH
    }

    for control in control_order:
        subset = treatment[treatment["Control"] == control].copy()
        if len(subset) == 0:
            continue

        # Build hover text
        hover_text = []
        for _, row in subset.iterrows():
            animal_id = row.get(id_col, "Unknown")
            weight = row["Weight"]
            age = row.get("Age", None)
            daily_dose = row["daily_dose_mcg"]
            dose_per_kg = row["dose_per_kg"]

            text = f"<b>{animal_id}</b><br>"
            text += f"Weight: {weight:.2f} kg<br>"
            if pd.notna(age):
                text += f"Age: {int(age)} days<br>"
            text += f"Daily Dose: {daily_dose:.0f} mcg<br>"
            text += f"Dose/kg: {dose_per_kg:.0f} mcg/kg<br>"
            text += f"Status: {CONTROL_LABELS.get(control, str(control))}"
            hover_text.append(text)

        fig.add_trace(go.Scatter(
            x=subset["Weight"],
            y=subset["dose_per_kg"],
            mode="markers",
            name=CONTROL_LABELS.get(control, f"Status {control}"),
            marker=dict(
                size=10,
                color=plotly_colors.get(control, "#888888"),
                opacity=0.7,
                line=dict(width=1, color="white"),
            ),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover_text,
        ))

    # Add fitted curve (from controlled cases only)
    w_range = np.linspace(treatment["Weight"].min() * 0.9, treatment["Weight"].max() * 1.1, 100)
    dose_per_kg_pred = model.predict_per_kg(w_range)

    fig.add_trace(go.Scatter(
        x=w_range,
        y=dose_per_kg_pred,
        mode="lines",
        name=f"Fitted: {model.a:.0f} × W^{model.b - 1:.2f}",
        line=dict(color="black", width=2),
        hovertemplate="Weight: %{x:.2f} kg<br>Predicted: %{y:.0f} mcg/kg<extra></extra>",
    ))

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"<b>Hypothyroid Kitten Dosing: Dose per kg vs Weight</b><br>"
                 f"<sub>Formula: Dose (mcg/kg) = {model.a:.0f} × Weight^{model.b - 1:.2f} | "
                 f"R² = {model.r_squared:.3f} | n = {model.n_observations} controlled cases</sub>",
            x=0.5,
            xanchor="center",
        ),
        xaxis=dict(
            title="Weight (kg)",
            gridcolor="lightgray",
        ),
        yaxis=dict(
            title="Dose per kg (mcg/kg/day)",
            gridcolor="lightgray",
        ),
        legend=dict(
            title="Control Status",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
        ),
        hovermode="closest",
        template="plotly_white",
        height=600,
        width=900,
    )

    # Save to HTML if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path), auto_open=auto_open)
        print(f"Interactive plot saved to: {output_path}")

    return fig


if __name__ == "__main__":
    generate_all_figures(output_dir=Path("figures"), show=True)
    # Also generate interactive plot
    create_interactive_plot(output_path=Path("figures") / "interactive_dose_vs_weight.html")
