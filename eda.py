"""
eda.py
------
All Exploratory Data Analysis functions.
Each function returns a Matplotlib / Seaborn figure so Streamlit
can render it with st.pyplot(fig).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ── palette ──────────────────────────────────
PALETTE = "YlGn"
ACCENT  = "#2d6a4f"
BG      = "#f8fdf4"


def _base_fig(figsize=(10, 5)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=BG)
    ax.set_facecolor(BG)
    return fig, ax


# ─────────────────────────────────────────────
# 1. Yield distribution
# ─────────────────────────────────────────────
def plot_yield_distribution(df: pd.DataFrame):
    fig, ax = _base_fig()
    sns.histplot(df["Yield"], bins=60, kde=True, color=ACCENT, ax=ax)
    ax.set_title("Distribution of Crop Yield (kg / hectare)", fontsize=14, color=ACCENT)
    ax.set_xlabel("Yield (kg/ha)")
    ax.set_ylabel("Count")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# 2. Top crops by average yield
# ─────────────────────────────────────────────
def plot_top_crops(df: pd.DataFrame, top_n: int = 15):
    top = (
        df.groupby("Crop")["Yield"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )
    fig, ax = _base_fig(figsize=(10, 6))
    bars = ax.barh(top["Crop"][::-1], top["Yield"][::-1],
                   color=sns.color_palette(PALETTE, top_n))
    ax.set_title(f"Top {top_n} Crops by Average Yield", fontsize=14, color=ACCENT)
    ax.set_xlabel("Average Yield (kg/ha)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# 3. Yield by Season
# ─────────────────────────────────────────────
def plot_yield_by_season(df: pd.DataFrame):
    fig, ax = _base_fig()
    order = df.groupby("Season")["Yield"].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x="Season", y="Yield", order=order,
                palette=PALETTE, ax=ax)
    ax.set_title("Crop Yield by Season", fontsize=14, color=ACCENT)
    ax.set_xlabel("Season")
    ax.set_ylabel("Yield (kg/ha)")
    plt.xticks(rotation=20)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# 4. Yield trend over years
# ─────────────────────────────────────────────
def plot_yield_trend(df: pd.DataFrame):
    trend = df.groupby("Crop_Year")["Yield"].mean().reset_index()
    fig, ax = _base_fig()
    ax.plot(trend["Crop_Year"], trend["Yield"],
            color=ACCENT, linewidth=2.5, marker="o", markersize=4)
    ax.fill_between(trend["Crop_Year"], trend["Yield"],
                    alpha=0.15, color=ACCENT)
    ax.set_title("Average Crop Yield Over the Years", fontsize=14, color=ACCENT)
    ax.set_xlabel("Year")
    ax.set_ylabel("Avg Yield (kg/ha)")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# 5. Top states by production
# ─────────────────────────────────────────────
def plot_top_states(df: pd.DataFrame, top_n: int = 12):
    top = (
        df.groupby("State_Name")["Production"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )
    fig, ax = _base_fig(figsize=(10, 6))
    sns.barplot(data=top, x="Production", y="State_Name",
                palette=PALETTE, ax=ax)
    ax.set_title(f"Top {top_n} States by Total Production", fontsize=14, color=ACCENT)
    ax.set_xlabel("Total Production (tonnes)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# 6. Correlation heatmap
# ─────────────────────────────────────────────
def plot_correlation(df_encoded: pd.DataFrame, feature_cols: list):
    cols = feature_cols + ["Yield"]
    corr = df_encoded[cols].corr()
    fig, ax = plt.subplots(figsize=(9, 7), facecolor=BG)
    ax.set_facecolor(BG)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="YlGn",
                linewidths=0.5, ax=ax)
    ax.set_title("Feature Correlation Matrix", fontsize=14, color=ACCENT)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# 7. Crop-wise season heatmap
# ─────────────────────────────────────────────
def plot_crop_season_heatmap(df: pd.DataFrame, top_n_crops: int = 15):
    top_crops = df["Crop"].value_counts().head(top_n_crops).index
    sub = df[df["Crop"].isin(top_crops)]
    pivot = sub.pivot_table(values="Yield", index="Crop",
                            columns="Season", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(12, 7), facecolor=BG)
    ax.set_facecolor(BG)
    sns.heatmap(pivot, cmap="YlGn", annot=True, fmt=".0f",
                linewidths=0.4, ax=ax)
    ax.set_title("Average Yield by Crop × Season (kg/ha)", fontsize=14, color=ACCENT)
    plt.tight_layout()
    return fig
