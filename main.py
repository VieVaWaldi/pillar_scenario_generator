import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Parse the CSV with European number format (comma as decimal, semicolon as separator)
def parse_data(filepath):
    """Parse the CSV file with European formatting."""
    rows = []
    with open(filepath, "r") as f:
        lines = f.readlines()
        header = lines[0].strip()

        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            # Split by semicolon
            parts = line.split(";")
            # First part contains "Class,Ratio" - split by comma for the class value
            first_parts = parts[0].split(",")
            class_val = float(first_parts[0] + "." + first_parts[1])
            ratio_idx = int(parts[1])
            intercolumniation = float(parts[2].replace(",", "."))
            pillar = float(parts[3].replace(",", "."))
            ratio_values = float(parts[4].replace(",", "."))

            rows.append(
                {
                    "Distance": class_val,  # Class is our distance/y value
                    "Ratio": ratio_idx,
                    "Intercolumniation": intercolumniation,
                    "Pillar": pillar,
                    "Ratio_Values": ratio_values,
                }
            )

    return pd.DataFrame(rows)


# Load and parse data
print("=" * 60)
print("LOADING DATA")
print("=" * 60)
df = parse_data("data.csv")
print("\nParsed DataFrame:")
print(df.to_string())
print(f"\nShape: {df.shape}")

# Define features and target
y = df["Distance"].values
feature_names = ["Ratio", "Intercolumniation", "Pillar", "Ratio_Values"]
X = df[feature_names].values

print("\n" + "=" * 60)
print("LINEAR REGRESSION ANALYSIS")
print("=" * 60)

# Fit linear regression with all features
model = LinearRegression()
model.fit(X, y)

# Predictions
y_pred = model.predict(X)

# Metrics
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

print(f"\nModel Coefficients:")
print(f"  Intercept: {model.intercept_:.4f}")
for name, coef in zip(feature_names, model.coef_):
    print(f"  {name}: {coef:.4f}")

print(f"\nModel Performance:")
print(f"  R² Score: {r2:.4f}")
print(f"  RMSE: {rmse:.4f}")

# Individual feature regressions
print("\n" + "-" * 40)
print("Individual Feature Regressions:")
print("-" * 40)

for i, feature in enumerate(feature_names):
    X_single = df[[feature]].values
    model_single = LinearRegression()
    model_single.fit(X_single, y)
    y_pred_single = model_single.predict(X_single)
    r2_single = r2_score(y, y_pred_single)

    print(f"\n{feature}:")
    print(f"  Coefficient: {model_single.coef_[0]:.4f}")
    print(f"  Intercept: {model_single.intercept_:.4f}")
    print(f"  R² Score: {r2_single:.4f}")

# 3D Visualizations
print("\n" + "=" * 60)
print("CREATING 3D VISUALIZATIONS")
print("=" * 60)

# Create color map based on Distance values for consistent coloring
colors = plt.cm.viridis(
    (df["Distance"] - df["Distance"].min())
    / (df["Distance"].max() - df["Distance"].min())
)

# Define the visualization pairs
viz_pairs = [
    ("Distance", "Ratio", "Intercolumniation"),
    ("Distance", "Ratio", "Pillar"),
    ("Distance", "Ratio", "Ratio_Values"),
    ("Distance", "Intercolumniation", "Pillar"),
    ("Distance", "Intercolumniation", "Ratio_Values"),
    ("Distance", "Pillar", "Ratio_Values"),
]

# Create a figure with subplots
fig = plt.figure(figsize=(16, 12))
fig.suptitle(
    "3D Visualizations: Distance vs Feature Pairs", fontsize=14, fontweight="bold"
)

for idx, (x_col, y_col, z_col) in enumerate(viz_pairs, 1):
    ax = fig.add_subplot(2, 3, idx, projection="3d")

    scatter = ax.scatter(
        df[x_col],
        df[y_col],
        df[z_col],
        c=df["Distance"],
        cmap="viridis",
        s=80,
        alpha=0.8,
        edgecolors="black",
        linewidth=0.5,
    )

    ax.set_xlabel(x_col, fontsize=9)
    ax.set_ylabel(y_col, fontsize=9)
    ax.set_zlabel(z_col, fontsize=9)
    ax.set_title(f"{idx}. X={x_col[:4]}, Y={y_col[:4]}, Z={z_col[:4]}", fontsize=10)
    ax.tick_params(labelsize=7)

# Add colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = fig.colorbar(scatter, cax=cbar_ax)
cbar.set_label("Distance", fontsize=10)

plt.tight_layout(rect=[0, 0, 0.9, 0.95])
plt.savefig("3d_visualizations.png", dpi=150, bbox_inches="tight")
print("\nSaved: 3d_visualizations.png")

# Also create individual larger plots for better detail
fig2, axes = plt.subplots(2, 2, figsize=(14, 12), subplot_kw={"projection": "3d"})
fig2.suptitle("Key 3D Visualizations (Detailed)", fontsize=14, fontweight="bold")

key_pairs = [
    ("Distance", "Intercolumniation", "Pillar"),
    ("Distance", "Ratio_Values", "Intercolumniation"),
    ("Distance", "Ratio_Values", "Pillar"),
    ("Intercolumniation", "Pillar", "Ratio_Values"),
]

for ax, (x_col, y_col, z_col) in zip(axes.flat, key_pairs):
    scatter = ax.scatter(
        df[x_col],
        df[y_col],
        df[z_col],
        c=df["Distance"],
        cmap="plasma",
        s=100,
        alpha=0.8,
        edgecolors="black",
        linewidth=0.5,
    )

    ax.set_xlabel(x_col, fontsize=10)
    ax.set_ylabel(y_col, fontsize=10)
    ax.set_zlabel(z_col, fontsize=10)
    ax.set_title(f"X={x_col}, Y={y_col}, Z={z_col}", fontsize=11)

plt.tight_layout()
plt.savefig("3d_detailed.png", dpi=150, bbox_inches="tight")
print("Saved: 3d_detailed.png")

# Regression summary plot
fig3, axes3 = plt.subplots(2, 2, figsize=(12, 10))
fig3.suptitle(
    "Linear Regression: Distance vs Individual Features", fontsize=14, fontweight="bold"
)

for ax, feature in zip(axes3.flat, feature_names):
    X_single = df[[feature]].values
    model_single = LinearRegression()
    model_single.fit(X_single, y)

    # Scatter plot
    ax.scatter(
        df[feature],
        y,
        c="steelblue",
        s=60,
        alpha=0.7,
        edgecolors="black",
        linewidth=0.5,
    )

    # Regression line
    x_range = np.linspace(df[feature].min(), df[feature].max(), 100).reshape(-1, 1)
    y_line = model_single.predict(x_range)
    ax.plot(
        x_range,
        y_line,
        "r-",
        linewidth=2,
        label=f"R² = {r2_score(y, model_single.predict(X_single)):.3f}",
    )

    ax.set_xlabel(feature, fontsize=11)
    ax.set_ylabel("Distance", fontsize=11)
    ax.set_title(f"Distance vs {feature}", fontsize=12)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("regression_plots.png", dpi=150, bbox_inches="tight")
print("Saved: regression_plots.png")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)

plt.show()
