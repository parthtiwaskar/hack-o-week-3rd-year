"""Week 3–4: Python essentials and practical data analysis.

Run with: python week3_4_data_science.py
Add --save-chart to save the visualization as week3_4_sales.png.
Requires numpy, pandas, matplotlib, and seaborn.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def calculate_line_total(unit_price: float, quantity: int) -> float:
    """Return the extended price for one order line."""
    return unit_price * quantity


@dataclass
class SalesReport:
    """Small object that provides summaries for a sales DataFrame."""

    sales: pd.DataFrame

    def total_revenue(self) -> float:
        return float(self.sales["revenue"].sum())

    def revenue_by_region(self) -> pd.DataFrame:
        return (
            self.sales.groupby("region", as_index=False)["revenue"]
            .sum()
            .sort_values("revenue", ascending=False)
        )


def make_sales_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create intentionally imperfect sales and product tables."""
    sales = pd.DataFrame(
        {
            "order_id": [101, 102, 103, 104, 105, 106],
            "product_id": [1, 2, 1, 3, 2, 3],
            "quantity": [2, 1, 3, 2, 4, 1],
            "region": ["North", "South", "North", "West", None, "South"],
        }
    )
    products = pd.DataFrame(
        {
            "product_id": [1, 2, 3],
            "product": ["Notebook", "Pen", "Backpack"],
            "unit_price": [4.50, 1.25, 28.00],
        }
    )
    return sales, products


def clean_and_enrich_sales(
    sales: pd.DataFrame, products: pd.DataFrame
) -> pd.DataFrame:
    """Fill missing regions, merge product details, and derive revenue."""
    cleaned = sales.copy()
    cleaned["region"] = cleaned["region"].fillna("Unassigned")
    cleaned = cleaned.merge(products, on="product_id", how="left", validate="many_to_one")
    cleaned["revenue"] = cleaned["unit_price"] * cleaned["quantity"]
    return cleaned


def python_essentials() -> None:
    """Practice functions, objects, and comprehensions with order lines."""
    order_lines = [(4.50, 2), (1.25, 4), (28.00, 1)]
    line_totals = [calculate_line_total(price, quantity) for price, quantity in order_lines]
    totals_by_order = {
        f"order_{index + 1}": total
        for index, total in enumerate(line_totals)
        if total >= 5
    }
    print("Python essentials")
    print("Line totals:", line_totals)
    print("Dictionary comprehension (totals >= $5):", totals_by_order)


def numpy_examples() -> None:
    """Show array creation, vectorized arithmetic, and broadcasting."""
    prices = np.array([4.50, 1.25, 28.00])
    quantities = np.array([2, 4, 1])
    line_totals = prices * quantities
    discount_rates = np.array([0.0, 0.10, 0.20])
    discounted_prices = prices[:, np.newaxis] * (1 - discount_rates)

    print("\nNumPy")
    print("Vectorized line totals:", line_totals)
    print("Broadcast rows are products; columns are discount rates:")
    print(discounted_prices)


def pandas_examples() -> pd.DataFrame:
    """Clean, merge, and group sales data with Pandas."""
    sales, products = make_sales_data()
    enriched = clean_and_enrich_sales(sales, products)
    summary = (
        enriched.groupby("region", as_index=False)
        .agg(order_count=("order_id", "count"), revenue=("revenue", "sum"))
        .sort_values("revenue", ascending=False)
    )

    print("\nPandas: cleaned and enriched rows")
    print(enriched.to_string(index=False))
    print("\nPandas: revenue by region")
    print(summary.to_string(index=False))
    print("\nOOP report total:", SalesReport(enriched).total_revenue())
    return enriched


def visualize_sales(sales: pd.DataFrame, output: Path | None = None) -> None:
    """Compare regional revenue and individual order values."""
    regional = sales.groupby("region", as_index=False)["revenue"].sum()
    sns.set_theme(style="whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    sns.barplot(data=regional, x="region", y="revenue", hue="region", legend=False, ax=axes[0])
    axes[0].set(title="Revenue by region", xlabel="Region", ylabel="Revenue ($)")

    sns.scatterplot(
        data=sales,
        x="quantity",
        y="revenue",
        hue="product",
        s=90,
        ax=axes[1],
    )
    axes[1].set(title="Order quantity and revenue", xlabel="Quantity", ylabel="Revenue ($)")
    figure.tight_layout()

    if output:
        figure.savefig(output, dpi=160, bbox_inches="tight")
        print(f"\nSaved chart to {output}")
    else:
        plt.show()
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save-chart",
        action="store_true",
        help="save the chart to week3_4_sales.png instead of opening a window",
    )
    args = parser.parse_args()

    python_essentials()
    numpy_examples()
    enriched_sales = pandas_examples()
    chart_path = Path("week3_4_sales.png") if args.save_chart else None
    visualize_sales(enriched_sales, chart_path)


if __name__ == "__main__":
    main()