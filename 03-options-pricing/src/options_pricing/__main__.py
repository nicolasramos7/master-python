"""Print the convergence tables: `uv run python -m options_pricing`.

Display only. Every number printed here is computed in `convergence.py`.
"""

from numpy.random import default_rng

from .convergence import Row, binomial_convergence, mc_convergence

SPOT = 100.0
STRIKE = 100.0
RATE = 0.02
VOL = 0.2
EXPIRY = 1.0


def format_table(title: str, rows: list[Row]) -> str:
    """Render rows as a fixed-width table."""
    lines = []
    lines.append(f"{title:^61}") 
    lines.append("-" * 61)  
    lines.append(f"{'n':>12}{'estimate':>13}{'mean_abs_err':>15}{'ratio':>7}{'abs_mean_err':>14}")


    for i, row in enumerate(rows):
        ratio = (
            "-"
            if i == 0
            else f"{rows[i - 1].mean_abs_error / row.mean_abs_error:.2f}"
        )
        lines.append(
            f"{row.n:>12,}"
            f"{row.estimate:>13.4f}"
            f"{row.mean_abs_error:>15.6f}"
            f"{ratio:>7}"
            f"{row.abs_mean_error:>14.6f}"
        )

    return "\n".join(lines)

def main() -> None:
    """TODO: build both tables and print them."""
    rng = default_rng(20260922)
    print(
        format_table(
            "Monte Carlo",
            mc_convergence(
                rng, SPOT, STRIKE, RATE, VOL, EXPIRY, (1_000, 10_000, 100_000, 1_000_000)
            ),
        )
    )
    print(
        format_table(
            "Binomial (CRR)",
            binomial_convergence(SPOT, STRIKE, RATE, VOL, EXPIRY, (10, 100, 1_000, 10_000)),
        )
    )


if __name__ == "__main__":
    main()
