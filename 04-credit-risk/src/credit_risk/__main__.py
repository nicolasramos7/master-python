"""Print the report: `uv run python -m credit_risk`.

Display only. Every number printed here is computed somewhere else.
"""

from .clean import clean
from .data import make_loans
from .evaluate import Scores, baseline, score
from .models import honest_experiment, leaky_experiment, oracle


def format_table(rows: list[tuple[str, Scores]]) -> str:
    """Render one line per model: name, AUC, PR-AUC, log loss, accuracy@0.5.

    TODO: implement. f-strings with width and precision, e.g. f"{x:>8.4f}".
    """
    header = f"{'model':<10}{'auc':>8}{'pr_auc':>8}{'log_loss':>10}{'accuracy':>10}"
    lines = [header]
    for name, s in rows:
        lines.append(
            f"{name:<10}{s.auc:>8.4f}{s.pr_auc:>8.4f}{s.log_loss:>10.4f}{s.accuracy_at_half:>10.4f}"
        )
    return "\n".join(lines)


def main() -> None:
    """Build the comparison table and print it. Extend with thresholds and the bootstrap."""
    loans = clean(make_loans())
    honest = honest_experiment(loans)
    leaky = leaky_experiment(loans)
    best = oracle(loans)
    rows = [
        ("baseline", score(honest.y_true, baseline(honest.train_base_rate, honest.y_true))),
        ("leaky", score(leaky.y_true, leaky.y_score)),
        ("honest", score(honest.y_true, honest.y_score)),
        ("oracle", score(best.y_true, best.y_score)),
    ]
    print(format_table(rows))


if __name__ == "__main__":
    main()
