"""Pure-Python GUI utilities that can be imported without Streamlit."""


def classify_bias_score(score):
    """Convert a bias score into a label plus Streamlit style."""
    if isinstance(score, str):
        try:
            score = float(score.strip("%")) / 100 if "%" in score else float(score)
        except Exception:
            return "Unknown", "secondary"

    abs_score = abs(score)

    if abs_score < 0.1:
        return "🟢 Low Bias", "success"
    elif abs_score < 0.3:
        return "🟡 Moderate Bias", "warning"
    else:
        return "🔴 High Bias", "error"
