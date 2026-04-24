from django import template

register = template.Library()


@register.filter
def as_pct(value, decimals=1):
    """Convert a 0–1 probability to a display percentage string (without the % sign).
    E.g. 0.9123 → '91.2'  so template can append % itself.
    """
    try:
        return f"{float(value) * 100:.{int(decimals)}f}"
    except (ValueError, TypeError):
        return "—"


@register.filter
def human_feature(value):
    """Turn a snake_case feature name into Title Case words.
    E.g. 'quality_of_life_score' → 'Quality Of Life Score'
    """
    return str(value).replace('_', ' ').title()
