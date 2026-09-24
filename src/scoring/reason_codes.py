"""
Rule-based adverse-action reason codes.

Direct port of the reason-code rule block inside notebook cell 35's
credit_decision_engine() — same 6 rules, same thresholds, same fallback
message, same "top 3" cap.

IMPORTANT — serve-time feature availability gap, read before wiring this
into predict.py:

  The notebook tests this against `orig_row = df_fe.iloc[...]`, i.e. a
  fully engineered + merged training row that has EVERY column below
  populated from real history. At serve time, a live applicant's row only
  has as much of that as application_mapper.py / history_mapper.py could
  actually reconstruct from the payload:

    EXT_SOURCE_MEAN        — available for every applicant (application.py
                              computes it from the I-Score proxy, or 0.0/NaN
                              handling if unavailable).
    CREDIT_INCOME_RATIO     — available for every applicant (application.py).
    EMPLOYED_YEARS          — available for every applicant (application.py).
    INST_LATE_RATIO         — only available for RETURNING customers
                              (history_mapper.py). New customers: rule can't
                              fire (defaults to 0.0, same as "no late history").
    PREV_REFUSED_RATIO      — same: returning customers only.
    CC_AVG_UTILIZATION      — NOT available from either payload today. There
                              is no credit-card adapter yet (no equivalent of
                              application_mapper.py / history_mapper.py for
                              credit_card_balance-shaped data), so this rule
                              will never fire for a live applicant until one
                              is built. Flagging as a known gap, not silently
                              dropping the rule.

  None of this breaks anything — .get(key, default) just means the rule
  quietly can't fire without that signal, exactly like a Kaggle applicant
  with no rows in that source table. But it does mean a returning
  customer's reason codes are more informative than a new customer's,
  which is worth knowing before this goes in front of an underwriter.
"""

# (threshold, feature_key, default_if_missing, message) — same order as
# the notebook, since the caller truncates to the first 3 that fire.
_RULES = [
    ("EXT_SOURCE_MEAN", 1.0, lambda v: v < 0.35,
     "Low External Bureau Score / Credit History Rating"),
    ("INST_LATE_RATIO", 0.0, lambda v: v > 0.15,
     "Historical Late Payment Record on Past Loans"),
    ("CREDIT_INCOME_RATIO", 0.0, lambda v: v > 4.0,
     "High Credit-to-Income Ratio (Excessive Leverage)"),
    ("CC_AVG_UTILIZATION", 0.0, lambda v: v > 0.60,
     "High Revolving Credit Card Balance Utilization"),
    ("PREV_REFUSED_RATIO", 0.0, lambda v: v > 0.25,
     "High Previous Loan Rejection History"),
    ("EMPLOYED_YEARS", 10.0, lambda v: v < 1.0,
     "Short Employment Tenure (< 1 Year)"),
]

_FALLBACK_MESSAGE = "No critical risk flags detected; standard portfolio profile"


def generate_reason_codes(row: dict, max_codes: int = 3) -> list[str]:
    """
    Parameters
    ----------
    row : the applicant's fully engineered feature dict/row — application.py's
        output merged with history_mapper.py's output where available (see
        module docstring for which keys are reliably present for new vs
        returning customers).
    max_codes : how many reason codes to return, in rule order (matches the
        notebook's `reason_codes[:3]`).

    Returns
    -------
    List of human-readable reason-code strings, capped at max_codes. Always
    non-empty — falls back to a "no critical flags" message, same as the
    notebook.
    """
    reason_codes = []
    for feature_key, default, fires, message in _RULES:
        value = row.get(feature_key, default)
        if fires(value):
            reason_codes.append(message)

    if not reason_codes:
        reason_codes.append(_FALLBACK_MESSAGE)

    return reason_codes[:max_codes]