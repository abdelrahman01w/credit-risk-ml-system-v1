"""
Aggregates a returning customer's own loan and bureau history
(internal_history.previous_bank_loans[] and
iscore_report_fields.bureau_facilities[]) into the PREV_*/INST_*/BUREAU_*
feature columns the model expects.

Only called for returning customers (internal_history.internal_history_missing
== 0). New customers skip this entirely and get "no history" behavior
identical to a Kaggle applicant with zero rows in previous_application.csv /
installments_payments.csv (left-join + fillna(0), same as previous_app.py /
installments.py do at training time).

IMPORTANT — this is a reconstruction from a much thinner record than the raw
Kaggle tables, not a drop-in equivalent:

  - previous_app.py's PREV_* features come from previous_application.csv,
    which has AMT_APPLICATION, NAME_PORTFOLIO, DAYS_DECISION, CNT_PAYMENT,
    AMT_DOWN_PAYMENT — none of which exist on PreviousBankLoan. The PREV_*
    columns that depend on those (PREV_AVG_APPLICATION, PREV_AVG_CNT_PAYMENT,
    PREV_AVG_DOWN_PAYMENT, PREV_AVG_CREDIT_TO_APPLICATION, PREV_RECENT_1Y/2Y_*,
    PREV_CONTRACT_*, PREV_PRODUCT_*) have no honest source here and are
    deliberately left out. build_matrix.py's reindex-against-feature_names_v2
    step (see pos_cash.py's docstring) will fill those as missing/0 the same
    way it already handles any other absent column — but a returning
    customer's PREV_* block is INCOMPLETE, not equivalent to training-time
    PREV_*. Flag this in the model card once build_matrix.py exists.

  - INST_* IS reconstructable close to 1:1: paid_on_time_count /
    late_payments_count / severe_late_count / max_days_past_due are basically
    the same signal installments.py derives from raw installment rows, just
    pre-aggregated per loan instead of per installment.

  - BUREAU_* has no equivalent module yet (bureau.py hasn't been built). The
    column names below are a best-effort naming guess and MUST be reconciled
    with bureau.py once it exists, so training and serving produce identical
    column names for the same concept — don't treat these as final.
"""
import numpy as np

# Loan-record statuses treated as "approved"/"refused" for PREV_APPROVED_COUNT
# / PREV_REFUSED_COUNT. Extend as new status values show up in production.
_APPROVED_STATUSES = {"closed", "active", "current"}
_REFUSED_STATUSES = {"defaulted", "written_off", "rejected"}


def build_history_features_for_customer(internal_history: dict, bureau_facilities: list) -> dict:
    """
    Parameters
    ----------
    internal_history : ReturningCustomerInternalHistory as a dict (has
        .previous_bank_loans[] — that's the only part this function reads).
    bureau_facilities : iscore_report_fields.bureau_facilities[] — passed
        separately because it comes from the I-Score report, not the bank's
        own records.

    Returns
    -------
    dict of {feature_name: value}. Merge into the row from
    application_mapper.map_payload_to_application_row() with row.update(...).
    """
    loans = internal_history.get("previous_bank_loans", [])
    features = {}
    features.update(_prev_and_inst_features(loans))
    features.update(_bureau_features(bureau_facilities))
    return features


def _prev_and_inst_features(loans: list) -> dict:
    if not loans:
        return {
            "PREV_APP_COUNT": 0, "PREV_APPROVED_COUNT": 0, "PREV_REFUSED_COUNT": 0,
            "PREV_APPROVED_RATIO": 0.0, "PREV_REFUSED_RATIO": 0.0,
            "PREV_AVG_CREDIT": 0.0, "PREV_AVG_ANNUITY": 0.0,
            "INST_PAYMENT_COUNT": 0, "INST_LATE_COUNT": 0, "INST_SEVERE_LATE_COUNT": 0,
            "INST_LATE_RATIO": 0.0, "INST_SEVERE_LATE_RATIO": 0.0,
            "INST_AVG_DAYS_LATE": 0.0, "INST_MAX_DAYS_LATE": 0,
        }

    n_loans = len(loans)
    approved = sum(1 for l in loans if l["status"] in _APPROVED_STATUSES)
    refused = sum(1 for l in loans if l["status"] in _REFUSED_STATUSES)

    avg_credit = float(np.mean([l["principal_amount"] for l in loans]))
    # ASSUMPTION: PreviousBankLoan has no AMT_ANNUITY equivalent. Approximated
    # as principal / tenure (straight-line), NOT the true amortized annuity
    # previous_app.py's PREV_AVG_ANNUITY is trained on. Revisit if this
    # feature shows up as important and the approximation looks too crude.
    avg_annuity = float(np.mean([
        l["principal_amount"] / l["tenure_months"] if l["tenure_months"] else 0.0
        for l in loans
    ]))

    total_installments = sum(l["installments_count"] for l in loans)
    total_late = sum(l["late_payments_count"] for l in loans)
    total_severe_late = sum(l["severe_late_count"] for l in loans)
    max_days_late = max((l["max_days_past_due"] for l in loans), default=0)
    # Per-loan max_days_past_due averaged across loans — coarser than
    # installments.py's true per-installment average, but the closest signal
    # available from this pre-aggregated-per-loan record.
    avg_days_late = float(np.mean([l["max_days_past_due"] for l in loans]))

    return {
        "PREV_APP_COUNT": n_loans,
        "PREV_APPROVED_COUNT": approved,
        "PREV_REFUSED_COUNT": refused,
        "PREV_APPROVED_RATIO": approved / n_loans,
        "PREV_REFUSED_RATIO": refused / n_loans,
        "PREV_AVG_CREDIT": avg_credit,
        "PREV_AVG_ANNUITY": avg_annuity,
        "INST_PAYMENT_COUNT": total_installments,
        "INST_LATE_COUNT": total_late,
        "INST_SEVERE_LATE_COUNT": total_severe_late,
        "INST_LATE_RATIO": total_late / total_installments if total_installments else 0.0,
        "INST_SEVERE_LATE_RATIO": total_severe_late / total_installments if total_installments else 0.0,
        "INST_AVG_DAYS_LATE": avg_days_late,
        "INST_MAX_DAYS_LATE": max_days_late,
    }


def _bureau_features(bureau_facilities: list) -> dict:
    if not bureau_facilities:
        return {
            "BUREAU_FACILITY_COUNT": 0,
            "BUREAU_ACTIVE_COUNT": 0,
            "BUREAU_TOTAL_OUTSTANDING": 0.0,
            "BUREAU_TOTAL_OVERDUE": 0.0,
            "BUREAU_MAX_OVERDUE_DAYS": 0,
            "BUREAU_LEGAL_ACTION_FLAG": 0,
        }

    n = len(bureau_facilities)
    active = sum(1 for f in bureau_facilities if f["status"] == "active")
    total_outstanding = float(sum(f["outstanding_amount"] for f in bureau_facilities))
    total_overdue = float(sum(f["overdue_amount"] for f in bureau_facilities))
    max_overdue_days = max((f["overdue_days"] for f in bureau_facilities), default=0)
    legal_flag = int(any(f["legal_action_flag"] for f in bureau_facilities))

    return {
        "BUREAU_FACILITY_COUNT": n,
        "BUREAU_ACTIVE_COUNT": active,
        "BUREAU_TOTAL_OUTSTANDING": total_outstanding,
        "BUREAU_TOTAL_OVERDUE": total_overdue,
        "BUREAU_MAX_OVERDUE_DAYS": max_overdue_days,
        "BUREAU_LEGAL_ACTION_FLAG": legal_flag,
    }