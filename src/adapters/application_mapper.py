"""
Maps either intake payload (NewCustomerApplicationPayload or
ReturningCustomerApplicationPayload) into a single-row dict shaped like
a Kaggle application_train.csv row, so it can be fed unchanged into
features/application.py's engineer_application_features(fit=False, ...).

Only covers the application-level block. History features (PREV_*,
INST_*, POS_*, BUREAU_*, CC_*) are handled separately by history_mapper.py
for returning customers, or left as "no history" (matching training data's
own no-bureau-history applicants) for new customers.

Columns with NO source in either payload (OCCUPATION_TYPE, ORGANIZATION_TYPE,
REGION_RATING_CLIENT, FLAG_DOCUMENT_*, etc.) are deliberately left out of
the returned dict. engineer_application_features()'s existing fit-time
median / "Missing" imputation handles anything absent — that logic doesn't
change here, this adapter's only job is to populate what the payload
actually gives us.
"""
from typing import Union

from src import config

# Kaggle NAME_FAMILY_STATUS categories the OneHotEncoder was fit on.
_FAMILY_STATUS_MAP = {
    "single": "Single / not married",
    "married": "Married",
    "civil_marriage": "Civil marriage",
    "widow": "Widow",
    "separated": "Separated",
}

_HOUSING_TYPE_MAP = {
    "owned": "House / apartment",
    "rented": "Rented apartment",
    "with_parents": "With parents",
    "municipal": "Municipal apartment",
    "office": "Office apartment",
    "co_op": "Co-op apartment",
}

_EDUCATION_TYPE_MAP = {
    "secondary": "Secondary / secondary special",
    "higher_education": "Higher education",
    "incomplete_higher": "Incomplete higher",
    "lower_secondary": "Lower secondary",
    "academic_degree": "Academic degree",
}

# form_data.loan_purpose -> Kaggle's NAME_CONTRACT_TYPE. Extend this map as
# new loan_purpose values show up; anything unmapped defaults to "Cash loans".
_CONTRACT_TYPE_MAP = {
    "personal_cash": "Cash loans",
    "revolving": "Revolving loans",
}


def _normalize_iscore(credit_score: float) -> float:
    """ASSUMPTION 2 in config.py — see that docstring before trusting this."""
    clipped = max(config.ISCORE_MIN, min(config.ISCORE_MAX, credit_score))
    return (clipped - config.ISCORE_MIN) / (config.ISCORE_MAX - config.ISCORE_MIN)


def map_payload_to_application_row(payload: Union[dict, object]) -> dict:
    """
    Parameters
    ----------
    payload : a NewCustomerApplicationPayload or ReturningCustomerApplicationPayload
        instance (or its .model_dump() dict) — both have identical shape for
        every field this function reads.

    Returns
    -------
    dict of {kaggle_column_name: value}, ready to become a one-row DataFrame
    and passed into engineer_application_features(df, fit=False, fit_artifacts=...).
    """
    p = payload if isinstance(payload, dict) else payload.model_dump()

    national_id = p["national_id_fields"]
    salary = p["salary_certificate_fields"]
    form = p["form_data"]
    iscore = p["iscore_report_fields"]

    age_years = national_id["age_years"]["value"]
    employed_years = salary["employment_tenure_years"]["value"]
    monthly_salary = salary["declared_net_salary"]["value"]

    row = {
        # Back-derived so engineer_application_features's existing
        # DAYS_BIRTH / DAYS_EMPLOYED math runs completely unchanged.
        "DAYS_BIRTH": -round(age_years * 365.25),
        "DAYS_EMPLOYED": -round(employed_years * 365.25),

        "CODE_GENDER": national_id["gender"]["value"],  # already 'M'/'F', matches Kaggle directly

        # ASSUMPTION 1 in config.py — monthly salary annualized to match
        # Kaggle's AMT_INCOME_TOTAL scale.
        "AMT_INCOME_TOTAL": monthly_salary * config.INCOME_ANNUALIZATION_MULTIPLIER,
        "AMT_CREDIT": form["requested_amount"],
        "AMT_ANNUITY": form["requested_annuity"],
        "AMT_GOODS_PRICE": form["goods_price"] or form["requested_amount"],

        "CNT_CHILDREN": form["children_count"],
        "CNT_FAM_MEMBERS": form["family_members_count"],
        "NAME_FAMILY_STATUS": _FAMILY_STATUS_MAP.get(form["family_status"], "Unknown"),
        "NAME_HOUSING_TYPE": _HOUSING_TYPE_MAP.get(form["housing_type"], "House / apartment"),
        "NAME_EDUCATION_TYPE": _EDUCATION_TYPE_MAP.get(form["education_type"], "Secondary / secondary special"),
        "NAME_CONTRACT_TYPE": _CONTRACT_TYPE_MAP.get(form["loan_purpose"], "Cash loans"),
        "FLAG_OWN_CAR": "Y" if form["owns_car"] else "N",
        "FLAG_OWN_REALTY": "Y" if form["owns_realty"] else "N",
    }

    # EXT_SOURCE_1/2/3: proxy from I-Score if available, else np.nan.
    # IMPORTANT: these keys must always be present (even as NaN) — application.py's
    # _secondary_features() does df[[EXT_SOURCE_1, _2, _3]].mean(axis=1), which
    # raises KeyError on missing COLUMNS (as opposed to missing VALUES). Setting
    # NaN here lets the existing fit-time median imputation handle "no I-Score"
    # the same way Kaggle's own missing EXT_SOURCE values are already handled.
    if iscore.get("is_available") and iscore.get("credit_score"):
        proxied = _normalize_iscore(iscore["credit_score"]["value"])
    else:
        proxied = float("nan")
    row["EXT_SOURCE_1"] = proxied
    row["EXT_SOURCE_2"] = proxied
    row["EXT_SOURCE_3"] = proxied

    # OBS_30/60_CNT_SOCIAL_CIRCLE, DEF_30/60_CNT_SOCIAL_CIRCLE: deliberately
    # NOT set. Per your own payload note these are "permanently dropped" —
    # application.py's _secondary_features() already does
    # df.get("OBS_30_CNT_SOCIAL_CIRCLE", 0), so omitting them here correctly
    # yields 0 without any special-case code.

    return row