"""
Pydantic request schemas for the two intake payloads.

These mirror sample_new_to_bank_payload.json and
sample_returning_customer_payload.json field-for-field. Shared sections
(documents, national_id_fields, salary_certificate_fields,
bank_statement_fields, iscore_report_fields, form_data, consistency_checks)
live in one place; the two top-level request models differ only in
`internal_history`, which is (correctly) much richer for returning
customers.
"""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ------------------------------------------------------------------
# Shared building blocks
# ------------------------------------------------------------------
class FieldWithConfidence(BaseModel):
    value: object
    confidence: float


class DocumentEntry(BaseModel):
    document_type: str
    overall_quality_score: float
    is_tampered_suspected: bool
    extracted_fields: dict


class NationalIDFields(BaseModel):
    full_name: FieldWithConfidence
    national_id: FieldWithConfidence
    date_of_birth: FieldWithConfidence
    age_years: FieldWithConfidence
    gender: FieldWithConfidence
    governorate: FieldWithConfidence
    address: FieldWithConfidence
    profession_on_id: FieldWithConfidence
    marital_status_on_id: FieldWithConfidence
    id_expiry_date: FieldWithConfidence


class SalaryCertificateFields(BaseModel):
    employer_name: FieldWithConfidence
    employer_sector: FieldWithConfidence
    job_title: FieldWithConfidence
    declared_net_salary: FieldWithConfidence
    declared_gross_salary: FieldWithConfidence
    employment_date: FieldWithConfidence
    employment_tenure_years: FieldWithConfidence
    issue_date: FieldWithConfidence


class BankStatementFields(BaseModel):
    bank_name: FieldWithConfidence
    account_number: FieldWithConfidence
    statement_period_months: FieldWithConfidence
    avg_monthly_net_inflow: FieldWithConfidence
    avg_monthly_balance: FieldWithConfidence
    min_monthly_balance: FieldWithConfidence
    max_monthly_balance: FieldWithConfidence
    balance_volatility_std: FieldWithConfidence
    overdraft_frequency: FieldWithConfidence
    returned_cheques_count: FieldWithConfidence
    income_regularity_score: FieldWithConfidence


class BureauFacility(BaseModel):
    """One entry in iscore_report_fields.bureau_facilities[] — only present
    for returning customers with active/past external bureau-reported credit.
    This is the raw data history_mapper.py aggregates into BUREAU_* features."""
    lender_name: str
    facility_type: str
    granted_amount: float
    outstanding_amount: float
    installment_amount: float
    status: str
    overdue_days: int
    overdue_amount: float
    start_date: str
    legal_action_flag: bool


class IScoreReportFields(BaseModel):
    is_available: bool
    credit_score: Optional[FieldWithConfidence] = None
    score_tier: Optional[FieldWithConfidence] = None
    score_date: Optional[FieldWithConfidence] = None
    total_active_loans_limit: Optional[FieldWithConfidence] = None
    total_outstanding_balance: Optional[FieldWithConfidence] = None
    total_overdue_amount: Optional[FieldWithConfidence] = None
    max_days_past_due: Optional[FieldWithConfidence] = None
    active_credit_cards_count: Optional[FieldWithConfidence] = None
    total_credit_card_utilization: Optional[FieldWithConfidence] = None
    bureau_facilities: list[BureauFacility] = Field(default_factory=list)


class FormData(BaseModel):
    requested_amount: float
    tenure_months: int
    requested_annuity: float
    loan_purpose: str
    goods_price: float
    family_status: str
    children_count: int
    family_members_count: int
    housing_type: str
    education_type: str
    owns_car: bool
    owns_realty: bool
    branch_id: int


class ConsistencyChecks(BaseModel):
    income_mismatch_ratio: float
    employer_name_match: bool
    employer_match_similarity_score: float
    national_id_match_across_documents: bool
    iscore_report_age_days: int
    document_tampering_flag: bool
    running_balance_math_valid: bool
    fraud_risk_level: str
    triggered_fraud_rules: list[str] = Field(default_factory=list)


# ------------------------------------------------------------------
# New-to-bank: internal_history is present but structurally empty
# (internal_history_missing=1, no accounts/loans, aggregated_metrics all 0)
# ------------------------------------------------------------------
class NewCustomerAggregatedMetrics(BaseModel):
    prev_app_count: int = 0
    prev_approved_count: int = 0
    prev_refused_count: int = 0
    prev_approved_ratio: float = 0.0
    prev_avg_credit: float = 0.0
    inst_payment_count: int = 0
    inst_late_count: int = 0
    inst_severe_late_count: int = 0
    inst_late_ratio: float = 0.0
    inst_avg_days_late: float = 0.0
    inst_max_days_late: int = 0
    pos_record_count: int = 0
    pos_avg_dpd: float = 0.0
    cc_avg_balance: float = 0.0
    cc_balance_to_limit_ratio: float = 0.0


class NewCustomerInternalHistory(BaseModel):
    internal_history_missing: int = 1
    customer_id: Optional[str] = None
    kyc_status: str = "not_applicable"
    customer_since_date: Optional[str] = None
    relationship_tenure_months: int = 0
    bank_accounts: list = Field(default_factory=list)
    previous_bank_loans: list = Field(default_factory=list)
    aggregated_metrics: NewCustomerAggregatedMetrics = Field(default_factory=NewCustomerAggregatedMetrics)


class NewCustomerApplicationPayload(BaseModel):
    application_id: str
    applicant_type: str
    submission_timestamp: datetime
    is_returning_customer: bool = False
    documents: list[DocumentEntry]
    national_id_fields: NationalIDFields
    salary_certificate_fields: SalaryCertificateFields
    bank_statement_fields: BankStatementFields
    iscore_report_fields: IScoreReportFields
    form_data: FormData
    internal_history: NewCustomerInternalHistory
    consistency_checks: ConsistencyChecks
    excluded_features_note: Optional[str] = None


# ------------------------------------------------------------------
# Returning customer: internal_history carries real per-account/per-loan
# records that history_mapper.py aggregates into PREV_*/INST_*/POS_* features.
# ------------------------------------------------------------------
class BankAccount(BaseModel):
    account_id: str
    account_type: str
    currency: str
    current_balance: float
    open_date: str
    status: str


class PreviousBankLoan(BaseModel):
    """One of the applicant's own past loans with this bank. Raw enough to
    reconstruct INST_LATE_RATIO / INST_SEVERE_LATE_RATIO / PREV_* the same
    way installments.py / previous_app.py do, just for a single applicant."""
    loan_id: str
    loan_type: str
    principal_amount: float
    interest_rate: float
    tenure_months: int
    status: str
    start_date: str
    installments_count: int
    paid_on_time_count: int
    late_payments_count: int
    severe_late_count: int
    max_days_past_due: int


class ReturningCustomerAggregatedMetrics(NewCustomerAggregatedMetrics):
    """Same shape as the new-customer version, but populated. Used as a
    fallback / cross-check against history_mapper.py's reconstruction from
    previous_bank_loans, not as the primary feature source (see history_mapper.py)."""
    pass


class ReturningCustomerInternalHistory(BaseModel):
    internal_history_missing: int = 0
    customer_id: str
    kyc_status: str
    customer_since_date: str
    relationship_tenure_months: int
    bank_accounts: list[BankAccount] = Field(default_factory=list)
    previous_bank_loans: list[PreviousBankLoan] = Field(default_factory=list)
    aggregated_metrics: ReturningCustomerAggregatedMetrics


class ReturningCustomerApplicationPayload(BaseModel):
    application_id: str
    applicant_type: str
    submission_timestamp: datetime
    is_returning_customer: bool = True
    documents: list[DocumentEntry]
    national_id_fields: NationalIDFields
    salary_certificate_fields: SalaryCertificateFields
    bank_statement_fields: BankStatementFields
    iscore_report_fields: IScoreReportFields
    form_data: FormData
    internal_history: ReturningCustomerInternalHistory
    consistency_checks: ConsistencyChecks
    excluded_features_note: Optional[str] = None


# ------------------------------------------------------------------
# Shared response schema (both endpoints return the same shape)
# ------------------------------------------------------------------
class CreditDecisionResponse(BaseModel):
    application_id: str
    credit_score: int
    default_probability: float
    decision: str            # AUTO-APPROVE / MANUAL REVIEW / AUTO-REJECT
    risk_tier: str
    reason_codes: list[str]
    model_version: str