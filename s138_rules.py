"""Deterministic checks for a Section 138 NI Act cheque-dishonour matter.

Nothing in this module involves a language model.  Limitation periods and the
statutory ingredients of the offence are arithmetic and rule application, and
arithmetic must not be delegated to something that can hallucinate.

Statutory basis (Negotiable Instruments Act 1881)
-------------------------------------------------
s.138 main clause -- the cheque must have been drawn on the drawer's own
account, for the discharge of a legally enforceable debt or liability, and
returned unpaid for insufficiency of funds or because it exceeds the
arrangement with the bank.

s.138 provisos, all three of which must be satisfied:
  (a) the cheque is presented to the bank within its validity period;
  (b) the payee makes a written demand for payment within 30 days of receiving
      information from the bank about the dishonour;
  (c) the drawer fails to pay within 15 days of receiving that notice.

s.142(1)(b) -- the complaint must be filed within one month of the date on which
the cause of action arises under proviso (c).  Under the proviso to s.142(1)(b)
a court may take cognizance later if the complainant shows sufficient cause.

Cheque validity is three months from the date the cheque bears, following the
Reserve Bank of India's reduction from six months with effect from 1 April 2012.

These rules are applied here as written.  They are not a substitute for advice
from an advocate on the facts of a particular case.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

CHEQUE_VALIDITY_DAYS = 90          # three months, RBI, w.e.f. 01-04-2012
NOTICE_WINDOW_DAYS = 30            # s.138 proviso (b)
PAYMENT_WINDOW_DAYS = 15           # s.138 proviso (c)
COMPLAINT_WINDOW_DAYS = 30         # s.142(1)(b), "within one month"


@dataclass
class CaseFacts:
    """The dates and facts a s.138 assessment turns on."""

    cheque_date: date | None = None
    presented_date: date | None = None
    dishonour_intimation_date: date | None = None
    notice_sent_date: date | None = None
    notice_served_date: date | None = None
    payment_received: bool = False
    complaint_filed_date: date | None = None
    drawn_on_own_account: bool | None = None
    legally_enforceable_debt: bool | None = None
    returned_for_funds: bool | None = None


@dataclass
class Finding:
    """One checked requirement, with the provision it comes from."""

    label: str
    status: str          # "met" | "not_met" | "at_risk" | "unknown"
    detail: str
    provision: str


@dataclass
class Assessment:
    findings: list[Finding] = field(default_factory=list)
    deadlines: dict[str, date] = field(default_factory=dict)
    blocking_problems: list[str] = field(default_factory=list)

    @property
    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for finding in self.findings:
            counts[finding.status] = counts.get(finding.status, 0) + 1
        return counts

    @property
    def has_blocking_problem(self) -> bool:
        return bool(self.blocking_problems)


def _add(assessment: Assessment, label: str, status: str, detail: str, provision: str) -> None:
    assessment.findings.append(Finding(label, status, detail, provision))


def assess(facts: CaseFacts, today: date | None = None) -> Assessment:
    """Apply the s.138 ingredients and limitation periods to the given facts."""
    today = today or date.today()
    assessment = Assessment()

    # --- Ingredients of the offence (s.138 main clause) --------------------
    for label, value, provision, detail_yes, detail_no in [
        ("Cheque drawn on the drawer's own account", facts.drawn_on_own_account, "s.138 main clause",
         "The cheque was drawn on the drawer's own account.",
         "s.138 applies only to a cheque drawn on the drawer's own account with a banker."),
        ("For a legally enforceable debt or liability", facts.legally_enforceable_debt, "s.138 main clause",
         "The cheque was given for a legally enforceable debt or liability.",
         "A cheque given as a gift, security, or for a time-barred debt may fall outside s.138."),
        ("Returned unpaid for insufficiency of funds", facts.returned_for_funds, "s.138 main clause",
         "The cheque was returned unpaid for insufficiency of funds or exceeding arrangement.",
         "Return for another reason may not attract s.138 -- the bank's return memo decides this."),
    ]:
        if value is True:
            _add(assessment, label, "met", detail_yes, provision)
        elif value is False:
            _add(assessment, label, "not_met", detail_no, provision)
            assessment.blocking_problems.append(f"{label}: {detail_no}")
        else:
            _add(assessment, label, "unknown", "Not stated.", provision)

    # --- Proviso (a): presentation within validity -------------------------
    if facts.cheque_date and facts.presented_date:
        expiry = facts.cheque_date + timedelta(days=CHEQUE_VALIDITY_DAYS)
        assessment.deadlines["Cheque valid until"] = expiry
        days = (facts.presented_date - facts.cheque_date).days
        if facts.presented_date <= expiry:
            _add(assessment, "Presented within validity", "met",
                 f"Presented {days} days after the cheque date, within the {CHEQUE_VALIDITY_DAYS}-day validity.",
                 "s.138 proviso (a)")
        else:
            detail = (f"Presented {days} days after the cheque date, beyond the "
                      f"{CHEQUE_VALIDITY_DAYS}-day validity that expired on {expiry:%d-%m-%Y}.")
            _add(assessment, "Presented within validity", "not_met", detail, "s.138 proviso (a)")
            assessment.blocking_problems.append(f"Presented within validity: {detail}")
    else:
        _add(assessment, "Presented within validity", "unknown",
             "Needs both the cheque date and the presentation date.", "s.138 proviso (a)")

    # --- Proviso (b): demand notice within 30 days -------------------------
    if facts.dishonour_intimation_date:
        deadline = facts.dishonour_intimation_date + timedelta(days=NOTICE_WINDOW_DAYS)
        assessment.deadlines["Demand notice due by"] = deadline
        if facts.notice_sent_date:
            days = (facts.notice_sent_date - facts.dishonour_intimation_date).days
            if facts.notice_sent_date <= deadline:
                _add(assessment, "Demand notice sent in time", "met",
                     f"Notice sent {days} days after intimation of dishonour, within {NOTICE_WINDOW_DAYS} days.",
                     "s.138 proviso (b)")
            else:
                detail = (f"Notice sent {days} days after intimation of dishonour. The "
                          f"{NOTICE_WINDOW_DAYS}-day window closed on {deadline:%d-%m-%Y}.")
                _add(assessment, "Demand notice sent in time", "not_met", detail, "s.138 proviso (b)")
                assessment.blocking_problems.append(f"Demand notice sent in time: {detail}")
        elif today > deadline:
            detail = f"No notice recorded and the {NOTICE_WINDOW_DAYS}-day window closed on {deadline:%d-%m-%Y}."
            _add(assessment, "Demand notice sent in time", "not_met", detail, "s.138 proviso (b)")
            assessment.blocking_problems.append(f"Demand notice sent in time: {detail}")
        else:
            _add(assessment, "Demand notice sent in time", "at_risk",
                 f"No notice recorded yet. It must be sent by {deadline:%d-%m-%Y} "
                 f"({(deadline - today).days} days left).", "s.138 proviso (b)")
    else:
        _add(assessment, "Demand notice sent in time", "unknown",
             "Needs the date the bank informed the payee of the dishonour.", "s.138 proviso (b)")

    # --- Proviso (c) and s.142: payment window, then the complaint ---------
    service_date = facts.notice_served_date or facts.notice_sent_date
    if service_date:
        pay_by = service_date + timedelta(days=PAYMENT_WINDOW_DAYS)
        assessment.deadlines["Drawer must pay by"] = pay_by
        # The cause of action arises the day the 15-day payment window expires.
        cause_of_action = pay_by + timedelta(days=1)
        complaint_deadline = cause_of_action + timedelta(days=COMPLAINT_WINDOW_DAYS)
        assessment.deadlines["Cause of action arises"] = cause_of_action
        assessment.deadlines["Complaint must be filed by"] = complaint_deadline

        if facts.payment_received:
            _add(assessment, "Drawer failed to pay in 15 days", "not_met",
                 "Payment was received. If the full cheque amount was paid within 15 days of "
                 "the notice, no offence under s.138 is made out.", "s.138 proviso (c)")
            assessment.blocking_problems.append(
                "Payment was made within the statutory period, which extinguishes the s.138 offence.")
        elif today <= pay_by:
            _add(assessment, "Drawer failed to pay in 15 days", "at_risk",
                 f"The drawer still has until {pay_by:%d-%m-%Y} to pay. A complaint filed before "
                 f"that date is premature.", "s.138 proviso (c)")
        else:
            _add(assessment, "Drawer failed to pay in 15 days", "met",
                 f"The {PAYMENT_WINDOW_DAYS}-day period expired on {pay_by:%d-%m-%Y} without payment.",
                 "s.138 proviso (c)")

        if facts.complaint_filed_date:
            if facts.complaint_filed_date < cause_of_action:
                detail = (f"Complaint filed on {facts.complaint_filed_date:%d-%m-%Y}, before the cause of "
                          f"action arose on {cause_of_action:%d-%m-%Y}. A premature complaint is liable to "
                          f"be dismissed.")
                _add(assessment, "Complaint filed in time", "not_met", detail, "s.142(1)(b)")
                assessment.blocking_problems.append(f"Complaint filed in time: {detail}")
            elif facts.complaint_filed_date <= complaint_deadline:
                _add(assessment, "Complaint filed in time", "met",
                     f"Filed on {facts.complaint_filed_date:%d-%m-%Y}, within the one-month window "
                     f"closing {complaint_deadline:%d-%m-%Y}.", "s.142(1)(b)")
            else:
                detail = (f"Filed on {facts.complaint_filed_date:%d-%m-%Y}, after the window closed on "
                          f"{complaint_deadline:%d-%m-%Y}. The court may still take cognizance if "
                          f"sufficient cause for the delay is shown.")
                _add(assessment, "Complaint filed in time", "at_risk", detail,
                     "s.142(1)(b) proviso")
        elif today > complaint_deadline:
            detail = (f"No complaint recorded and the one-month window closed on "
                      f"{complaint_deadline:%d-%m-%Y}. Filing now requires the court to be satisfied "
                      f"there was sufficient cause for the delay.")
            _add(assessment, "Complaint filed in time", "at_risk", detail, "s.142(1)(b) proviso")
        else:
            _add(assessment, "Complaint filed in time", "at_risk",
                 f"Not yet filed. The window runs from {cause_of_action:%d-%m-%Y} to "
                 f"{complaint_deadline:%d-%m-%Y} ({(complaint_deadline - today).days} days left).",
                 "s.142(1)(b)")
    else:
        _add(assessment, "Drawer failed to pay in 15 days", "unknown",
             "Needs the date the demand notice was sent or served.", "s.138 proviso (c)")
        _add(assessment, "Complaint filed in time", "unknown",
             "Cannot be computed until the notice date is known.", "s.142(1)(b)")

    return assessment


def search_terms_for(facts: CaseFacts, assessment: Assessment) -> list[str]:
    """Suggest retrieval queries targeted at this case's weak points.

    Precedent is most useful where a requirement is disputed or missed, so the
    queries are built from the findings rather than from generic keywords.
    """
    queries: list[str] = []
    by_label = {finding.label: finding for finding in assessment.findings}

    if by_label.get("Presented within validity", Finding("", "", "", "")).status == "not_met":
        queries.append("cheque presented after expiry of validity period section 138")
    if by_label.get("Demand notice sent in time", Finding("", "", "", "")).status == "not_met":
        queries.append("demand notice beyond thirty days section 138 limitation")
    if by_label.get("Complaint filed in time", Finding("", "", "", "")).status in {"not_met", "at_risk"}:
        queries.append("delay in filing complaint section 142 sufficient cause condonation")
    if facts.legally_enforceable_debt is False:
        queries.append("cheque not for legally enforceable debt rebuttal presumption section 139")
    if facts.drawn_on_own_account is False:
        queries.append("cheque drawn on account of another person section 138 liability")

    if not queries:
        queries = [
            "presumption under section 139 rebuttal by accused",
            "essential ingredients of offence under section 138",
        ]
    return queries
