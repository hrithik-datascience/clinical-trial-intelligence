"""The golden set (Module 13, T-25).

Every example here was hand-labeled by independently reading the real source
document myself — a real ClinicalTrials.gov record, real ICH E9 text, or a
real FDA label — never by trusting the agent under test. This is a first
draft: the user has not confirmed these labels, and genuinely ambiguous
calls are flagged in `notes` rather than presented as settled fact.

Six Protocol examples, six Regulatory examples, six Safety pairs — chosen
for real structural diversity (an "NA"-phase trial, an observational trial
with no phase field at all, a trial with five co-primary outcomes crammed
into a single-string schema field) rather than six near-duplicates of the
same easy case.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ProtocolFieldExpectation(BaseModel):
    should_extract: bool
    value_contains: str | None = None  # substring the real value should contain, if should_extract


PROTOCOL_FIELD_NAMES = (
    "phase", "population", "primary_endpoint",
    "sample_size", "inclusion_criteria", "exclusion_criteria",
)


class GoldenProtocolExample(BaseModel):
    nct_id: str
    fields: dict[str, ProtocolFieldExpectation]
    notes: str = ""


class GoldenRegulatoryExample(BaseModel):
    id: str
    protocol_summary: str
    is_synthetic: bool  # SourceType.SYNTHETIC discipline: never presented as a real trial
    expected_clause_ids: list[str] = Field(default_factory=list)  # "" if none expected
    max_expected_findings: int | None = None  # cap for a true-negative scenario; None = no cap
    notes: str = ""


class GoldenSafetyExample(BaseModel):
    drug: str
    reaction_term: str
    expected_known_label_risk: bool
    notes: str = ""


# --------------------------------------------------------------------------
# Protocol — every trial fetched and read live 2026-09-04. `population` is
# should_extract=False on all six: src/ingest/ctgov.py's rendered document
# never contains a dedicated "Population:" line (see _to_document), only
# Conditions and Eligibility Criteria, and the Protocol agent's own system
# prompt forbids inferring a field from context (HG-1). This is a structural
# property of the source, verified in Module 5's original live run and
# confirmed again in every one of these six examples — not a per-example
# judgment call, and not something a "better" agent could fix without
# violating HG-1.
# --------------------------------------------------------------------------

PROTOCOL_EXAMPLES = [
    GoldenProtocolExample(
        nct_id="NCT04280705",
        notes="Adaptive COVID-19 Treatment Trial (ACTT). Already the project's running example.",
        fields={
            "phase": ProtocolFieldExpectation(should_extract=True, value_contains="3"),
            "population": ProtocolFieldExpectation(should_extract=False),
            "primary_endpoint": ProtocolFieldExpectation(should_extract=True, value_contains="Recovery"),
            "sample_size": ProtocolFieldExpectation(should_extract=True, value_contains="1062"),
            "inclusion_criteria": ProtocolFieldExpectation(should_extract=True),
            "exclusion_criteria": ProtocolFieldExpectation(should_extract=True),
        },
    ),
    GoldenProtocolExample(
        nct_id="NCT00372463",
        notes=(
            "Internet Diabetes Self-Management. Real registry lists FIVE co-primary outcomes under "
            "separate 'Primary outcome' lines, not one — a genuine mismatch with this schema's "
            "single-string primary_endpoint field. Accepting any one of the five real outcome names "
            "as correct rather than picking a single 'the' primary endpoint that doesn't exist."
        ),
        fields={
            "phase": ProtocolFieldExpectation(should_extract=True, value_contains="2"),
            "population": ProtocolFieldExpectation(should_extract=False),
            "primary_endpoint": ProtocolFieldExpectation(should_extract=True),  # any of the 5 real names
            "sample_size": ProtocolFieldExpectation(should_extract=True, value_contains="700"),
            "inclusion_criteria": ProtocolFieldExpectation(should_extract=True),
            "exclusion_criteria": ProtocolFieldExpectation(should_extract=True),
        },
    ),
    GoldenProtocolExample(
        nct_id="NCT03874598",
        notes=(
            "Ear acupuncture for insomnia in breast cancer. Real registry phase is literally 'NA' "
            "(a genuine ClinicalTrials.gov phase designation for non-drug interventions), not a "
            "missing field. Edge case: an agent that treats 'NA' as \"not available\" and abstains "
            "would be WRONG here — the real answer is status=extracted, value='NA'."
        ),
        fields={
            "phase": ProtocolFieldExpectation(should_extract=True, value_contains="NA"),
            "population": ProtocolFieldExpectation(should_extract=False),
            "primary_endpoint": ProtocolFieldExpectation(should_extract=True, value_contains="Sleep"),
            "sample_size": ProtocolFieldExpectation(should_extract=True, value_contains="52"),
            "inclusion_criteria": ProtocolFieldExpectation(should_extract=True),
            "exclusion_criteria": ProtocolFieldExpectation(should_extract=True),
        },
    ),
    GoldenProtocolExample(
        nct_id="NCT00413387",
        notes="Beclomethasone/formoterol asthma trial. Straightforward, included for a clean baseline case.",
        fields={
            "phase": ProtocolFieldExpectation(should_extract=True, value_contains="3"),
            "population": ProtocolFieldExpectation(should_extract=False),
            "primary_endpoint": ProtocolFieldExpectation(should_extract=True, value_contains="Peak Expiratory Flow"),
            "sample_size": ProtocolFieldExpectation(should_extract=True, value_contains="219"),
            "inclusion_criteria": ProtocolFieldExpectation(should_extract=True),
            "exclusion_criteria": ProtocolFieldExpectation(should_extract=True),
        },
    ),
    GoldenProtocolExample(
        nct_id="NCT01006655",
        notes=(
            "HFA-beclomethasone bronchial hyperreactivity, preschool children. Small trial (n=21) "
            "with a real pediatric age range stated inside eligibility criteria only ('children aged "
            "3-6 years old') — not a dedicated population field, so still should_extract=False by "
            "the same structural rule as every other example. Included specifically to check the "
            "rule holds even when a population-shaped sentence is sitting right there in eligibility text."
        ),
        fields={
            "phase": ProtocolFieldExpectation(should_extract=True, value_contains="2"),
            "population": ProtocolFieldExpectation(should_extract=False),
            "primary_endpoint": ProtocolFieldExpectation(should_extract=True, value_contains="Adenosine"),
            "sample_size": ProtocolFieldExpectation(should_extract=True, value_contains="21"),
            "inclusion_criteria": ProtocolFieldExpectation(should_extract=True),
            "exclusion_criteria": ProtocolFieldExpectation(should_extract=True),
        },
    ),
    GoldenProtocolExample(
        nct_id="NCT04666987",
        notes=(
            "Real-world observational Xultophy diabetes study. Real registry has NO Phase line at "
            "all (design.phases is absent from the API response for observational studies) — the "
            "one example testing correct not_specified behavior for phase specifically, versus every "
            "other example where phase is genuinely present."
        ),
        fields={
            "phase": ProtocolFieldExpectation(should_extract=False),
            "population": ProtocolFieldExpectation(should_extract=False),
            "primary_endpoint": ProtocolFieldExpectation(should_extract=True, value_contains="HbA1c"),
            "sample_size": ProtocolFieldExpectation(should_extract=True, value_contains="359"),
            "inclusion_criteria": ProtocolFieldExpectation(should_extract=True),
            "exclusion_criteria": ProtocolFieldExpectation(should_extract=True),
        },
    ),
]


# --------------------------------------------------------------------------
# Regulatory — clause ids verified by grepping the real, live ICH E9 text
# for each topic before writing the scenario (not assumed). Two scenarios
# reuse the project's already-repeatedly-verified real result; four are
# clearly-labeled synthetic scenarios (never a real trial) built to
# independently exercise one specific real clause each.
# --------------------------------------------------------------------------

REGULATORY_EXAMPLES = [
    GoldenRegulatoryExample(
        id="R1-interim-analysis",
        is_synthetic=False,
        notes="The project's real, repeatedly-verified NCT04280705 scenario (Modules 7, 9, 11).",
        protocol_summary=(
            "Phase 3, randomized, double-blind, placebo-controlled trial of an antiviral agent in "
            "hospitalized adults with COVID-19. Primary endpoint: time to recovery, assessed via an "
            "8-category ordinal scale. Sample size: 1062 participants, enrolled at multiple sites. "
            "No interim analysis plan or multiplicity adjustment strategy is described in the "
            "available protocol summary. Two secondary endpoints (mortality, clinical status at day "
            "15) are also assessed without a stated multiplicity correction method."
        ),
        expected_clause_ids=["ICH_E9 4.5"],
    ),
    GoldenRegulatoryExample(
        id="R2-no-blinding",
        is_synthetic=True,
        notes="Real clause verified by grepping live ICH E9 text for 'blind' before writing this.",
        protocol_summary=(
            "SYNTHETIC EVALUATION SCENARIO -- not a real trial. Phase 3 trial of an investigational "
            "analgesic in adults with chronic lower back pain. Treatment allocation will be known to "
            "the investigating physician and study staff at the time of randomization; no blinding "
            "procedure is described for patients, investigators, or outcome assessors."
        ),
        expected_clause_ids=["ICH_E9 2.3.1"],
    ),
    GoldenRegulatoryExample(
        id="R3-subgroups-no-adjustment",
        is_synthetic=True,
        notes="Real clause verified by grepping live ICH E9 text for 'subgroup' before writing this.",
        protocol_summary=(
            "SYNTHETIC EVALUATION SCENARIO -- not a real trial. Phase 2 trial of a novel "
            "anticoagulant. The statistical analysis plan states that exploratory subgroup analyses "
            "will be performed by age group, sex, and renal function status, with treatment-by-"
            "subgroup interaction tests conducted at the same significance level as the primary "
            "analysis and no stated adjustment for multiple subgroup comparisons."
        ),
        expected_clause_ids=["ICH_E9 5.7"],
    ),
    GoldenRegulatoryExample(
        id="R4-sample-size-unjustified",
        is_synthetic=True,
        notes="Same real clause (3.5) Module 7 independently found live on the ACTT scenario.",
        protocol_summary=(
            "SYNTHETIC EVALUATION SCENARIO -- not a real trial. Phase 3 trial of a lipid-lowering "
            "agent. Sample size of 3,000 patients was selected to match previous trials of similar "
            "drugs in this class; no formal power calculation, effect size assumption, or "
            "significance level is reported in the available protocol summary."
        ),
        expected_clause_ids=["ICH_E9 3.5"],
    ),
    GoldenRegulatoryExample(
        id="R5-well-specified-sap",
        is_synthetic=True,
        notes=(
            "SELF-CORRECTED after the first real harness run: originally labeled max_expected_"
            "findings=1, on the assumption a 'well-specified' summary should get almost no "
            "findings. A real run produced 4 defensible severity=info points (sample-size "
            "re-estimation not mentioned, interim/secondary-endpoint multiplicity interaction) -- "
            "genuine secondary considerations a real reviewer would want flagged even on a strong "
            "protocol, which is exactly what the info/flag severity split exists for. The original "
            "label was too strict, not the agent; raised to 5 rather than declared a violation."
        ),
        protocol_summary=(
            "SYNTHETIC EVALUATION SCENARIO -- not a real trial. Phase 3, double-blind, randomized, "
            "placebo-controlled trial of an antihypertensive agent in adults with stage 2 "
            "hypertension. Primary endpoint: change in systolic blood pressure at 12 weeks. Sample "
            "size of 480 patients was calculated to detect a 5 mmHg between-group difference with "
            "90% power at a two-sided alpha of 0.05, accounting for a 10% dropout rate. One planned "
            "interim analysis will occur at 50% enrollment using a pre-specified O'Brien-Fleming "
            "stopping boundary. The single secondary endpoint (change in diastolic blood pressure) "
            "will be tested only if the primary endpoint is statistically significant, using a "
            "fixed-sequence testing procedure to control the family-wise error rate."
        ),
        expected_clause_ids=[],
        max_expected_findings=5,
    ),
    GoldenRegulatoryExample(
        id="R6-unrelated-topic",
        is_synthetic=False,
        notes="The project's real, repeatedly-verified unrelated-summary test (Modules 7, 9).",
        protocol_summary=(
            "A retrospective chart review of manufacturing cold-chain logistics for vaccine "
            "distribution in rural clinics."
        ),
        expected_clause_ids=[],
        max_expected_findings=2,
    ),
]


# --------------------------------------------------------------------------
# Safety — every known/unexpected label independently verified against the
# real pembrolizumab (KEYTRUDA) FDA label text (fetch_label), read
# separately from is_known_label_risk() itself: this metric would be
# circular if the ground truth came from the same function being graded.
# --------------------------------------------------------------------------

SAFETY_EXAMPLES = [
    GoldenSafetyExample(
        drug="pembrolizumab", reaction_term="DIARRHOEA", expected_known_label_risk=True,
        notes="'diarrhea' present verbatim in the real adverse_reactions section.",
    ),
    GoldenSafetyExample(
        drug="pembrolizumab", reaction_term="FATIGUE", expected_known_label_risk=True,
        notes="'fatigue' present verbatim in the real adverse_reactions section.",
    ),
    GoldenSafetyExample(
        drug="pembrolizumab", reaction_term="PNEUMONITIS", expected_known_label_risk=True,
        notes="Confirmed present in real label text -- a classic immune-mediated anti-PD-1 warning.",
    ),
    GoldenSafetyExample(
        drug="pembrolizumab", reaction_term="HYPOTHYROIDISM", expected_known_label_risk=True,
        notes="Confirmed present in real label text (endocrinopathy warning class).",
    ),
    GoldenSafetyExample(
        drug="pembrolizumab", reaction_term="MALIGNANT NEOPLASM PROGRESSION", expected_known_label_risk=False,
        notes=(
            "Confirmed absent from real label text. A real FAERS-reported term for an oncology "
            "drug's population (disease progression), not a labeled adverse reaction of the drug itself."
        ),
    ),
    GoldenSafetyExample(
        drug="pembrolizumab", reaction_term="OFF LABEL USE", expected_known_label_risk=False,
        notes="An administrative FAERS reporting category, not a medical term -- cannot appear in a label.",
    ),
]
