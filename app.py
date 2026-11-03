from pathlib import Path
import json
import pandas as pd
import streamlit as st

from reconciliation import reconcile
from exception_classifier import (
    LABELS, LABEL_DESCRIPTIONS, train_exception_classifier, classify_with_context,
)

ROOT = Path(__file__).parent

st.set_page_config(
    page_title="ReconIQ | Evidence-first trade reconciliation",
    page_icon="🔎",
    layout="wide",
)

st.title("ReconIQ")
st.caption("Evidence-first trade reconciliation with answerability-aware extraction and multi-label exception screening")
st.info(
    "Educational prototype using synthetic data. It does not connect to brokers or custodians, "
    "settle trades, post journal entries or replace an operations analyst."
)


@st.cache_resource(show_spinner="Training the synthetic exception-screening model...")
def load_exception_model():
    return train_exception_classifier(ROOT / "data" / "exception_training.csv")


@st.cache_data(show_spinner=False)
def load_synthetic_samples():
    internal = pd.read_csv(ROOT / "data" / "internal_trades.csv", keep_default_na=False)
    external = pd.read_csv(ROOT / "data" / "external_trades.csv", keep_default_na=False)
    return internal, external


with st.sidebar:
    st.header("Data inputs")
    st.caption("Upload both files to use a custom synthetic / approved public dataset. Otherwise, the bundled demo is used.")
    internal_file = st.file_uploader("Internal trade export (CSV)", type=["csv"], key="internal_upload")
    external_file = st.file_uploader("External broker / custodian export (CSV)", type=["csv"], key="external_upload")
    st.divider()
    st.caption("The exception model is trained on the bundled synthetic annotation examples. Its suggestions do not change deterministic reconciliation results.")

if bool(internal_file) != bool(external_file):
    st.error("Upload both the internal and external CSV files. ReconIQ will not mix one uploaded file with a bundled sample file.")
    st.stop()
elif internal_file and external_file:
    try:
        internal_df = pd.read_csv(internal_file, keep_default_na=False)
        external_df = pd.read_csv(external_file, keep_default_na=False)
    except Exception as exc:
        st.error(f"Could not read one of the uploaded CSV files: {exc}")
        st.stop()
else:
    internal_df, external_df = load_synthetic_samples()
    st.caption("Showing bundled synthetic example trades.")

with st.expander("Preview source files", expanded=False):
    left, right = st.columns(2)
    with left:
        st.markdown("**Internal source**")
        st.dataframe(internal_df.head(12), use_container_width=True, hide_index=True)
    with right:
        st.markdown("**External source**")
        st.dataframe(external_df.head(12), use_container_width=True, hide_index=True)

try:
    result = reconcile(internal_df, external_df)
except Exception as exc:
    st.error(f"Reconciliation could not run: {exc}")
    st.stop()

matches = result["matches"].copy()
evidence = result["evidence"].copy()
profile = result["profile"].copy()
candidates = result["candidates"].copy()
duplicates = result["duplicates"].copy()
unmatched_i = result["unmatched_internal"].copy()
unmatched_e = result["unmatched_external"].copy()

try:
    exception_model = load_exception_model()
except Exception as exc:
    st.error(f"The synthetic exception classifier could not be loaded: {exc}")
    st.stop()

# Sentence-level exception screening for each external confirmation note.
screening_frames = []
if "confirmation_text" in external_df.columns:
    for row in external_df.itertuples(index=False):
        note = getattr(row, "confirmation_text", "")
        if not isinstance(note, str) or not note.strip():
            continue
        segments = classify_with_context(exception_model, note, context_radius=1)
        if not segments.empty:
            segments.insert(0, "external_trade_id", str(getattr(row, "trade_id", "")))
            screening_frames.append(segments)

if screening_frames:
    screening = pd.concat(screening_frames, ignore_index=True)
else:
    screening = pd.DataFrame(columns=[
        "external_trade_id", "sentence", "start_offset", "end_offset", "context_text",
        "exception_detected", "detection_margin", "suggested_labels", "category_suggestions"
    ])

# Build a cross-check between deterministic rule flags and ML suggestions.
rule_by_external = {}
if not matches.empty:
    for trade_id, group in matches.groupby("external_trade_id"):
        labels = set()
        for raw in group.get("rule_exception_labels", pd.Series(dtype=str)).fillna(""):
            labels.update(x for x in str(raw).split("|") if x and x in LABELS)
        rule_by_external[str(trade_id)] = labels

ml_by_external = {}
if not screening.empty:
    for trade_id, group in screening.groupby("external_trade_id"):
        labels = set()
        for raw in group["suggested_labels"].fillna(""):
            labels.update(x for x in str(raw).split("|") if x and x in LABELS)
        ml_by_external[str(trade_id)] = labels

cross_rows = []
for trade_id in sorted(set(rule_by_external) | set(ml_by_external)):
    rules = rule_by_external.get(trade_id, set())
    ml = ml_by_external.get(trade_id, set())
    cross_rows.append({
        "external_trade_id": trade_id,
        "deterministic_rule_flags": "|".join(sorted(rules)),
        "ML_screening_suggestions": "|".join(sorted(ml)),
        "shared_labels": "|".join(sorted(rules & ml)),
        "ML_only_suggestions": "|".join(sorted(ml - rules)),
        "rule_only_flags": "|".join(sorted(rules - ml)),
    })
cross_check = pd.DataFrame(cross_rows)

exact_count = int(matches["status"].eq("MATCHED - EXACT").sum()) if not matches.empty else 0
review_count = int(matches["status"].str.startswith("REVIEW").sum()) if not matches.empty else 0
unmatched_count = len(unmatched_i) + len(unmatched_e)
exception_notes = int(screening.loc[screening["exception_detected"].eq(True), "external_trade_id"].nunique()) if not screening.empty else 0

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Internal trades", len(internal_df))
m2.metric("External trades", len(external_df))
m3.metric("Exact matches", exact_count)
m4.metric("Needs review", review_count)
m5.metric("Unmatched rows", unmatched_count)
st.caption(f"The text-screening stage flagged {exception_notes} external note(s) for review. This is a separate signal, not a reconciliation decision.")

tabs = st.tabs([
    "Reconciliation",
    "Evidence inspector",
    "Exception screening",
    "Rule / ML cross-check",
    "Data quality",
    "Research method",
])

with tabs[0]:
    st.subheader("Deterministic reconciliation work queue")
    st.write(
        "Exact matching is controlled by security identity and the R1–R6 evidence map. "
        "The text classifier is not allowed to override these statuses."
    )
    if matches.empty:
        st.warning("No candidate pairs were generated. Inspect the source schema, security identifiers and descriptions.")
    else:
        display_cols = [
            "internal_trade_id", "external_trade_id", "status", "rank_score",
            "R1_IDENTITY", "R2_TRADE_DATE", "R3_SIDE", "R4_QUANTITY",
            "R5_PRICE_CURRENCY", "R6_SETTLEMENT_ALLOCATION", "reason",
            "rule_exception_labels", "human_decision",
        ]
        display_cols = [c for c in display_cols if c in matches.columns]
        if "human_decision" not in matches.columns:
            matches["human_decision"] = "Pending"
        reviewed = st.data_editor(
            matches[display_cols],
            use_container_width=True,
            hide_index=True,
            disabled=[c for c in display_cols if c != "human_decision"],
            column_config={
                "human_decision": st.column_config.SelectboxColumn(
                    "Analyst decision", options=["Pending", "Approve", "Reject", "Investigate"], required=True
                ),
                "rank_score": st.column_config.NumberColumn("Candidate rank (not probability)", format="%.3f"),
            },
            key="review_decisions",
        )
        st.download_button(
            "Download reviewed reconciliation CSV",
            reviewed.to_csv(index=False).encode("utf-8"),
            file_name="reconiq_reviewed_reconciliation.csv",
            mime="text/csv",
        )
    a, b = st.columns(2)
    with a:
        st.markdown("**Unmatched internal records**")
        st.dataframe(unmatched_i, use_container_width=True, hide_index=True)
    with b:
        st.markdown("**Unmatched external records**")
        st.dataframe(unmatched_e, use_container_width=True, hide_index=True)
    with st.expander("Duplicate business-key groups"):
        st.dataframe(duplicates, use_container_width=True, hide_index=True)

with tabs[1]:
    st.subheader("Answerability-aware evidence inspector")
    st.write(
        "Each note field is classified as ANSWERABLE, NOT_FOUND or AMBIGUOUS. "
        "An answerable value retains an exact source quote and character offsets. "
        "NOT_FOUND is not the same as zero or false."
    )
    if evidence.empty:
        st.warning("No confirmation-note evidence is available. Add a `confirmation_text` column to the external CSV.")
    else:
        trade_ids = sorted(evidence["external_trade_id"].dropna().astype(str).unique().tolist())
        selected = st.selectbox("External trade", trade_ids, key="evidence_trade")
        selected_rows = evidence[evidence["external_trade_id"].astype(str) == selected].copy()
        note = ""
        if "trade_id" in external_df.columns and "confirmation_text" in external_df.columns:
            notes = external_df.loc[external_df["trade_id"].astype(str) == selected, "confirmation_text"]
            if not notes.empty:
                note = str(notes.iloc[0])
        st.markdown("**Original confirmation note**")
        st.code(note or "No note supplied.", language="text")
        fields = ["field", "answerability", "value", "evidence_quote", "start_offset", "end_offset", "method", "note"]
        fields = [x for x in fields if x in selected_rows.columns]
        st.dataframe(selected_rows[fields], use_container_width=True, hide_index=True)
        st.download_button(
            "Download evidence ledger rows",
            selected_rows.to_csv(index=False).encode("utf-8"),
            file_name=f"reconiq_evidence_{selected}.csv",
            mime="text/csv",
        )

with tabs[2]:
    st.subheader("CLAUDETTE-inspired exception detection and categorisation")
    st.caption("The SVM model is trained only on the bundled synthetic examples. It is an analyst-assist suggestion, not a probability, not a match decision and not production validated.")
    if screening.empty:
        st.warning("No external confirmation notes are available for text screening.")
    else:
        available = sorted(screening["external_trade_id"].astype(str).unique().tolist())
        selected = st.selectbox("Confirmation note", available, key="classifier_trade")
        note_rows = screening[screening["external_trade_id"].astype(str) == selected].copy()
        note = ""
        if "confirmation_text" in external_df.columns:
            vals = external_df.loc[external_df["trade_id"].astype(str) == selected, "confirmation_text"]
            if not vals.empty:
                note = str(vals.iloc[0])
        st.markdown("**Source note**")
        st.code(note, language="text")
        st.markdown("**Sentence-level screening results**")
        st.caption("Raw SVM detector output is shown separately from the narrow post-processing guard. The margin is an SVM decision score, not a calibrated probability.")
        screening_cols = [
            "sentence", "start_offset", "end_offset", "context_text",
            "raw_detector_flag", "exception_detected", "detection_margin",
            "postprocess_suppressed_labels", "suggested_labels",
        ]
        screening_cols = [c for c in screening_cols if c in note_rows.columns]
        st.dataframe(note_rows[screening_cols], use_container_width=True, hide_index=True)
        st.markdown("**Suggested exception taxonomy**")
        label_details = []
        for label in LABELS:
            if note_rows["suggested_labels"].fillna("").str.contains(label, regex=False).any():
                label_details.append({"category": label, "meaning": LABEL_DESCRIPTIONS[label]})
        if label_details:
            st.dataframe(pd.DataFrame(label_details), use_container_width=True, hide_index=True)
        else:
            st.success("No exception category was suggested for this note. This is not proof that the trade is correct.")
        st.download_button(
            "Download sentence screening results",
            note_rows.to_csv(index=False).encode("utf-8"),
            file_name=f"reconiq_exception_screening_{selected}.csv",
            mime="text/csv",
        )

with tabs[3]:
    st.subheader("Compare rule flags with model suggestions")
    st.write(
        "This view exposes disagreements between structured-field rules and narrative exception screening. "
        "A model-only suggestion can be a false positive; a rule-only flag can indicate that the note does not mention the issue. "
        "Neither system is allowed to silently override the other."
    )
    if cross_check.empty:
        st.info("No candidate-level comparison is available.")
    else:
        st.dataframe(cross_check, use_container_width=True, hide_index=True)
        st.download_button(
            "Download rule / ML cross-check",
            cross_check.to_csv(index=False).encode("utf-8"),
            file_name="reconiq_rule_ml_cross_check.csv",
            mime="text/csv",
        )

with tabs[4]:
    st.subheader("Input data quality")
    st.dataframe(profile, use_container_width=True, hide_index=True)
    st.markdown("**Input controls**")
    st.markdown(
        "- Raw uploads are copied before normalisation.\n"
        "- Required columns are validated before matching.\n"
        "- Duplicate source trade IDs and duplicate business keys are flagged.\n"
        "- Unambiguous note evidence may fill blank external fields; populated fields are not overwritten.\n"
        "- Ambiguous date formats remain unknown rather than being silently interpreted."
    )

with tabs[5]:
    st.subheader("Research-informed system design")
    st.markdown(
        """**Paper 1 - Han, Wang and Huang (2026), IEEE Access.** This paper studies IFRS 16 lease-contract intelligence. ReconIQ adapts its answerability-aware evidence extraction, exact source-span traceability, separation of extraction from downstream rules, and explicit unknown states. It does not reuse the paper's IFRS 16 D1-D6 accounting logic, CUAD / lease disclosure data, trained RoBERTa model, threshold or published scores.

        **Paper 2 - Lippi et al. (2019), CLAUDETTE.** This paper studies potentially unfair online contract clauses, not trade exceptions. ReconIQ adapts its two-task pattern: first detect whether a sentence indicates an issue, then categorise it. The categorisation is multi-label because one narrative can describe more than one break. It uses a lightweight word/character TF-IDF + Linear SVM implementation inspired by the paper's category-specific SVMs and lexical baselines. Adjacent sentence context is included as a low-complexity approximation; it is not the SVM-HMM sequence model used by CLAUDETTE.

        **Why this model choice:** CLAUDETTE reports that C2, its combination of category-specific SVMs, had F1 0.781, while its ensemble C8 had F1 0.805. For the public web service, the authors selected C2 because it was more efficient despite slightly lower performance. ReconIQ similarly favours a compact, inspectable category-specific SVM design for this small prototype rather than reproducing the paper's more complex ensemble.

        **Research discipline adapted from CLAUDETTE:** category definitions, explicit annotation guidance, grouped wording-family splits, per-category precision / recall / F1, false-positive / false-negative review, context inspection and reviewer-facing sentence highlights. CLAUDETTE's legal fairness labels and contract dataset are not transferred to finance.

        **Decision authority:** deterministic trade checks determine the reconciliation status. The text classifier only adds review suggestions. A human analyst approves, rejects or investigates; no trade settlement or ledger posting occurs.

        **Where knowledge lives:** input schemas and R1-R6 checks live in code; market calendars, price tolerances and counterparty conventions must be moved to effective-dated, version-controlled configuration before a pilot; model weights and thresholds need a model version and training-data provenance; final exception adjudication remains with the operations owner.

        **Business-case boundary:** this prototype has no target firm's trade volumes, exception rate, handling time or loaded labour cost. Measure these with an operations owner before estimating cost or savings. A transparent labour baseline is: trades per run x exception rate x minutes per exception / 60 x loaded hourly cost. Rework and actual late-settlement impacts should be measured separately.

        **Weakest point:** the classifier has only synthetic phrase examples and no independently expert-labelled finance corpus. New counterparties, abbreviations, instrument classes and wider document context may change performance."""
    )
    try:
        with open(ROOT / "evaluation" / "exception_classifier_summary.json", encoding="utf-8") as f:
            summary = json.load(f)
        st.markdown("**Synthetic phrase-family-held-out classifier checks**")
        a, b, c = st.columns(3)
        a.metric("Detection F1", f"{summary['detection_f1']:.3f}")
        b.metric("Multi-label macro F1", f"{summary['end_to_end_category_macro_f1']:.3f}")
        c.metric("Held-out examples", summary["evaluation_rows"])
        st.caption("These metrics are from a small synthetic fixture. They are not real-world accuracy or a production benchmark.")
        per_label = pd.DataFrame([
            {"category": label, **values}
            for label, values in summary.get("per_label_end_to_end", {}).items()
        ])
        if not per_label.empty:
            st.markdown("**Per-category held-out results**")
            st.dataframe(per_label, use_container_width=True, hide_index=True)
        error_ids = summary.get("multi_label_exact_set_error_cases", [])
        if error_ids:
            st.markdown("**Multi-label exact-set errors**")
            prediction_path = ROOT / "evaluation" / "exception_classifier_predictions.csv"
            if prediction_path.exists():
                predictions = pd.read_csv(prediction_path, keep_default_na=False)
                error_rows = predictions[predictions["case_id"].astype(str).isin(error_ids)]
                st.dataframe(
                    error_rows[["case_id", "text", "expected_labels", "predicted_labels", "false_positive_labels", "false_negative_labels"]],
                    use_container_width=True, hide_index=True,
                )
            else:
                st.write(", ".join(error_ids))
    except Exception:
        st.caption("Run `python scripts/evaluate_synthetic.py` to generate the synthetic evaluation summary.")

st.caption("Synthetic data only. Keep model suggestions separate from deterministic trade status and obtain human review before any operational action.")
