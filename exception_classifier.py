
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, hamming_loss
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MultiLabelBinarizer

LABELS = [
    "SECURITY_IDENTITY",
    "TRADE_DATE",
    "SIDE",
    "QUANTITY",
    "PRICE_CURRENCY",
    "SETTLEMENT_DATE",
    "ACCOUNT_ALLOCATION",
    "DUPLICATE_OR_MISSING_LINK",
]
LABEL_DESCRIPTIONS = {
    "SECURITY_IDENTITY": "Security / instrument identifier or description differs",
    "TRADE_DATE": "Execution / trade date differs",
    "SIDE": "BUY / SELL direction differs",
    "QUANTITY": "Units, shares or notional differs",
    "PRICE_CURRENCY": "Execution price or currency differs",
    "SETTLEMENT_DATE": "Expected settlement / value date differs",
    "ACCOUNT_ALLOCATION": "Fund, portfolio, account or counterparty allocation differs",
    "DUPLICATE_OR_MISSING_LINK": "Duplicate, orphan or missing counterpart link",
}


@dataclass
class ExceptionModel:
    vectorizer: FeatureUnion
    detector: LinearSVC
    categorizer: OneVsRestClassifier
    binarizer: MultiLabelBinarizer
    detection_threshold: float
    category_thresholds: dict[str, float]
    training_rows: int
    positive_training_rows: int
    validation_rows: int
    validation_detection_f1: float
    validation_category_macro_f1: float
    method: str = "Two-stage TF-IDF + LinearSVC detector + one-vs-rest category SVMs"


def parse_labels(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    labels = [x.strip() for x in str(value).split("|") if x.strip()]
    return list(dict.fromkeys(x for x in labels if x in LABELS))


def _make_vectorizer() -> FeatureUnion:
    # CLAUDETTE found simple lexical TF-IDF features highly competitive. We use
    # word and character n-grams to tolerate field names, abbreviations and codes.
    return FeatureUnion([
        ("word_tfidf", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), lowercase=True,
            strip_accents="unicode", sublinear_tf=True, min_df=1,
        )),
        ("char_tfidf", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), lowercase=True,
            strip_accents="unicode", sublinear_tf=True, min_df=1,
        )),
    ])


def _score_matrix(model: Any, X: Any) -> np.ndarray:
    scores = np.asarray(model.decision_function(X))
    if scores.ndim == 1:
        scores = scores.reshape(-1, 1)
    return scores


def _choose_binary_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    best_threshold, best_f1 = 0.0, -1.0
    for threshold in np.arange(-0.80, 0.81, 0.025):
        pred = (scores >= threshold).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
        # Prefer a higher threshold on ties to reduce noisy exception flags.
        if f1 > best_f1 + 1e-9 or (abs(f1 - best_f1) <= 1e-9 and threshold > best_threshold):
            best_threshold, best_f1 = float(threshold), float(f1)
    return round(best_threshold, 3), float(best_f1)


def _choose_category_thresholds(y_true: np.ndarray, scores: np.ndarray,
                                labels: list[str]) -> tuple[dict[str, float], float]:
    thresholds: dict[str, float] = {}
    f1s = []
    for idx, label in enumerate(labels):
        best_t, best_f1 = 0.0, -1.0
        for threshold in np.arange(-0.80, 0.81, 0.025):
            pred = (scores[:, idx] >= threshold).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(
                y_true[:, idx], pred, average="binary", zero_division=0
            )
            if f1 > best_f1 + 1e-9 or (abs(f1 - best_f1) <= 1e-9 and threshold > best_t):
                best_t, best_f1 = float(threshold), float(f1)
        thresholds[label] = round(best_t, 3)
        f1s.append(float(best_f1))
    return thresholds, float(np.mean(f1s)) if f1s else 0.0


def train_exception_classifier(training_csv: str | Path) -> ExceptionModel:
    data = pd.read_csv(training_csv, keep_default_na=False)
    if not {"text", "labels"}.issubset(data.columns):
        raise ValueError("Training file must contain 'text' and 'labels' columns.")
    texts = data["text"].fillna("").astype(str).tolist()
    label_lists = [parse_labels(x) for x in data["labels"].tolist()]
    y_multi = MultiLabelBinarizer(classes=LABELS).fit_transform(label_lists)
    y_detect = (np.asarray([bool(x) for x in label_lists], dtype=int))
    if len(texts) < 10 or y_multi.shape[1] != len(LABELS) or len(set(y_detect)) < 2:
        raise ValueError("Training data must contain at least 10 rows, all 8 categories, and positive + negative examples.")

    # Grouped split prevents wording variants from the same synthetic source family
    # appearing in both training and validation. This mirrors the paper's document-level
    # evaluation principle, though this is a fixture split rather than a real corpus split.
    if "group_id" in data.columns:
        validation_mask = data["group_id"].astype(str).str.match(r".*_(?:TRAIN|NEG)_4$")
    else:
        validation_mask = pd.Series(False, index=data.index)
    if not validation_mask.any() or validation_mask.all():
        raise ValueError("Training CSV must contain separate grouped validation rows ending in _TRAIN_4 or _NEG_4.")

    train_mask = ~validation_mask
    train_texts = data.loc[train_mask, "text"].astype(str).tolist()
    val_texts = data.loc[validation_mask, "text"].astype(str).tolist()
    train_label_lists = [parse_labels(x) for x in data.loc[train_mask, "labels"].tolist()]
    val_label_lists = [parse_labels(x) for x in data.loc[validation_mask, "labels"].tolist()]
    y_train_detect = np.asarray([bool(x) for x in train_label_lists], dtype=int)
    y_val_detect = np.asarray([bool(x) for x in val_label_lists], dtype=int)
    binarizer = MultiLabelBinarizer(classes=LABELS)
    y_train_multi = binarizer.fit_transform(train_label_lists)
    y_val_multi = binarizer.transform(val_label_lists)

    val_vectorizer = _make_vectorizer()
    X_train = val_vectorizer.fit_transform(train_texts)
    X_val = val_vectorizer.transform(val_texts)

    val_detector = LinearSVC(C=1.0, class_weight="balanced", random_state=42)
    val_detector.fit(X_train, y_train_detect)
    detection_scores = val_detector.decision_function(X_val)
    detection_threshold, detection_f1 = _choose_binary_threshold(y_val_detect, detection_scores)

    positive_train_mask = y_train_detect.astype(bool)
    val_categorizer = OneVsRestClassifier(LinearSVC(C=1.0, class_weight="balanced", random_state=42))
    val_categorizer.fit(X_train[positive_train_mask], y_train_multi[positive_train_mask])
    category_scores = _score_matrix(val_categorizer, X_val)
    category_thresholds, category_macro_f1 = _choose_category_thresholds(
        y_val_multi, category_scores, list(binarizer.classes_)
    )

    # Refit both stages on the complete labelled synthetic training corpus after
    # threshold selection. The evaluation file remains untouched.
    vectorizer = _make_vectorizer()
    X_all = vectorizer.fit_transform(texts)
    detector = LinearSVC(C=1.0, class_weight="balanced", random_state=42)
    detector.fit(X_all, y_detect)
    positive_mask = y_detect.astype(bool)
    categorizer = OneVsRestClassifier(LinearSVC(C=1.0, class_weight="balanced", random_state=42))
    categorizer.fit(X_all[positive_mask], y_multi[positive_mask])

    return ExceptionModel(
        vectorizer=vectorizer,
        detector=detector,
        categorizer=categorizer,
        binarizer=binarizer,
        detection_threshold=detection_threshold,
        category_thresholds=category_thresholds,
        training_rows=len(data),
        positive_training_rows=int(y_detect.sum()),
        validation_rows=len(val_texts),
        validation_detection_f1=round(detection_f1, 4),
        validation_category_macro_f1=round(category_macro_f1, 4),
    )


def _postprocess_categories(text: str, suggestions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Apply one narrow rule for a recurring duplicate-link false positive.

    CLAUDETTE's error analysis proposes targeted rule-based post-processing.
    Here, an explicit statement that one unique counterpart exists or that no
    duplicate/orphan is present suppresses only the duplicate/missing-link tag.
    Other issue categories remain untouched.
    """
    text_lower = str(text or "").lower()
    clean_link_patterns = [
        r"\bno\s+(?:duplicate|orphan|missing (?:counterpart|link))\b",
        r"\b(one|single)\s+unique\s+(?:internal\s+)?transaction\b",
        r"\b(one|single)\s+(?:matching|corresponding)\s+(?:broker|custodian|external)?\s*(?:confirmation|record|counterpart)\b",
        r"\bexactly one\s+(?:internal\s+)?(?:trade|booking)\b",
    ]
    explicitly_clean = any(re.search(pattern, text_lower) for pattern in clean_link_patterns)
    if not explicitly_clean:
        return suggestions, []
    kept = [x for x in suggestions if x["label"] != "DUPLICATE_OR_MISSING_LINK"]
    suppressed = ["DUPLICATE_OR_MISSING_LINK"] if len(kept) != len(suggestions) else []
    return kept, suppressed


def predict_texts(model: ExceptionModel, texts: list[str], force_categorize: bool = False) -> list[dict[str, Any]]:
    clean = [str(x or "") for x in texts]
    X = model.vectorizer.transform(clean)
    detection_margins = np.asarray(model.detector.decision_function(X)).reshape(-1)
    gate_flags = detection_margins >= model.detection_threshold
    category_scores = _score_matrix(model.categorizer, X)
    results = []
    for i, text in enumerate(clean):
        gate = bool(gate_flags[i])
        category_items = []
        if gate or force_categorize:
            for idx, label in enumerate(model.binarizer.classes_):
                margin = float(category_scores[i, idx])
                threshold = model.category_thresholds[str(label)]
                if margin >= threshold:
                    category_items.append({
                        "label": str(label),
                        "description": LABEL_DESCRIPTIONS[str(label)],
                        "margin": round(margin, 4),
                        "threshold": threshold,
                    })
        category_items.sort(key=lambda x: x["margin"], reverse=True)
        category_items, suppressed = _postprocess_categories(text, category_items)
        # Suppress the binary alert only when its only category was the duplicate
        # tag and the source explicitly states a unique / clean link.
        final_gate = gate and not (suppressed and not category_items)
        results.append({
            "exception_detected": final_gate,
            "raw_detector_flag": gate,
            "postprocess_suppressed_labels": "|".join(suppressed),
            "detection_margin": round(float(detection_margins[i]), 4),
            "category_suggestions": category_items,
            "suggested_labels": "|".join(x["label"] for x in category_items),
            "text": text,
        })
    return results


def split_sentences(text: str) -> list[dict[str, Any]]:
    """Split a confirmation note into sentence-like segments and source offsets."""
    text = str(text or "")
    segments = []
    start_cursor = 0
    # A full stop is a sentence boundary only when followed by whitespace or
    # end-of-text. This avoids splitting decimal prices such as 25.10.
    for match in re.finditer(r"[.!?](?=\s|$)|\n+", text):
        raw_start, raw_end = start_cursor, match.end()
        while raw_start < raw_end and text[raw_start].isspace():
            raw_start += 1
        while raw_end > raw_start and text[raw_end - 1].isspace():
            raw_end -= 1
        if raw_end > raw_start:
            segments.append({"sentence": text[raw_start:raw_end],
                             "start_offset": raw_start, "end_offset": raw_end})
        start_cursor = match.end()
    raw_start, raw_end = start_cursor, len(text)
    while raw_start < raw_end and text[raw_start].isspace():
        raw_start += 1
    while raw_end > raw_start and text[raw_end - 1].isspace():
        raw_end -= 1
    if raw_end > raw_start:
        segments.append({"sentence": text[raw_start:raw_end],
                         "start_offset": raw_start, "end_offset": raw_end})
    return segments


def classify_with_context(model: ExceptionModel, text: str, context_radius: int = 1) -> pd.DataFrame:
    """Detect and categorize sentence segments with adjacent-sentence context.

    The sentence window adds local context because a trade break may be described
    across two adjacent sentences. This is deliberately a lightweight approximation;
    it is not CLAUDETTE's structured SVM-HMM collective classifier.
    """
    segments = split_sentences(text)
    columns = ["sentence", "start_offset", "end_offset", "context_text", "exception_detected",
               "raw_detector_flag", "postprocess_suppressed_labels", "detection_margin",
               "suggested_labels", "category_suggestions"]
    if not segments:
        return pd.DataFrame(columns=columns)
    contexts = []
    for i in range(len(segments)):
        lo = max(0, i - context_radius)
        hi = min(len(segments), i + context_radius + 1)
        contexts.append(" ".join(x["sentence"] for x in segments[lo:hi]))
    predictions = predict_texts(model, contexts)
    rows = []
    for segment, context, prediction in zip(segments, contexts, predictions):
        rows.append({**segment, "context_text": context, **{k: v for k, v in prediction.items() if k != "text"}})
    return pd.DataFrame(rows, columns=columns)


def evaluate_classifier(model: ExceptionModel, evaluation_csv: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    data = pd.read_csv(evaluation_csv, keep_default_na=False)
    texts = data["text"].fillna("").astype(str).tolist()
    truth = [parse_labels(x) for x in data["labels"].tolist()]
    y_true_multi = model.binarizer.transform(truth)
    y_true_detect = np.asarray([bool(x) for x in truth], dtype=int)
    predictions = predict_texts(model, texts)
    y_pred_detect = np.asarray([x["exception_detected"] for x in predictions], dtype=int)
    y_pred_raw_detect = np.asarray([x["raw_detector_flag"] for x in predictions], dtype=int)
    y_pred_multi = model.binarizer.transform([parse_labels(x["suggested_labels"]) for x in predictions])

    p_raw, r_raw, f_raw, _ = precision_recall_fscore_support(y_true_detect, y_pred_raw_detect, average="binary", zero_division=0)
    p_det, r_det, f_det, _ = precision_recall_fscore_support(y_true_detect, y_pred_detect, average="binary", zero_division=0)
    p_micro, r_micro, f_micro, _ = precision_recall_fscore_support(y_true_multi, y_pred_multi, average="micro", zero_division=0)
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(y_true_multi, y_pred_multi, average="macro", zero_division=0)

    true_positive_indices = np.flatnonzero(y_true_detect == 1)
    # Conditional category result removes the binary detection gate and measures
    # the independent category model on examples known to contain an exception.
    forced = predict_texts(model, [texts[i] for i in true_positive_indices], force_categorize=True)
    if len(true_positive_indices):
        cond_pred = model.binarizer.transform([parse_labels(x["suggested_labels"]) for x in forced])
        cond_true = y_true_multi[true_positive_indices]
        p_cond, r_cond, f_cond, _ = precision_recall_fscore_support(cond_true, cond_pred, average="macro", zero_division=0)
    else:
        p_cond = r_cond = f_cond = 0.0

    per_label = {}
    for idx, label in enumerate(model.binarizer.classes_):
        p, r, f, support = precision_recall_fscore_support(y_true_multi[:, idx], y_pred_multi[:, idx], average="binary", zero_division=0)
        per_label[str(label)] = {"precision": round(float(p), 4), "recall": round(float(r), 4), "f1": round(float(f), 4), "support": int(y_true_multi[:, idx].sum())}

    result_rows = []
    for i, row in data.iterrows():
        expected, predicted = truth[i], parse_labels(predictions[i]["suggested_labels"])
        result_rows.append({
            "case_id": row.get("case_id", str(i)), "group_id": row.get("group_id", ""), "text": texts[i],
            "expected_exception": bool(expected), "predicted_exception": predictions[i]["exception_detected"],
            "raw_svm_detector_flag": predictions[i]["raw_detector_flag"],
            "postprocess_suppressed_labels": predictions[i]["postprocess_suppressed_labels"],
            "detection_margin": predictions[i]["detection_margin"],
            "expected_labels": "|".join(expected), "predicted_labels": "|".join(predicted),
            "exact_set_match": set(expected) == set(predicted),
            "true_positive_labels": "|".join(sorted(set(expected) & set(predicted))),
            "false_positive_labels": "|".join(sorted(set(predicted) - set(expected))),
            "false_negative_labels": "|".join(sorted(set(expected) - set(predicted))),
        })

    summary = {
        "training_rows": model.training_rows,
        "positive_training_rows": model.positive_training_rows,
        "validation_rows_for_threshold_selection": model.validation_rows,
        "evaluation_rows": len(data),
        "detection_threshold": model.detection_threshold,
        "category_thresholds": model.category_thresholds,
        "validation_detection_f1": model.validation_detection_f1,
        "validation_category_macro_f1": model.validation_category_macro_f1,
        "raw_svm_detection_precision": round(float(p_raw), 4),
        "raw_svm_detection_recall": round(float(r_raw), 4),
        "raw_svm_detection_f1": round(float(f_raw), 4),
        "postprocessed_detection_precision": round(float(p_det), 4),
        "postprocessed_detection_recall": round(float(r_det), 4),
        "postprocessed_detection_f1": round(float(f_det), 4),
        "detection_precision": round(float(p_det), 4),
        "detection_recall": round(float(r_det), 4),
        "detection_f1": round(float(f_det), 4),
        "end_to_end_category_micro_precision": round(float(p_micro), 4),
        "end_to_end_category_micro_recall": round(float(r_micro), 4),
        "end_to_end_category_micro_f1": round(float(f_micro), 4),
        "end_to_end_category_macro_precision": round(float(p_macro), 4),
        "end_to_end_category_macro_recall": round(float(r_macro), 4),
        "end_to_end_category_macro_f1": round(float(f_macro), 4),
        "conditional_category_macro_precision_on_true_exceptions": round(float(p_cond), 4),
        "conditional_category_macro_recall_on_true_exceptions": round(float(r_cond), 4),
        "conditional_category_macro_f1_on_true_exceptions": round(float(f_cond), 4),
        "exact_label_set_accuracy": round(float(accuracy_score(y_true_multi, y_pred_multi)), 4),
        "label_hamming_loss": round(float(hamming_loss(y_true_multi, y_pred_multi)), 4),
        "per_label_end_to_end": per_label,
        "detector_false_positive_cases": [str(data.iloc[i].get("case_id", i)) for i in range(len(data)) if y_true_detect[i] == 0 and y_pred_detect[i] == 1],
        "detector_false_negative_cases": [str(data.iloc[i].get("case_id", i)) for i in range(len(data)) if y_true_detect[i] == 1 and y_pred_detect[i] == 0],
        "multi_label_exact_set_error_cases": [str(data.iloc[i].get("case_id", i)) for i in range(len(data)) if set(truth[i]) != set(parse_labels(predictions[i]["suggested_labels"]))],
        "note": "Synthetic, phrase-family-held-out fixture only. Not a production benchmark or real-world accuracy estimate.",
    }
    return pd.DataFrame(result_rows), summary
