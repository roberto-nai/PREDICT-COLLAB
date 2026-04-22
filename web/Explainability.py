"""
SHAP explainability utilities for trained classification models.
"""

import json
import os
from collections import OrderedDict, defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

from ConfigLoader import get_processing_dir, get_shap_explanations_dirname, get_max_decimal_places
from FuncionesAuxiliares import load_variables
from processtransformer.constants import Task
from processtransformer.models.transformer import TokenAndPositionEmbedding, TransformerBlock


class ShapExplainer:
    def __init__(self):
        pass

    def _validate_task(self, input_task):
        if input_task not in (Task.NEXT_ACTIVITY, Task.NEXT_MESSAGE_SEND):
            raise ValueError("SHAP Explain is currently available only for classification models.")

    def _prepare_prediction_column(self, df, tipo_pred, column1, column2, column3):
        if tipo_pred in ("NextActivity", "Participant"):
            missing = [c for c in [column1] if c not in df.columns]
            if missing:
                raise ValueError(f"Missing required column(s): {missing}")
            return df[column1].astype(str)

        if tipo_pred in ("ParticipantSend", "ActivityParticipant"):
            required = [column1, column2]
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise ValueError(f"Missing required column(s): {missing}")
            return df[column1].astype(str) + '_' + df[column2].astype(str)

        if tipo_pred == "ActivityParticipantSend":
            required = [column1, column2, column3]
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise ValueError(f"Missing required column(s): {missing}")
            return (
                df[column1].astype(str)
                + '_'
                + df[column2].astype(str)
                + '_'
                + df[column3].astype(str)
            )

        raise ValueError(f"Unsupported prediction type for SHAP explainability: {tipo_pred}")

    def _build_case_sequences(self, df, case_id_col, pred_series):
        if case_id_col not in df.columns:
            raise ValueError(f"Case ID column '{case_id_col}' not found in log file.")

        cases = OrderedDict()
        for case_id, raw_value in zip(df[case_id_col].tolist(), pred_series.tolist()):
            normalized = str(raw_value).replace(' ', '-').lower()
            if case_id not in cases:
                cases[case_id] = []
            cases[case_id].append(normalized)

        return list(cases.keys()), list(cases.values())

    def _build_token_x(self, sequences, x_word_dict, max_case_length):
        token_x = np.zeros((len(sequences), max_case_length), dtype=np.float32)

        for row_idx, seq in enumerate(sequences):
            row = np.zeros((max_case_length,), dtype=np.float32)
            for event in seq:
                token = x_word_dict.get(event, 0)
                row = np.insert(row[1:], row.size - 1, token)
            token_x[row_idx] = row

        return token_x

    def _predict_probabilities(self, model, token_x):
        logits = model.predict(token_x, verbose=0)
        probs = tf.nn.softmax(logits, axis=1).numpy()
        return probs

    def _invert_dict(self, data):
        return {v: k for k, v in data.items()}

    def _load_existing_results(self, detail_csv_path, summary_csv_path, metadata_path):
        detail_df = pd.read_csv(detail_csv_path)
        summary_df = pd.read_csv(summary_csv_path)

        # Normalize NaN values to empty strings for safe template rendering
        detail_df = detail_df.where(pd.notnull(detail_df), "")
        summary_df = summary_df.where(pd.notnull(summary_df), "")

        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as metadata_file:
                metadata = json.load(metadata_file)

        explained_cases = metadata.get('parameters', {}).get('explained_cases', len(detail_df))

        return {
            "rows": detail_df.to_dict(orient='records'),
            "summary_rows": summary_df.to_dict(orient='records'),
            "explained_cases": int(explained_cases),
            "metadata": metadata,
        }

    def _get_row_shap_for_predicted_class(self, shap_values, sample_idx, pred_class, num_samples, num_features):
        """
        Extract SHAP values for one sample and one predicted class, handling
        different return shapes across SHAP versions.
        """
        if isinstance(shap_values, list):
            # Common format: list[num_classes] with each item shaped (num_samples, num_features)
            if pred_class >= len(shap_values):
                raise ValueError(
                    f"SHAP class index {pred_class} out of bounds for list size {len(shap_values)}"
                )
            per_class_values = np.asarray(shap_values[pred_class])
            if per_class_values.ndim == 1:
                return per_class_values
            if sample_idx >= per_class_values.shape[0]:
                raise ValueError(
                    f"SHAP sample index {sample_idx} out of bounds for shape {per_class_values.shape}"
                )
            return per_class_values[sample_idx]

        shap_arr = np.asarray(shap_values)

        if shap_arr.ndim == 2:
            # Binary/single-output style: (num_samples, num_features)
            if sample_idx >= shap_arr.shape[0]:
                raise ValueError(
                    f"SHAP sample index {sample_idx} out of bounds for shape {shap_arr.shape}"
                )
            return shap_arr[sample_idx]

        if shap_arr.ndim == 3:
            # Possible formats:
            # 1) (num_samples, num_features, num_classes)
            # 2) (num_classes, num_samples, num_features)
            # 3) (num_samples, num_classes, num_features)
            if shap_arr.shape[0] == num_samples and shap_arr.shape[1] == num_features:
                # (num_samples, num_features, num_classes)
                if pred_class >= shap_arr.shape[2]:
                    raise ValueError(
                        f"SHAP class index {pred_class} out of bounds for shape {shap_arr.shape}"
                    )
                return shap_arr[sample_idx, :, pred_class]

            if shap_arr.shape[0] != num_samples and shap_arr.shape[1] == num_samples and shap_arr.shape[2] == num_features:
                # (num_classes, num_samples, num_features)
                if pred_class >= shap_arr.shape[0]:
                    raise ValueError(
                        f"SHAP class index {pred_class} out of bounds for shape {shap_arr.shape}"
                    )
                return shap_arr[pred_class, sample_idx, :]

            if shap_arr.shape[0] == num_samples and shap_arr.shape[2] == num_features:
                # (num_samples, num_classes, num_features)
                if pred_class >= shap_arr.shape[1]:
                    raise ValueError(
                        f"SHAP class index {pred_class} out of bounds for shape {shap_arr.shape}"
                    )
                return shap_arr[sample_idx, pred_class, :]

        raise ValueError(f"Unsupported SHAP values shape: {shap_arr.shape}")

    def _extract_top_events_per_case(
        self,
        token_x,
        case_ids,
        predicted_idx,
        predicted_labels,
        probs,
        shap_values,
        inv_x_dict,
        top_k=3,
        mean_abs_decimal_places=6,
    ):
        rows = []
        event_importance_acc = defaultdict(lambda: {"sum_abs": 0.0, "count": 0})
        num_samples, num_features = token_x.shape
        total_cases_considered = len(case_ids)

        for i, case_id in enumerate(case_ids):
            pred_class = int(predicted_idx[i])
            row_shap = self._get_row_shap_for_predicted_class(
                shap_values=shap_values,
                sample_idx=i,
                pred_class=pred_class,
                num_samples=num_samples,
                num_features=num_features,
            )

            row_tokens = token_x[i]
            nonzero_positions = np.where(row_tokens != 0)[0]

            ranked = []
            if len(nonzero_positions) > 0:
                ranked_positions = sorted(nonzero_positions, key=lambda p: abs(float(row_shap[p])), reverse=True)
                ranked_positions = ranked_positions[:top_k]

                real_position_lookup = {int(pos): idx + 1 for idx, pos in enumerate(nonzero_positions.tolist())}
                for pos in ranked_positions:
                    token_value = int(row_tokens[pos])
                    event_name = inv_x_dict.get(token_value, f"<UNK:{token_value}>")
                    shap_val = float(row_shap[pos])
                    real_pos = real_position_lookup.get(int(pos), int(pos) + 1)

                    ranked.append(
                        {
                            "event_name": event_name,
                            "event_position": real_pos,
                            "padded_position": int(pos) + 1,
                            "shap_value": shap_val,
                        }
                    )

                    event_importance_acc[event_name]["sum_abs"] += abs(shap_val)
                    event_importance_acc[event_name]["count"] += 1

            row = {
                "case_id": str(case_id),
                "predicted_label": str(predicted_labels[i]),
                "predicted_confidence": float(probs[i, pred_class]),
            }

            for rank_idx in range(top_k):
                key = f"event_rank_{rank_idx + 1}"
                if rank_idx < len(ranked):
                    item = ranked[rank_idx]
                    row[key] = (
                        f"pos={item['event_position']} | event={item['event_name']} | "
                        f"shap={item['shap_value']:.6f}"
                    )
                else:
                    row[key] = ""

            rows.append(row)

        summary_rows = []
        for event_name, data in event_importance_acc.items():
            count = data["count"]
            mean_abs = data["sum_abs"] / count if count else 0.0
            per_occurrences = round((count / total_cases_considered) * 100, 2) if total_cases_considered else 0.0
            summary_rows.append(
                {
                    "event_name": event_name,
                    "mean_abs_shap": round(mean_abs, mean_abs_decimal_places),
                    "occurrences_total": int(count),
                    "occurrences_percent": per_occurrences,
                    "cases": int(total_cases_considered),
                }
            )

        summary_rows = sorted(summary_rows, key=lambda item: item["mean_abs_shap"], reverse=True)

        return rows, summary_rows

    def explain_model(self, process_name, log_name, prediction_type, model_name, top_k=3, nsamples=100, force_recompute=False):
        started_at_utc = datetime.utcnow()

        try:
            import shap
        except ImportError as exc:
            raise ValueError("Package 'shap' is required. Install dependencies from requirements.txt.") from exc

        processing_dir = get_processing_dir()
        model_dir = os.path.join(processing_dir, process_name, log_name, prediction_type, model_name)
        model_properties_path = os.path.join(model_dir, 'propiedadesModelo.txt')
        model_file = os.path.join(model_dir, model_name + 'T.keras')
        source_log = os.path.join(processing_dir, process_name, log_name, log_name)

        if not os.path.exists(model_dir):
            raise ValueError(f"Model path not found: {model_dir}")
        if not os.path.exists(model_properties_path):
            raise ValueError(f"Missing model properties file: {model_properties_path}")
        if not os.path.exists(model_file):
            raise ValueError(f"Missing trained model file: {model_file}")
        if not os.path.exists(source_log):
            raise ValueError(f"Missing source log file: {source_log}")

        shap_folder_name = get_shap_explanations_dirname()
        output_dir = os.path.join(model_dir, shap_folder_name)
        detail_csv_path = os.path.join(output_dir, 'shap_top_events_per_case.csv')
        summary_csv_path = os.path.join(output_dir, 'shap_summary.csv')
        metadata_path = os.path.join(output_dir, 'shap_metadata.json')

        if (not force_recompute
                and os.path.exists(detail_csv_path)
                and os.path.exists(summary_csv_path)
                and os.path.exists(metadata_path)):
            cached_result = self._load_existing_results(detail_csv_path, summary_csv_path, metadata_path)
            return {
                "output_dir": output_dir,
                "detail_csv_path": detail_csv_path,
                "summary_csv_path": summary_csv_path,
                "metadata_path": metadata_path,
                "rows": cached_result["rows"],
                "summary_rows": cached_result["summary_rows"],
                "explained_cases": cached_result["explained_cases"],
                "model": model_name,
                "prediction_type": prediction_type,
                "cached": True,
            }

        properties = load_variables(model_properties_path)

        x_word_dict = properties[0]
        y_word_dict = properties[1]
        max_case_length = properties[2]
        column_id = properties[3]
        column_1 = properties[5]
        column_2 = properties[6]
        column_3 = properties[7]
        input_task = properties[8]
        tipo_pred = properties[9]

        self._validate_task(input_task)

        df = pd.read_csv(source_log)
        prediction_values = self._prepare_prediction_column(df, tipo_pred, column_1, column_2, column_3)
        case_ids, sequences = self._build_case_sequences(df, column_id, prediction_values)

        if len(case_ids) == 0:
            raise ValueError("No cases found in source log.")

        token_x = self._build_token_x(sequences, x_word_dict, max_case_length)

        model = load_model(
            model_file,
            custom_objects={
                "TokenAndPositionEmbedding": TokenAndPositionEmbedding,
                "TransformerBlock": TransformerBlock,
            },
            compile=False,
        )

        probs = self._predict_probabilities(model, token_x)
        predicted_idx = np.argmax(probs, axis=1)

        inv_y_dict = self._invert_dict(y_word_dict)
        inv_x_dict = self._invert_dict(x_word_dict)
        predicted_labels = [inv_y_dict.get(int(idx), str(idx)) for idx in predicted_idx]

        background_size = min(30, token_x.shape[0])
        explain_size = min(200, token_x.shape[0])
        background = token_x[:background_size]
        explain_samples = token_x[:explain_size]
        explain_case_ids = case_ids[:explain_size]
        explain_predicted_idx = predicted_idx[:explain_size]
        explain_predicted_labels = predicted_labels[:explain_size]
        explain_probs = probs[:explain_size]

        explainer = shap.KernelExplainer(lambda x: self._predict_probabilities(model, x), background)
        shap_values = explainer.shap_values(explain_samples, nsamples=nsamples)

        detail_rows, summary_rows = self._extract_top_events_per_case(
            explain_samples,
            explain_case_ids,
            explain_predicted_idx,
            explain_predicted_labels,
            explain_probs,
            shap_values,
            inv_x_dict,
            top_k=top_k,
            mean_abs_decimal_places=get_max_decimal_places(),
        )

        os.makedirs(output_dir, exist_ok=True)

        pd.DataFrame(detail_rows).to_csv(detail_csv_path, index=False)
        pd.DataFrame(summary_rows).to_csv(summary_csv_path, index=False)

        ended_at_utc = datetime.utcnow()
        delta_time_seconds = round((ended_at_utc - started_at_utc).total_seconds(), 2)
        delta_time_minutes = round(delta_time_seconds / 60, 2)

        metadata = {
            "started_at": started_at_utc.isoformat() + "Z",
            "ended_at": ended_at_utc.isoformat() + "Z",
            "delta_time_sec": delta_time_seconds,
            "delta_time_min": delta_time_minutes,
            "process": process_name,
            "log": log_name,
            "prediction_type": prediction_type,
            "model": model_name,
            "input_task": str(input_task),
            "source_log": source_log,
            "model_file": model_file,
            "output_dir": output_dir,
            "parameters": {
                "top_k": int(top_k),
                "nsamples": int(nsamples),
                "background_size": int(background_size),
                "explained_cases": int(explain_size),
                "max_case_length": int(max_case_length),
                "explainer": "KernelExplainer",
                "sequence_only": True,
            },
            "outputs": {
                "detail_csv": detail_csv_path,
                "summary_csv": summary_csv_path,
                "metadata_json": metadata_path,
            },
        }

        with open(metadata_path, 'w', encoding='utf-8') as metadata_file:
            json.dump(metadata, metadata_file, indent=2)

        return {
            "output_dir": output_dir,
            "detail_csv_path": detail_csv_path,
            "summary_csv_path": summary_csv_path,
            "metadata_path": metadata_path,
            "rows": detail_rows,
            "summary_rows": summary_rows,
            "explained_cases": explain_size,
            "model": model_name,
            "prediction_type": prediction_type,
            "cached": False,
        }
