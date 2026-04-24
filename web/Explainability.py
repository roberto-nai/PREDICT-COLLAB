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

from ConfigLoader import get_processing_dir, get_shap_explanations_dirname, get_max_decimal_places, get_shap_max_cases
from FuncionesAuxiliares import load_variables
from processtransformer.constants import Task
from processtransformer.models.transformer import TokenAndPositionEmbedding, TransformerBlock


class ShapExplainer:
    """Computes SHAP-based explanations for trained classification models.

    Uses a model-agnostic KernelExplainer to approximate Shapley values over
    tokenised event sequences, then aggregates results at both model and log level.
    """

    def __init__(self):
        pass

    def _get_log_level_shap_paths(self, processing_dir, process_name, log_name):
        """Return the paths of the two log-level SHAP aggregate CSV files.

        Args:
            processing_dir (str): Root processing directory.
            process_name (str): Name of the process.
            log_name (str): Name of the event log.

        Returns:
            tuple[str, str, str]: (log_dir, shap_metrics_path, shap_common_top10_path)
        """
        log_dir = os.path.join(processing_dir, process_name, log_name)
        shap_metrics_path = os.path.join(log_dir, 'shap_metrics.csv')
        shap_common_top10_path = os.path.join(log_dir, 'shap_common_top10.csv')
        return log_dir, shap_metrics_path, shap_common_top10_path

    def _upsert_shap_metrics(self, shap_metrics_path, process_name, log_name, prediction_type, model_name, summary_rows, top_n=10):
        """Write or update the top-N SHAP events for one model/task in the log-level metrics CSV.

        If an entry already exists for the same process/log/task/model combination it is
        replaced, preserving entries from all other models (upsert semantics).

        Args:
            shap_metrics_path (str): Absolute path to shap_metrics.csv.
            process_name (str): Name of the process.
            log_name (str): Name of the event log.
            prediction_type (str): Prediction task label (e.g. 'NextActivity').
            model_name (str): Identifier of the trained model.
            summary_rows (list[dict]): Per-event summary dicts from _extract_top_events_per_case.
            top_n (int): Maximum number of top events to persist. Defaults to 10.
        """
        # Keep only the top-N events from the per-model summary
        top_rows = summary_rows[:top_n]
        if not top_rows:
            return

        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        new_rows = []
        for idx, item in enumerate(top_rows, start=1):
            new_rows.append(
                {
                    'Timestamp': now_str,
                    'Process': process_name,
                    'File': log_name,
                    'Prediction_Type': prediction_type,
                    'Model_Name': model_name,
                    'Rank': int(idx),
                    'Event_Name': str(item.get('event_name', '')),
                    'Mean_Abs_SHAP': float(item.get('mean_abs_shap', 0.0)),
                    'Occurrences_Total': int(item.get('occurrences_total', 0)),
                    'Occurrences_Percent': float(item.get('occurrences_percent', 0.0)),
                    'Cases': int(item.get('cases', 0)),
                }
            )

        new_df = pd.DataFrame(new_rows)

        required_cols = {'Process', 'File', 'Prediction_Type', 'Model_Name'}

        if os.path.exists(shap_metrics_path):
            existing_df = pd.read_csv(shap_metrics_path)
            if required_cols.issubset(set(existing_df.columns)):
                # Remove stale rows for this exact model/task combination
                keep_mask = ~(
                    (existing_df['Process'].astype(str) == str(process_name))
                    & (existing_df['File'].astype(str) == str(log_name))
                    & (existing_df['Prediction_Type'].astype(str) == str(prediction_type))
                    & (existing_df['Model_Name'].astype(str) == str(model_name))
                )
                existing_df = existing_df[keep_mask]
                merged_df = pd.concat([existing_df, new_df], ignore_index=True)
                merged_df.to_csv(shap_metrics_path, index=False)
            else:
                # Existing file has an incompatible schema; overwrite it
                new_df.to_csv(shap_metrics_path, index=False)
        else:
            # First run: create the file from scratch
            new_df.to_csv(shap_metrics_path, index=False)

    def _refresh_shap_common_top10(self, shap_metrics_path, shap_common_top10_path, top_n=10):
        """Recompute and overwrite the global cross-model top-N events CSV.

        Reads all rows from shap_metrics.csv, aggregates per unique event name across
        every model/task, computes Coverage_Rate, Composite_Score and
        Normalized_Composite_Score, and writes the top-N events ranked by
        Normalized_Composite_Score.

        Args:
            shap_metrics_path (str): Absolute path to shap_metrics.csv.
            shap_common_top10_path (str): Absolute path to shap_common_top10.csv (output).
            top_n (int): Number of top events to keep in the output. Defaults to 10.
        """
        if not os.path.exists(shap_metrics_path):
            return

        metrics_df = pd.read_csv(shap_metrics_path)
        if metrics_df.empty:
            return

        # Coerce numeric columns to float/int, replacing any parse errors with 0
        metrics_df['Mean_Abs_SHAP'] = pd.to_numeric(metrics_df['Mean_Abs_SHAP'], errors='coerce').fillna(0.0)
        metrics_df['Occurrences_Total'] = pd.to_numeric(metrics_df['Occurrences_Total'], errors='coerce').fillna(0)

        # Build a composite key to count distinct models unambiguously
        metrics_df['Model_Key'] = (
            metrics_df['Process'].astype(str)
            + '|'
            + metrics_df['File'].astype(str)
            + '|'
            + metrics_df['Prediction_Type'].astype(str)
            + '|'
            + metrics_df['Model_Name'].astype(str)
        )

        # Total number of distinct models in the metrics file (used to normalise Coverage_Rate)
        total_models = metrics_df['Model_Key'].nunique()

        aggregated = (
            metrics_df.groupby('Event_Name', dropna=False)
            .agg(
                Mean_Weight=('Mean_Abs_SHAP', 'mean'),
                Median_Weight=('Mean_Abs_SHAP', 'median'),
                Models_Count=('Model_Key', pd.Series.nunique),
                Tasks_Count=('Prediction_Type', pd.Series.nunique),
                Total_Occurrences=('Occurrences_Total', 'sum'),
            )
            .reset_index()
        )

        # Coverage_Rate: fraction of models in which the event appears (0–1)
        aggregated['Coverage_Rate'] = aggregated['Models_Count'] / total_models if total_models else 0.0
        # Composite_Score: absolute product of model count and mean SHAP weight
        aggregated['Composite_Score'] = aggregated['Models_Count'] * aggregated['Mean_Weight']
        # Normalized_Composite_Score: Coverage_Rate × Mean_Weight — comparable across experiments
        aggregated['Normalized_Composite_Score'] = aggregated['Coverage_Rate'] * aggregated['Mean_Weight']

        aggregated = aggregated.sort_values(
            by=['Normalized_Composite_Score', 'Composite_Score', 'Models_Count', 'Mean_Weight', 'Total_Occurrences'],
            ascending=[False, False, False, False, False],
        )

        aggregated['Coverage_Rate'] = aggregated['Coverage_Rate'].round(get_max_decimal_places())
        aggregated['Composite_Score'] = aggregated['Composite_Score'].round(get_max_decimal_places())
        aggregated['Normalized_Composite_Score'] = aggregated['Normalized_Composite_Score'].round(get_max_decimal_places())

        top_df = aggregated.head(top_n).copy()
        top_df.insert(0, 'Rank', range(1, len(top_df) + 1))
        top_df['Coverage_Rate'] = top_df['Coverage_Rate'].round(get_max_decimal_places())
        top_df['Composite_Score'] = top_df['Composite_Score'].round(get_max_decimal_places())
        top_df['Normalized_Composite_Score'] = top_df['Normalized_Composite_Score'].round(get_max_decimal_places())
        top_df['Mean_Weight'] = top_df['Mean_Weight'].round(get_max_decimal_places())
        top_df['Median_Weight'] = top_df['Median_Weight'].round(get_max_decimal_places())
        top_df.to_csv(shap_common_top10_path, index=False)

    def _validate_task(self, input_task):
        """Raise ValueError if the task is not supported by the SHAP explainer.

        Args:
            input_task (Task): Task enum value loaded from model properties.

        Raises:
            ValueError: If the task is not NEXT_ACTIVITY or NEXT_MESSAGE_SEND.
        """
        if input_task not in (Task.NEXT_ACTIVITY, Task.NEXT_MESSAGE_SEND):
            raise ValueError("SHAP Explain is currently available only for classification models.")

    def _prepare_prediction_column(self, df, tipo_pred, column1, column2, column3):
        """Build the prediction target series by combining the relevant log columns.

        For single-column tasks (NextActivity, Participant) returns column1 as-is.
        For two-column tasks the two columns are concatenated with an underscore.
        For three-column tasks all three columns are concatenated.

        Args:
            df (pd.DataFrame): Source event log.
            tipo_pred (str): Prediction type label.
            column1 (str): Primary column name.
            column2 (str): Secondary column name (may be unused).
            column3 (str): Tertiary column name (may be unused).

        Returns:
            pd.Series: String series used as the event label for sequence building.

        Raises:
            ValueError: If a required column is missing or the task is unsupported.
        """
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
        """Group log rows into ordered event sequences, one per case.

        Args:
            df (pd.DataFrame): Source event log.
            case_id_col (str): Column name that identifies each case.
            pred_series (pd.Series): String series of normalised event labels.

        Returns:
            tuple[list, list[list[str]]]: Ordered case IDs and their corresponding event sequences.

        Raises:
            ValueError: If case_id_col is absent from the DataFrame.
        """
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
        """Convert event sequences into a zero-padded integer token matrix.

        Each sequence is left-padded so that the most recent event occupies the
        last position, matching the training-time sliding-window encoding.

        Args:
            sequences (list[list[str]]): Ordered event sequences per case.
            x_word_dict (dict[str, int]): Vocabulary mapping event label → token index.
            max_case_length (int): Fixed sequence length used during training.

        Returns:
            np.ndarray: Float32 array of shape (n_cases, max_case_length).
        """
        token_x = np.zeros((len(sequences), max_case_length), dtype=np.float32)

        for row_idx, seq in enumerate(sequences):
            row = np.zeros((max_case_length,), dtype=np.float32)
            for event in seq:
                token = x_word_dict.get(event, 0)
                # Slide the window: drop oldest token and append the new one
                row = np.insert(row[1:], row.size - 1, token)
            token_x[row_idx] = row

        return token_x

    def _predict_probabilities(self, model, token_x):
        """Run a forward pass and return softmax class probabilities.

        Args:
            model (tf.keras.Model): Loaded Keras transformer model.
            token_x (np.ndarray): Token matrix of shape (n_samples, max_case_length).

        Returns:
            np.ndarray: Float32 probability matrix of shape (n_samples, n_classes).
        """
        logits = model.predict(token_x, verbose=0)
        probs = tf.nn.softmax(logits, axis=1).numpy()
        return probs

    def _invert_dict(self, data):
        """Swap keys and values in a dictionary.

        Args:
            data (dict): Original mapping (e.g. label → index).

        Returns:
            dict: Inverted mapping (e.g. index → label).
        """
        return {v: k for k, v in data.items()}

    def _load_existing_results(self, detail_csv_path, summary_csv_path, metadata_path):
        """Load previously computed SHAP results from disc.

        Args:
            detail_csv_path (str): Path to shap_top_events_per_case.csv.
            summary_csv_path (str): Path to shap_summary.csv.
            metadata_path (str): Path to shap_metadata.json.

        Returns:
            dict: Keys: rows, summary_rows, explained_cases, metadata,
                  started_at, ended_at, delta_time_sec, delta_time_min.
        """
        detail_df = pd.read_csv(detail_csv_path)
        summary_df = pd.read_csv(summary_csv_path)

        # Normalise NaN values to empty strings for safe template rendering
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
            "started_at": metadata.get('started_at', ''),
            "ended_at": metadata.get('ended_at', ''),
            "delta_time_sec": metadata.get('delta_time_sec', ''),
            "delta_time_min": metadata.get('delta_time_min', ''),
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
        """Extract the top-k most influential events per case and aggregate global event importance.

        For each explained case, SHAP values are retrieved for the predicted class,
        non-padding positions are ranked by absolute value, and the top-k are recorded.
        An accumulator simultaneously tracks the sum of absolute SHAP values per
        unique event name to produce a global per-event mean importance summary.

        Args:
            token_x (np.ndarray): Token matrix (n_cases, max_case_length).
            case_ids (list): Ordered case identifiers.
            predicted_idx (np.ndarray): Integer index of the predicted class per case.
            predicted_labels (list[str]): Human-readable predicted class labels.
            probs (np.ndarray): Softmax probability matrix (n_cases, n_classes).
            shap_values: Raw SHAP output — may be list, 2-D or 3-D array depending on SHAP version.
            inv_x_dict (dict[int, str]): Inverse token vocabulary (index → event label).
            top_k (int): Number of top events to report per case. Defaults to 3.
            mean_abs_decimal_places (int): Rounding precision for mean absolute SHAP. Defaults to 6.

        Returns:
            tuple[list[dict], list[dict]]:
                - detail rows: one dict per case with case_id, predicted_label,
                  predicted_confidence, and event_rank_1 … event_rank_k.
                - summary rows: one dict per unique event with event_name,
                  mean_abs_shap, occurrences_total, occurrences_percent, cases;
                  sorted descending by mean_abs_shap.
        """
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
            # Only consider non-padding positions (token == 0 means padding)
            nonzero_positions = np.where(row_tokens != 0)[0]

            ranked = []
            if len(nonzero_positions) > 0:
                # Sort real event positions by descending absolute SHAP value
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
        """Run or load SHAP explanations for a trained classification model.

        On the first call (or when force_recompute=True) the method loads the model,
        builds token sequences from the source log, runs SHAP KernelExplainer, and
        persists the results. On subsequent calls the cached CSV/JSON files are
        returned immediately. Either way, the log-level aggregate CSVs
        (shap_metrics.csv, shap_common_top10.csv) are always refreshed.

        Args:
            process_name (str): Name of the process folder under the processing directory.
            log_name (str): Name of the event log (also used as the CSV filename).
            prediction_type (str): Prediction task (e.g. 'NextActivity', 'ParticipantSend').
            model_name (str): Model folder name and base name for the .keras file.
            top_k (int): Top events to report per case in the detail CSV. Defaults to 3.
            nsamples (int): Number of perturbation samples for KernelExplainer. Defaults to 100.
            force_recompute (bool): Ignore cached results and recompute from scratch. Defaults to False.

        Returns:
            dict: Contains output_dir, detail_csv_path, summary_csv_path, metadata_path,
                  shap_metrics_path, shap_common_top10_path, rows, summary_rows,
                  explained_cases, model, prediction_type, cached, started_at,
                  ended_at, delta_time_sec, delta_time_min.

        Raises:
            ValueError: If required files are missing, the task is unsupported,
                        or the 'shap' package is not installed.
        """
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

            log_dir, shap_metrics_path, shap_common_top10_path = self._get_log_level_shap_paths(
                processing_dir=processing_dir,
                process_name=process_name,
                log_name=log_name,
            )
            os.makedirs(log_dir, exist_ok=True)
            self._upsert_shap_metrics(
                shap_metrics_path=shap_metrics_path,
                process_name=process_name,
                log_name=log_name,
                prediction_type=prediction_type,
                model_name=model_name,
                summary_rows=cached_result["summary_rows"],
                top_n=10,
            )
            self._refresh_shap_common_top10(
                shap_metrics_path=shap_metrics_path,
                shap_common_top10_path=shap_common_top10_path,
                top_n=10,
            )

            return {
                "output_dir": output_dir,
                "detail_csv_path": detail_csv_path,
                "summary_csv_path": summary_csv_path,
                "metadata_path": metadata_path,
                "shap_metrics_path": shap_metrics_path,
                "shap_common_top10_path": shap_common_top10_path,
                "rows": cached_result["rows"],
                "summary_rows": cached_result["summary_rows"],
                "explained_cases": cached_result["explained_cases"],
                "model": model_name,
                "prediction_type": prediction_type,
                "cached": True,
                "started_at": cached_result["started_at"],
                "ended_at": cached_result["ended_at"],
                "delta_time_sec": cached_result["delta_time_sec"],
                "delta_time_min": cached_result["delta_time_min"],
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

        max_cases = get_shap_max_cases()
        # Background: a small representative subset used to marginalise out features
        background_size = min(30, token_x.shape[0])
        # Limit the number of explained cases to avoid excessively long runtimes
        explain_size = min(max_cases, token_x.shape[0])
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

        log_dir, shap_metrics_path, shap_common_top10_path = self._get_log_level_shap_paths(
            processing_dir=processing_dir,
            process_name=process_name,
            log_name=log_name,
        )
        os.makedirs(log_dir, exist_ok=True)
        self._upsert_shap_metrics(
            shap_metrics_path=shap_metrics_path,
            process_name=process_name,
            log_name=log_name,
            prediction_type=prediction_type,
            model_name=model_name,
            summary_rows=summary_rows,
            top_n=10,
        )
        self._refresh_shap_common_top10(
            shap_metrics_path=shap_metrics_path,
            shap_common_top10_path=shap_common_top10_path,
            top_n=10,
        )

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
            "shap_metrics_path": shap_metrics_path,
            "shap_common_top10_path": shap_common_top10_path,
            "rows": detail_rows,
            "summary_rows": summary_rows,
            "explained_cases": explain_size,
            "model": model_name,
            "prediction_type": prediction_type,
            "cached": False,
            "started_at": metadata["started_at"],
            "ended_at": metadata["ended_at"],
            "delta_time_sec": metadata["delta_time_sec"],
            "delta_time_min": metadata["delta_time_min"],
        }
