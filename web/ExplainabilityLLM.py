"""LLM utilities for explaining SHAP aggregate outputs.

This module sends a compact prompt to a local Ollama instance and asks the
selected model to generate a short narrative explanation of the SHAP global
event summary.
"""

import json
from urllib import error, request

from ConfigLoader import get_ollama_generate_url, get_ollama_model_name, get_ollama_temperature


class ExplainabilityLLM:
    """Generate short natural-language explanations for SHAP summaries.

    The class targets a local Ollama server and is intentionally isolated from
    the core SHAP pipeline so that prompting behaviour can be adjusted without
    touching the explainer logic itself.
    """

    def __init__(self, base_url=None, model_name=None, temperature=None, timeout=30):
        """Initialise the local LLM client.

        Args:
            base_url (str | None): Ollama generate endpoint. If None, load from config.yml.
            model_name (str | None): Ollama model name. If None, load from config.yml.
            temperature (float | None): Ollama temperature. If None, load from config.yml.
            timeout (int): Request timeout in seconds.
        """
        self.base_url = base_url or get_ollama_generate_url()
        self.model_name = model_name or get_ollama_model_name()
        self.temperature = get_ollama_temperature() if temperature is None else float(temperature)
        self.timeout = timeout

    def _build_prompt(self, process_name, log_name, prediction_type, model_name, explained_cases, summary_rows):
        """Create a compact prompt describing the SHAP global event summary.

        Args:
            process_name (str): Process name.
            log_name (str): Log name.
            prediction_type (str): Prediction task.
            model_name (str): Model identifier.
            explained_cases (int): Number of explained cases.
            summary_rows (list[dict]): SHAP event summary rows.

        Returns:
            str: Prompt ready to be sent to the LLM.
        """
        top_rows = summary_rows[:10]
        rows_text = []
        for idx, row in enumerate(top_rows, start=1):
            rows_text.append(
                (
                    f"{idx}. event='{row.get('event_name', '')}'; "
                    f"mean_abs_shap={row.get('mean_abs_shap', '')}; "
                    f"occurrences_total={row.get('occurrences_total', '')}; "
                    f"occurrences_percent={row.get('occurrences_percent', '')}; "
                    f"cases={row.get('cases', '')}"
                )
            )

        return (
            "You are explaining SHAP results for a predictive process monitoring model. "
            "Write one short paragraph in British English for a web interface. "
            "Keep it factual, readable, and non-technical where possible. "
            "Mention the most influential events, whether they are recurrent or rare, "
            "and what mean_abs_shap and occurrences_percent imply. "
            "Do not invent facts, causes, or relationships that are not explicitly supported by the input. "
            "Whenever you mention an event name or any value taken from the event log or SHAP summary, wrap it in single quotes. "
            "Do not use bullet points. Do not mention being an AI.\n\n"
            f"Process: {process_name}\n"
            f"Log: {log_name}\n"
            f"Prediction type: {prediction_type}\n"
            f"Model: {model_name}\n"
            f"Explained cases: {explained_cases}\n\n"
            "Top events from shap_summary.csv:\n"
            + "\n".join(rows_text)
        )

    def generate_global_summary_explanation(self, process_name, log_name, prediction_type, model_name, explained_cases, summary_rows):
        """Generate a short explanation of the global SHAP summary via Ollama.

        Args:
            process_name (str): Process name.
            log_name (str): Log name.
            prediction_type (str): Prediction task.
            model_name (str): Model identifier.
            explained_cases (int): Number of explained cases.
            summary_rows (list[dict]): SHAP summary rows.

        Returns:
            dict: Keys: success, text, error.
        """
        if not summary_rows:
            return {
                'success': False,
                'text': '',
                'error': 'No SHAP summary rows available to explain.',
            }

        prompt = self._build_prompt(
            process_name=process_name,
            log_name=log_name,
            prediction_type=prediction_type,
            model_name=model_name,
            explained_cases=explained_cases,
            summary_rows=summary_rows,
        )

        payload = json.dumps(
            {
                'model': self.model_name,
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': self.temperature,
                },
            }
        ).encode('utf-8')

        http_request = request.Request(
            self.base_url,
            data=payload,
            headers={'Content-Type': 'application/json'},
            method='POST',
        )

        try:
            with request.urlopen(http_request, timeout=self.timeout) as response:
                body = response.read().decode('utf-8')
                data = json.loads(body)
                text = data.get('response', '').strip()
                if not text:
                    return {
                        'success': False,
                        'text': '',
                        'error': 'Ollama returned an empty response.',
                    }

                return {
                    'success': True,
                    'text': text,
                    'error': '',
                }
        except error.URLError as exc:
            return {
                'success': False,
                'text': '',
                'error': f'Unable to contact Ollama: {exc}',
            }
        except Exception as exc:
            return {
                'success': False,
                'text': '',
                'error': f'Unable to generate SHAP explanation: {exc}',
            }