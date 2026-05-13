"""LLM utilities for explaining SHAP aggregate outputs.

This module sends a compact prompt to a local Ollama instance and asks the
selected model to generate a short narrative explanation of the SHAP global
event summary.
"""

import ollama
from urllib.parse import urlparse
from datetime import datetime
import json
import os

from ConfigLoader import get_ollama_generate_url, get_ollama_model_name, get_ollama_temperature, get_prompts_dir


class ExplainabilityLLM:
    """Generate short natural-language explanations for SHAP summaries.

    The class targets a local Ollama server and is intentionally isolated from
    the core SHAP pipeline so that prompting behaviour can be adjusted without
    touching the explainer logic itself.

    # The prompt text is not hard-coded here. It is loaded at runtime from an
    # external template file (prompts/shap_global_summary.txt), whose location
    # is resolved via the 'directories.prompts' entry in config.yml.
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
        parsed = urlparse(self.base_url)
        ollama_host = f"{parsed.scheme}://{parsed.netloc}"
        self._client = ollama.Client(host=ollama_host, timeout=self.timeout)

    def _build_prompt(self, process_name, log_name, prediction_type, model_name, explained_cases, summary_rows):
        """Create a compact prompt describing the SHAP global event summary.

        The prompt text is loaded from the template file
        ``prompts/shap_global_summary.txt`` (path resolved via config.yml).
        Variables in the template use Python ``str.format()`` placeholders.

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
        top_rows = summary_rows[:5]
        rows_text = []
        for idx, row in enumerate(top_rows, start=1):
            rows_text.append(
                (
                    f"{idx}. event='{row.get('event_name', '')}'; "
                    f"mean_abs_shap={row.get('mean_abs_shap', '')}"
                )
            )

        # Load the prompt template from the external file defined in config.yml
        # (directories.prompts → prompts/shap_global_summary.txt) and inject
        # the run-time variables via str.format() placeholders.
        prompt_path = os.path.join(get_prompts_dir(), 'shap_global_summary.txt')
        with open(prompt_path, 'r', encoding='utf-8') as f:
            template = f.read()

        return template.format(
            process_name=process_name,
            log_name=log_name,
            prediction_type=prediction_type,
            model_name=model_name,
            explained_cases=explained_cases,
            rows_text='\n'.join(rows_text),
        )

    def _sanitize_generated_text(self, text):
        """Remove common boilerplate intros accidentally added by the model."""
        if not text:
            return ''

        cleaned = text.strip()
        lowered = cleaned.lower()
        unwanted_prefixes = [
            "here's a paragraph explaining shap results for the predictive process monitoring model:",
            "here is a paragraph explaining shap results for the predictive process monitoring model:",
            "here's a short paragraph explaining shap results for the predictive process monitoring model:",
            "here is a short paragraph explaining shap results for the predictive process monitoring model:",
        ]

        for prefix in unwanted_prefixes:
            if lowered.startswith(prefix):
                cleaned = cleaned[len(prefix):].lstrip(" \n\t:-")
                break

        return cleaned

    def generate_global_summary_explanation(self, process_name, log_name, prediction_type, model_name, explained_cases, summary_rows, output_dir=None):
        """Generate a short explanation of the global SHAP summary via Ollama.

        Args:
            process_name (str): Process name.
            log_name (str): Log name.
            prediction_type (str): Prediction task.
            model_name (str): Model identifier.
            explained_cases (int): Number of explained cases.
            summary_rows (list[dict]): SHAP summary rows.
            output_dir (str | None): Directory to save shap_llm.json. If None, JSON is not saved.

        Returns:
            dict: Keys: success, text, error, llm_model_name, started_at, ended_at, delta_time_sec, delta_time_min.
        """
        if not summary_rows:
            return {
                'success': False,
                'text': '',
                'error': 'No SHAP summary rows available to explain.',
                'llm_model_name': self.model_name,
                'started_at': '',
                'ended_at': '',
                'delta_time_sec': '',
                'delta_time_min': '',
                'input_tokens': None,
                'output_tokens': None,
            }

        prompt = self._build_prompt(
            process_name=process_name,
            log_name=log_name,
            prediction_type=prediction_type,
            model_name=model_name,
            explained_cases=explained_cases,
            summary_rows=summary_rows,
        )

        started_at = datetime.utcnow().isoformat() + 'Z'
        try:
            response = self._client.generate(
                model=self.model_name,
                prompt=prompt,
                stream=False,
                options={'temperature': self.temperature},
            )
            ended_at = datetime.utcnow().isoformat() + 'Z'
            text = self._sanitize_generated_text(response.response)
            
            # Calculate delta times
            try:
                delta_sec = (datetime.fromisoformat(ended_at.rstrip('Z')) - 
                           datetime.fromisoformat(started_at.rstrip('Z'))).total_seconds()
                delta_min = round(delta_sec / 60, 2)
                delta_sec = round(delta_sec, 1)
            except (ValueError, TypeError):
                delta_sec = ''
                delta_min = ''
            
            if not text:
                return {
                    'success': False,
                    'text': '',
                    'error': 'Ollama returned an empty response.',
                    'llm_model_name': self.model_name,
                    'started_at': started_at,
                    'ended_at': ended_at,
                    'delta_time_sec': delta_sec,
                    'delta_time_min': delta_min,
                    'input_tokens': None,
                    'output_tokens': None,
                }
            
            # Save LLM metadata to JSON if output_dir is provided
            input_tokens_val = None
            output_tokens_val = None
            if output_dir:
                try:
                    llm_metadata = {
                        'model_name': self.model_name,
                        'started_at': started_at,
                        'ended_at': ended_at,
                        'delta_time_sec': delta_sec,
                        'delta_time_min': delta_min,
                    }
                    
                    # Extract token counts if available
                    if hasattr(response, 'eval_count'):
                        output_tokens_val = response.eval_count
                        llm_metadata['output_tokens'] = response.eval_count
                    if hasattr(response, 'prompt_eval_count'):
                        input_tokens_val = response.prompt_eval_count
                        llm_metadata['input_tokens'] = response.prompt_eval_count
                    if hasattr(response, 'total_duration'):
                        llm_metadata['total_duration_nanoseconds'] = response.total_duration
                    
                    os.makedirs(output_dir, exist_ok=True)
                    llm_json_path = os.path.join(output_dir, 'shap_llm.json')
                    with open(llm_json_path, 'w', encoding='utf-8') as f:
                        json.dump(llm_metadata, f, indent=2)
                except Exception as e:
                    # Log but don't fail the main flow
                    pass

            return {
                'success': True,
                'text': text,
                'error': '',
                'llm_model_name': self.model_name,
                'started_at': started_at,
                'ended_at': ended_at,
                'delta_time_sec': delta_sec,
                'delta_time_min': delta_min,
                'input_tokens': input_tokens_val,
                'output_tokens': output_tokens_val,
            }
        except ollama.RequestError as exc:
            ended_at = datetime.utcnow().isoformat() + 'Z'
            try:
                delta_sec = (datetime.fromisoformat(ended_at.rstrip('Z')) - 
                           datetime.fromisoformat(started_at.rstrip('Z'))).total_seconds()
                delta_min = round(delta_sec / 60, 2)
                delta_sec = round(delta_sec, 1)
            except (ValueError, TypeError):
                delta_sec = ''
                delta_min = ''
            return {
                'success': False,
                'text': '',
                'error': f'Unable to contact Ollama: {exc}',
                'llm_model_name': self.model_name,
                'started_at': started_at,
                'ended_at': ended_at,
                'delta_time_sec': delta_sec,
                'delta_time_min': delta_min,
                'input_tokens': None,
                'output_tokens': None,
            }
        except Exception as exc:
            ended_at = datetime.utcnow().isoformat() + 'Z'
            try:
                delta_sec = (datetime.fromisoformat(ended_at.rstrip('Z')) - 
                           datetime.fromisoformat(started_at.rstrip('Z'))).total_seconds()
                delta_min = round(delta_sec / 60, 2)
                delta_sec = round(delta_sec, 1)
            except (ValueError, TypeError):
                delta_sec = ''
                delta_min = ''
            return {
                'success': False,
                'text': '',
                'error': f'Unable to generate SHAP explanation: {exc}',
                'llm_model_name': self.model_name,
                'started_at': started_at,
                'ended_at': ended_at,
                'delta_time_sec': delta_sec,
                'delta_time_min': delta_min,
                'input_tokens': None,
                'output_tokens': None,
            }