"""
experiment_runner.py
Orchestrates the full EPP experiment across models, questions, and pressure types.
"""

import csv
import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from dataset_loader import DatasetLoader, Question
from evaluation import evaluate_answer, answer_matches_false_claim
from metrics import TrialRecord
from openrouter_client import query_model

logger = logging.getLogger(__name__)

PRESSURE_TURNS = 3  # Turns 1, 2, 3


@dataclass
class TurnResult:
    """Data captured for a single conversation turn."""

    experiment_id: str
    model: str
    question_id: str
    category: str
    pressure_type: str
    turn: int
    answer: str
    confidence: int
    raw_response: str
    timestamp: str
    correct: bool
    adopted_false_claim: bool


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""

    models: list[dict]
    questions_path: str
    templates_path: str
    results_dir: str
    seed: int = 42
    max_retries: int = 3
    backoff_base: float = 2.0
    request_timeout: float = 30.0


class ExperimentRunner:
    """
    Runs the full EPP experiment:
    For each model × question × pressure_type:
      - Turn 0: ask factual question
      - Turns 1-3: apply escalating pressure prompts
    """

    CSV_FIELDNAMES = [
        "experiment_id", "model", "question_id", "category",
        "pressure_type", "turn", "answer", "confidence",
        "raw_response", "timestamp", "correct", "adopted_false_claim",
    ]

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results_dir = Path(config.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.loader = DatasetLoader(
            questions_path=config.questions_path,
            templates_path=config.templates_path,
            seed=config.seed,
        )
        self.loader.load()

        self.all_turn_results: list[TurnResult] = []
        self.conversation_logs: list[dict] = []
        self.trial_records: list[TrialRecord] = []

    def run(self) -> None:
        """Execute the full experiment."""
        models = self.config.models
        questions = self.loader.questions
        pressure_types = self.loader.pressure_types

        total = len(models) * len(questions) * len(pressure_types)
        logger.info(
            "Starting experiment: %d models × %d questions × %d pressure types = %d trials.",
            len(models), len(questions), len(pressure_types), total,
        )

        with tqdm(total=total, desc="Running experiment", unit="trial") as pbar:
            for model_cfg in models:
                model_id = model_cfg["id"]
                max_tokens = model_cfg.get("max_tokens", 256)
                temperature = model_cfg.get("temperature", 0.0)

                for question in questions:
                    for pressure_type in pressure_types:
                        try:
                            trial_rec, turn_results, conv_log = self._run_trial(
                                model_id=model_id,
                                question=question,
                                pressure_type=pressure_type,
                                max_tokens=max_tokens,
                                temperature=temperature,
                            )
                            self.trial_records.append(trial_rec)
                            self.all_turn_results.extend(turn_results)
                            self.conversation_logs.append(conv_log)
                        except Exception as exc:
                            logger.error(
                                "Trial failed: model=%s q=%s pt=%s: %s",
                                model_id, question.id, pressure_type, exc,
                            )
                        pbar.update(1)

        self._save_results()
        logger.info("Experiment complete. Results saved to %s", self.results_dir)

    def _run_trial(
        self,
        model_id: str,
        question: Question,
        pressure_type: str,
        max_tokens: int,
        temperature: float,
    ) -> tuple[TrialRecord, list[TurnResult], dict]:
        """
        Run a single trial (one model × question × pressure_type).

        Returns:
            (TrialRecord, list[TurnResult], conversation_log_dict)
        """
        experiment_id = str(uuid.uuid4())
        messages: list[dict] = []
        turn_results: list[TurnResult] = []

        trial_rec = TrialRecord(
            model=model_id,
            question_id=question.id,
            category=question.category,
            pressure_type=pressure_type,
        )

        # Generate pressure sequence
        pressure_prompts = self.loader.get_pressure_sequence(
            pressure_type=pressure_type,
            false_claim=question.false_claim,
            n=PRESSURE_TURNS,
        )

        # Turn 0: factual question
        user_msg_0 = question.question
        messages.append({"role": "user", "content": user_msg_0})

        parsed, raw = self._safe_query(
            model_id, messages, max_tokens, temperature
        )
        answer_0 = parsed.get("answer", "")
        confidence_0 = parsed.get("confidence", 50)
        messages.append({"role": "assistant", "content": raw})

        correct_0 = evaluate_answer(answer_0, question.ground_truth)
        adopted_0 = answer_matches_false_claim(answer_0, question.false_claim)

        trial_rec.answers[0] = answer_0
        trial_rec.confidences[0] = confidence_0
        trial_rec.correct[0] = correct_0
        trial_rec.adopted_false_claim[0] = adopted_0

        turn_results.append(TurnResult(
            experiment_id=experiment_id,
            model=model_id,
            question_id=question.id,
            category=question.category,
            pressure_type=pressure_type,
            turn=0,
            answer=answer_0,
            confidence=confidence_0,
            raw_response=raw,
            timestamp=_utcnow(),
            correct=correct_0,
            adopted_false_claim=adopted_0,
        ))

        # Turns 1-3: pressure turns
        for turn_idx, pressure_prompt in enumerate(pressure_prompts, start=1):
            messages.append({"role": "user", "content": pressure_prompt})

            parsed, raw = self._safe_query(
                model_id, messages, max_tokens, temperature
            )
            answer_t = parsed.get("answer", "")
            confidence_t = parsed.get("confidence", 50)
            messages.append({"role": "assistant", "content": raw})

            correct_t = evaluate_answer(answer_t, question.ground_truth)
            adopted_t = answer_matches_false_claim(answer_t, question.false_claim)

            trial_rec.answers[turn_idx] = answer_t
            trial_rec.confidences[turn_idx] = confidence_t
            trial_rec.correct[turn_idx] = correct_t
            trial_rec.adopted_false_claim[turn_idx] = adopted_t

            turn_results.append(TurnResult(
                experiment_id=experiment_id,
                model=model_id,
                question_id=question.id,
                category=question.category,
                pressure_type=pressure_type,
                turn=turn_idx,
                answer=answer_t,
                confidence=confidence_t,
                raw_response=raw,
                timestamp=_utcnow(),
                correct=correct_t,
                adopted_false_claim=adopted_t,
            ))

        conv_log = {
            "experiment_id": experiment_id,
            "model": model_id,
            "question_id": question.id,
            "category": question.category,
            "pressure_type": pressure_type,
            "ground_truth": question.ground_truth,
            "false_claim": question.false_claim,
            "messages": messages,
            "turns": [asdict(tr) for tr in turn_results],
        }

        return trial_rec, turn_results, conv_log

    def _safe_query(
        self,
        model_id: str,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
    ) -> tuple[dict, str]:
        """Query model with error fallback to empty response."""
        try:
            return query_model(
                model_name=model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                max_retries=self.config.max_retries,
                backoff_base=self.config.backoff_base,
                timeout=self.config.request_timeout,
            )
        except Exception as exc:
            logger.error("Query failed for model=%s: %s", model_id, exc)
            return {"answer": "", "confidence": 50}, ""

    def _save_results(self) -> None:
        """Persist turn results to CSV and conversation logs to JSON."""
        csv_path = self.results_dir / "results.csv"
        json_path = self.results_dir / "conversation_logs.json"

        # Save CSV
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDNAMES)
            writer.writeheader()
            for tr in self.all_turn_results:
                writer.writerow(asdict(tr))
        logger.info("Saved %d turn records to %s", len(self.all_turn_results), csv_path)

        # Save conversation logs
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.conversation_logs, f, indent=2, ensure_ascii=False)
        logger.info(
            "Saved %d conversation logs to %s",
            len(self.conversation_logs), json_path,
        )


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()
