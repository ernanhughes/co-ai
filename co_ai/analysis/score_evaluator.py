from co_ai.models.score_dimension import ScoreDimensionORM
from sqlalchemy.orm import Session
import yaml
from pathlib import Path
from jinja2 import Template
import re

class ScoreEvaluator:
    def __init__(self, dimensions, prompt_loader=None, agent_config=None):
        self.dimensions = dimensions
        self.prompt_loader = prompt_loader
        self.agent_config = agent_config or {}

    @classmethod
    def from_db(cls, session: Session, stage: str, prompt_loader=None, agent_config=None):
        rows = session.query(ScoreDimensionORM).filter_by(stage=stage).all()
        dimensions = [
            {
                "name": row.name,
                "prompt_template": row.prompt_template,
                "weight": row.weight,
                "parser": cls.get_parser(row.extra_data or {}),
                "file": row.extra_data.get("file") if row.extra_data else None
            }
            for row in rows
        ]
        return cls(dimensions, prompt_loader=prompt_loader, agent_config=agent_config)

    @classmethod
    def from_file(cls, filepath: str, prompt_loader=None, agent_config=None):
        with open(Path(filepath), 'r') as f:
            data = yaml.safe_load(f)
        dimensions = [
            {
                "name": d["name"],
                "file": d.get("file"),
                "prompt_template": d.get("prompt_template"),
                "weight": d.get("weight", 1.0),
                "parser": cls.get_parser(d.get("extra_data", {}))
            }
            for d in data["dimensions"]
        ]
        return cls(dimensions, prompt_loader=prompt_loader, agent_config=agent_config)

    @staticmethod
    def get_parser(extra_data):
        parser_type = extra_data.get("parser", "numeric")
        if parser_type == "numeric":
            return lambda r: ScoreEvaluator.extract_score_from_last_line(r)
        return lambda r: 0.0

    @staticmethod
    def extract_score_from_last_line(response: str) -> float:
        """
        Looks for a line ending with 'score: <number>' (case-insensitive).
        """
        lines = response.strip().splitlines()
        for line in reversed(lines):
            match = re.search(r'score:\s*(\d+(\.\d+)?)', line.strip(), re.IGNORECASE)
            if match:
                return float(match.group(1))
        return 0.0

    def evaluate(self, goal: str, hypothesis: str, context: dict = {}, llm_fn=None):
        if llm_fn is None:
            raise ValueError("You must pass a call_llm function (e.g., agent.call_llm) to ScoreEvaluator.evaluate")

        results = {}
        for dim in self.dimensions:
            if self.prompt_loader and dim.get("file"):
                prompt = self.prompt_loader.from_file(
                    file_name=dim["file"],
                    config=self.agent_config,
                    context={"goal": goal, "hypothesis": hypothesis, **context}
                )
            else:
                prompt = Template(dim["prompt_template"]).render(goal=goal, hypothesis=hypothesis, **context)

            response = llm_fn(prompt, context=context)
            print (f"Evaluating dimension: {dim['name']}, response: {response}")
            score = dim["parser"](response)
            results[dim["name"]] = {
                "score": score,
                "rationale": response,
                "weight": dim["weight"]
            }
        return results
