# co_ai/supervisor.py

import json
import os

import hydra
from omegaconf import DictConfig, OmegaConf

from co_ai.constants import (NAME, PROMPT_DIR, RUN_ID, SAVE_CONTEXT,
                             SKIP_IF_COMPLETED, STAGE)
from co_ai.logs.json_logger import JSONLogger
from co_ai.memory import MemoryTool
from co_ai.reports import ReportFormatter


class PipelineStage:
    def __init__(self, name: str, config: dict, stage_dict: dict):
        self.name = name
        self.cls = config.get("cls", "")
        self.enabled = config.get("enabled", True)
        self.iterations = config.get("iterations", 1)
        self.stage_dict = stage_dict


class Supervisor:
    def __init__(self, cfg, memory=None, logger=None):
        self.cfg = cfg
        self.memory = memory or MemoryTool(cfg=cfg.db, logger=logger)
        self.logger = logger or JSONLogger(log_path=cfg.logger.log_path)
        self.logger.log("SupervisorInit", {"cfg": cfg})

        # Parse pipeline stages from config
        self.pipeline_stages = self._parse_pipeline_stages(cfg.pipeline.stages)

    def _parse_pipeline_stages(
        self, stage_configs: list[dict[str, any]]
    ) -> list[PipelineStage]:
        """Parse and validate pipeline stages from config."""
        stages = []
        for stage_config in stage_configs:
            name = stage_config.name
            if not stage_config.enabled:
                print(f"Skipping disabled stage: {name}")
                continue
            stage_dict = self.cfg.agents[name]
            print(f"Stage dict: {stage_dict}")
            stages.append(PipelineStage(name, stage_config, stage_dict))
        return stages

    async def run_pipeline_config(self, input_data: dict) -> dict:
        """
        Run all stages defined in config.
        Each stage loads its class dynamically via hydra.utils.get_class()
        """
        self.logger.log("PipelineStart", input_data)
        input_file = input_data.get("input_file", self.cfg.get("input_file", None))
        # Handle JSONL file for batch input
        if input_file and os.path.exists(input_file):
            self.logger.log("BatchProcessingStart", {"file": input_file})
            with open(input_file, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    goal = json.loads(line)
                    run_id = goal.get("id", f"goal_{i}")
                    context = {
                        "goal": goal,
                        "run_id": run_id,
                        "prompt_dir": self.cfg.paths.prompts,
                    }
                    try:
                        await self._run_pipeline_stages(context)
                    except Exception as e:
                        self.logger.log(
                            "BatchItemFailed",
                            {"index": i, "run_id": run_id, "error": str(e)},
                        )
            self.logger.log("BatchProcessingComplete", {"file": input_file})
            return {"status": "completed_batch", "input_file": input_file}

        # Fallback to single goal execution
        context = input_data.copy()
        context[PROMPT_DIR] = self.cfg.paths.prompts
        context = await self.maybe_adjust_pipeline(context)
        return await self._run_pipeline_stages(context)

    def _parse_pipeline_stages_from_list(
        self, stage_names: list[str]
    ) -> list[PipelineStage]:
        return [
            PipelineStage(name, self.cfg.pipeline.stages[name], self.cfg.agents[name])
            for name in stage_names
            if name in self.cfg.agents
        ]

    async def _run_pipeline_stages(self, context: dict) -> dict:
        for stage in self.pipeline_stages:
            if not stage.enabled:
                self.logger.log(
                    "PipelineStageSkipped",
                    {STAGE: stage.name, "reason": "disabled_in_config"},
                )
                continue

            cls = hydra.utils.get_class(stage.cls)
            stage_dict = OmegaConf.to_container(stage.stage_dict, resolve=True)

            saved_context = self.load_context(stage_dict, run_id=context.get(RUN_ID))
            if saved_context:
                self.logger.log(
                    "PipelineStageSkipped",
                    {STAGE: stage.name, "reason": "context_loaded"},
                )
                context = {**context, **saved_context}
                continue

            agent = cls(cfg=stage_dict, memory=self.memory, logger=self.logger)
            self.logger.log("PipelineStageStart", {STAGE: stage.name})

            for i in range(stage.iterations):
                self.logger.log(
                    "PipelineIterationStart", {STAGE: stage.name, "iteration": i + 1}
                )
                context = await agent.run(context)
                self.logger.log(
                    "PipelineIterationEnd", {STAGE: stage.name, "iteration": i + 1}
                )

            self.save_context(stage_dict, context)
            self.logger.log("PipelineStageEnd", {STAGE: stage.name})
            self.logger.log("ContextAfterStage", {STAGE: stage.name, "context_keys": list(context.keys())})

        return context


    def generate_report(self, context: dict[str, any], run_id: str) -> str:
        """Generate a report based on the pipeline context."""
        formatter = ReportFormatter(self.cfg.report.path)
        report = formatter.format_report(context)
        self.memory.report.log(
            run_id, str(context.get("goal")), report, self.cfg.report.path
        )
        self.logger.log(
            "ReportGenerated", {RUN_ID: run_id, "report_snippet": report[:100]}
        )
        return report

    def save_context(self, cfg: DictConfig, context: dict):
        if self.memory and cfg.get(SAVE_CONTEXT, False):
            run_id = context.get(RUN_ID)
            name = cfg.get(NAME, "NoAgentNameInConfig")
            self.memory.context.save(run_id, name, context, cfg)
            self.logger.log(
                "ContextSaved",
                {NAME: name, RUN_ID: run_id, "context_keys": list(context.keys())},
            )

    def load_context(self, cfg: DictConfig, run_id:str):
        if self.memory and cfg.get(SKIP_IF_COMPLETED, False):
            name = cfg.get(NAME, None)
            if name and self.memory.context.has_completed(run_id, name):
                saved_context = self.memory.context.load(run_id, name)
                if saved_context:
                    self.logger.log("ContextLoaded", {RUN_ID: run_id, NAME: name})
                    return saved_context
        return None

    async def maybe_adjust_pipeline(self, context: dict) -> dict:
        """
        Optionally run LookaheadAgent before pipeline stages to revise or select the pipeline.
        """
        if not self.cfg.get("dynamic", {}).get("enabled", False):
            return context  # Skip if not enabled

        lookahead_cfg = OmegaConf.to_container(self.cfg.dynamic, resolve=True)
        stage_dict =  OmegaConf.to_container(self.cfg.agents.lookahead, resolve=True)
        agent_cls = hydra.utils.get_class(lookahead_cfg["cls"])
        lookahead_agent = agent_cls(
            cfg=stage_dict, memory=self.memory, logger=self.logger
        )

        self.logger.log("LookaheadStart", {"goal": context.get("goal", {})})

        # Add current pipeline so LookaheadAgent can reflect on it
        context["pipeline"] = [stage.name for stage in self.pipeline_stages]
        updated_context = await lookahead_agent.run(context)

        # Optional: if lookahead returned a revised pipeline
        if "suggested_pipeline" in updated_context:
            suggested = updated_context["suggested_pipeline"]
            self.logger.log("PipelineUpdatedByLookahead", {
                "original": [stage.name for stage in self.pipeline_stages],
                "suggested": suggested
            })
            self.pipeline_stages = self._parse_pipeline_stages_from_list(suggested)

        return updated_context
