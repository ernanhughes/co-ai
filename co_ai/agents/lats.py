import requests
import json
from co_ai.agents.base_mcts import BaseMCTSAgent, LATSNode
from co_ai.agents.base import BaseAgent
import re
from co_ai.agents.mixins.scoring_mixin import ScoringMixin
from co_ai.utils.timing import time_function
from co_ai.constants import GOAL


class LATSAgent(BaseMCTSAgent, ScoringMixin, BaseAgent):
    def __init__(self, cfg: dict, memory=None, logger=None):
        BaseMCTSAgent.__init__(self, cfg)
        ScoringMixin.__init__(self, cfg, memory, logger)
        BaseAgent.__init__(self, cfg, memory, logger)

        self.max_steps = self.cfg.get("max_steps", 10)
        self.branching_factor = self.cfg.get("branching_factor", 3)

    async def run(self, context: dict):
        """Main MCTS loop"""
        # Initialize root node
        goal = context["goal"]
        self.root = self.create_node(goal["goal_text"], [])
        
        # Run simulations
        for _ in range(self.num_simulations):
            node = self._select(self.root)
            if not self._is_terminal(node):
                await self.expand(node, context)
            reward = self._simulate(node, context)
            self._backpropagate(node, reward)
        
        # Return best path
        best_child = self._best_uct(self.root)
        final_score, best_node, best_trace = self._get_best_score(self.root)

        context["best_trace"] = best_trace
        context["best_score"] = final_score

        # Log final result
        self.logger.log("FinalResult", {
            "trace": best_trace,
            "score": final_score,
            "dimension_scores": best_node.dimension_scores
        })

        # Get timing logs
        timing_logs = self.logger.get_logs_by_type("FunctionTiming")
        for log in timing_logs:
            print(f"{log['timestamp']} - {log['data']['function']}: {log['data']['duration_ms']}ms")
        return {
            "trace": best_child.trace,
            "score": best_child.score,
            "dimension_scores": best_child.dimension_scores
        }

    async def expand(self, node: LATSNode, context: dict):
        # Build prompt with context
        merged = {
            **context,
            "state": node.state["current"] if isinstance(node.state, dict) else node.state,
            "trace": node.trace,
            "mode": "reason",
            "branching_factor": self.branching_factor
        }
        
        # Get similar hypotheses
        proximity_context = await self._run_proximity(context)
        merged["similar_hypotheses"] = proximity_context.get("most_similar", "")
        
        # Load promptMost similar
        prompt_text = self.prompt_loader.load_prompt(self.cfg, merged)
        response = self.call_llm(prompt_text, merged)
        
        # Parse completions
        completions = self._parse_thought_sections(response)
        if not completions:
            completions = self._fallback_parsing(response)
        
        # Create child nodes
        children = []
        for i, comp in enumerate(completions):
            new_state = self._update_state(node.state, comp["action"])
            new_trace = node.trace + [comp["step"]]
            
            # Score hypothesis
            hyp = {
                "text": comp["action"],
                "id": f"hyp_{node.id}_{i}",
                "goal_id": context[GOAL]["id"]
            }
            
            score_result = self._score_hypothesis(
                hyp, 
                {"goal": {"goal_text": node.state.goal}},
                metrics="lats_node"
            )
            
            # Create child node
            child = self.create_node(new_state, new_trace, parent=node)
            child["score"] = score_result["score"]
            child["dimension_scores"] = score_result["scores"]
            child["action"] = comp["action"]
            children.append(child)
        
        self.children[id(node)] = children
        return children[0] if children else node


    async def _run_proximity(self, context:dict):
        from co_ai.agents.proximity import ProximityAgent
        proximity_agent = ProximityAgent(
            self.cfg.get("proximity", {}), memory=self.memory, logger=self.logger
        )
        return await proximity_agent.run(context)

    # Set up LATSComponent
    @time_function(logger=None)
    def _generate_completions(self, node: LATSNode, context: dict):
        """Generate multiple reasoning steps using Ollama"""
        merged = {
            **context,
            "state": node.state,
            "trace": node.trace,
            "mode": "reason",
            "branching_factor": self.branching_factor
        }
        
        prompt = self.prompt_loader.load_prompt(self.cfg, merged)
        response = self.call_llm(prompt, merged)

        completions = self._parse_thought_sections(response)

        if not completions:
            self.logger.log(
                "FallbackParsing", {"prompt": prompt[:200], "response": response[:200]}
            )
            completions = self._fallback_parsing(response)

        return completions

    def _parse_thought_sections(self, response: str):
        """
        Extract thoughts using markdown-style sections
        Format: ### Thought N\n**Rationale**: ...\n**Action**: ...
        """
        thought_pattern = r"\s*Thought\s*\d+[\s\S]*?\n\*\*Rationale\*\*:\s*(.*?)\n\*\*Action\*\*:\s*(?=\n|$)(.*?)(?=\n###|\Z)"
        matches = re.findall(thought_pattern, response.strip(), re.DOTALL)

        # Return list of actions (or thoughts)
        return [
            {
                "step": f"Thought {i + 1}: {action.strip()}",
                "rationale": rationale.strip(),
                "action": action.strip(),
            }
            for i, (rationale, action) in enumerate(matches)
        ][: self.branching_factor]

    def _fallback_parsing(self, response: str):
        """Fallback parser for raw text responses"""
        # Match Thought N: patterns
        thought_pattern = r"[Tt]hought\s*\d+:\s*(.*?)(?=\n(?:[Tt]hought\s*\d+:\s|\Z))"
        matches = re.findall(thought_pattern, response.strip(), re.DOTALL)

        # Clean and extract
        completions = [{"step": match.strip()} for match in matches if match.strip()]

        # If no thoughts found, use entire response as one hypothesis
        if not completions and response.strip():
            completions = [{"step": response.strip()}]

        return completions[:self.branching_factor]

    @time_function(logger=None)
    def _fallback_parsing(self, response: str):
        """Fallback parser for raw text responses"""
        thought_pattern = r"(?:[Tt]hought\s*\d+|[Aa]ction\s*\d+|[-•])\s*(.*?)(?=\n(?:[Tt]hought\s*\d+|[Aa]ction\s*\d+|[-•])|\Z)"
        matches = re.findall(thought_pattern, response.strip(), re.DOTALL)
        return [match[-1].strip() for match in matches][:self.branching_factor]

    @time_function(logger=None)
    def _update_state(self, state, action: str):
        """Update state with new action"""
        if isinstance(state, dict):
            new_state = state.copy()
            new_state["current"] = f"{state['current']}\n{action}"
            new_state["trace"] = state.get("trace", []) + [action]
            return new_state
       
        # Fallback: string-based state → wrap into dict
        return {
            "goal": state,
            "current": f"{state}\n{action}",
            "trace": [action]
        }

    @time_function(logger=None)
    def _score_hypothesis(self, hypothesis: dict, context: dict, metrics: str = "lats_node"):
        """Use dimensional scoring system"""
        return super().score_hypothesis(hypothesis, context, metrics)

    @time_function(logger=None)
    def _simulate(self, node: LATSNode, context: dict):
        """Simulate until terminal state"""
        current = node
        
        while not self._is_terminal(current) and len(current.trace) < self.max_depth:
            prompt = self._build_prompt(current, context, mode="simulate")
            response = self.call_llm(prompt, context)
            completions = self._fallback_parsing(response)
            
            if completions:
                action = completions[0]
                new_state = self._update_state(current.state, action)
                current = self.create_node(new_state, current.trace + [action], parent=current)
        
        # Evaluate final node
        return self._get_value(current)

    @time_function(logger=None)
    def _get_value(self, node: LATSNode):
        """Hybrid value function using LM + self-consistency"""
        if self.cfg.get("use_environment", False):
            obs = self.env.step(node.state)
            return obs['reward']
        
        # Safely extract goal from state
        if isinstance(node.state, dict):
            goal_text = node.state.get("goal", "Unknown goal")
        else:
            goal_text = str(node.state)
        
        score_result = self._score_hypothesis(
            {"text": "\n".join(node.trace)},
            {"goal": {"goal_text": goal_text}},
            metrics="lats_reflection"
        )
        return score_result["score"] / 100  # Normalize

    @time_function(logger=None)
    def _build_prompt(self, node, context:dict, mode="reason"):
        """Build prompt from node state"""
        if isinstance(node.state, dict):
            state = node.state["current"]
        else:
            state = str(node.state)
        
        merged = {
            **context,
            "state": state,
            "trace": node.trace,
            "mode": mode,
        }   
        prompt = self.prompt_loader.load_prompt(self.cfg, merged)
        print(f"Prompt for {mode}: {prompt[:100]}...")  # Debugging output
        return prompt