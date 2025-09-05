#!/usr/bin/env python3
"""
mAgent2.py

An intelligent browser automation agent built with Anthropic's Claude model.
This agent observes webpage state (DOM, accessibility tree, screenshots),
plans actions, critiques its past actions, and executes the next best step
towards accomplishing a goal. Includes memory of past actions and screenshot
comparisons to avoid repetition and more thorough analysis of actions' effects.

Modified from the AGI SDK hackable.py with Anthropic integration replacing OpenAI.
"""

import argparse
import base64
import dataclasses
import io
import logging
import re
from typing import Literal

import numpy as np
import anthropic
from PIL import Image

# AGI SDK components
from agisdk import REAL
from agisdk.REAL.demo_agent.run_demo import str2bool
from agisdk.REAL.browsergym.experiments import Agent, AbstractAgentArgs
from agisdk.REAL.browsergym.core.action.highlevel import HighLevelActionSet
from agisdk.REAL.browsergym.utils.obs import (
    flatten_axtree_to_str,
    flatten_dom_to_str,
    prune_html,
)

# --- Configure logging ---
logging.basicConfig(
    level=logging.INFO,  # Use DEBUG for very detailed logs
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[
        logging.FileHandler("agent.log", mode="a"),  # persistent logs
        logging.StreamHandler(),                     # live console logs
    ],
)
logger = logging.getLogger(__name__)


# ------------------------
# Utility Functions
# ------------------------

def image_to_jpg_base64_url(image: np.ndarray | Image.Image) -> str:
    """
    Convert an image (numpy array or PIL Image) into a base64-encoded JPEG data URL.

    Args:
        image (np.ndarray | PIL.Image): Image input.

    Returns:
        str: Base64-encoded JPEG as `data:image/jpeg;base64,...`
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode in ("RGBA", "LA"):
        image = image.convert("RGB")

    with io.BytesIO() as buffer:
        image.save(buffer, format="JPEG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode()

    return f"data:image/jpeg;base64,{image_base64}"


# ------------------------
# DemoAgent Definition
# ------------------------

class DemoAgent(Agent):
    """
    An Anthropic-powered browser agent that:
    - Observes the environment (goal, DOM, accessibility tree, screenshots).
    - Plans actions with critique and memory.
    - Avoids redundant or failed actions.
    - Produces actions inside Python code blocks.
    """

    def __init__(self, model_name: str, use_screenshot: bool) -> None:
        super().__init__()
        self.use_screenshot = use_screenshot
        self.client = anthropic.Anthropic()
        self.model_name = model_name

        logger.info(
            f"🚀 Initializing DemoAgent with model={self.model_name}, "
            f"use_screenshot={self.use_screenshot}"
        )

        # Define action space (chat, bidding, infeasible responses)
        self.action_set = HighLevelActionSet(
            subsets=["chat", "bid", "infeas"],
            strict=False,
            multiaction=False,
        )

        # Agent state
        self.action_history = []   # [(action, error), ...]
        self.memory_text = []      # summarized natural language memory
        self.max_memory_steps = 15
        self.last_screenshot = None  # stored for comparison

    # ------------------------
    # Preprocessing
    # ------------------------

    def obs_preprocessor(self, obs: dict) -> dict:
        """
        Convert raw environment observation into processed dict for LLM input.

        Args:
            obs (dict): Environment observation.

        Returns:
            dict: Processed observation with text and screenshot data.
        """
        logger.debug("Preprocessing observation...")
        processed = {
            "chat_messages": obs["chat_messages"],
            "screenshot": obs["screenshot"],
            "goal_object": obs["goal_object"],
            "last_action": obs["last_action"],
            "last_action_error": obs["last_action_error"],
            "axtree_txt": flatten_axtree_to_str(obs["axtree_object"]),
            "pruned_html": prune_html(flatten_dom_to_str(obs["dom_object"])),
        }
        logger.debug("Observation preprocessed successfully.")
        return processed

    # ------------------------
    # LLM Prompt Construction
    # ------------------------

    def _build_system_message(self) -> str:
        """System instructions for the LLM."""
        logger.debug("Building system message for LLM...")
        return """You are an expert AI agent controlling a web browser to accomplish a goal.
You are given:
- the current goal,
- the accessibility tree of the page,
- a screenshot of the current page,
- a screen of the page before the previous action (if applicable)
You can perform actions to interact with the browser.

Follow this cycle:

1. Critique: Evaluate the previous actions. Check if the last action succeeded or failed. Analyze whether the actions already taken were productive towards the goal. Identify the differences between the previous screenshot (if given) and the current screenshot to determine what the previous action did and how to proceed.
2. Think: Decide the next best step towards the goal, avoiding any actions that have already succeeded (e.g., do not re-add items to cart).
3. Act: Output a single, precise action enclosed in a ```python code block.

⚠️ Important:
- Only take actions if necessary. Avoid repeating actions.
- Avoid getting stuck in a cycle of non-productive actions. Check the History of Past Actions and the Summarized Memory of Past Actions.
- Keep in mind some actions or information requested by the user may not be possible or available (e.g., a desired flight is not available). Tell the user if this is the case.
- Only output the action inside the code block. Do not include any text outside it.
- When you have succeeded the goal or if the goal is not possible, make the action a send_msg_to_user() action.
"""

    def _build_user_messages(self, obs: dict) -> list[dict]:
        """
        Build user-facing messages for the LLM input, including:
        - Goal
        - Screenshots (prev + current)
        - Accessibility tree
        - Action space
        - Past history & memory
        """
        logger.debug("Building user messages from observation...")
        user_msgs = []

        # Goal
        goal_text = obs['goal_object'][0]['text']
        user_msgs.append({"type": "text", "text": f"# Goal\n\n{goal_text}"})

        # Screenshots (if enabled)
        if self.use_screenshot:
            if self.last_screenshot is not None:
                logger.debug("Adding previous screenshot...")
                user_msgs.extend([
                    {"type": "text", "text": "# Previous Page Screenshot"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_to_jpg_base64_url(self.last_screenshot).split(",")[1],
                        }
                    }
                ])
            logger.debug("Adding current screenshot...")
            user_msgs.extend([
                {"type": "text", "text": "# Current Page Screenshot"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_to_jpg_base64_url(obs["screenshot"]).split(",")[1],
                    }
                }
            ])

        # Accessibility Tree
        user_msgs.append({
            "type": "text",
            "text": f"# Current page Accessibility Tree\n\n{obs['axtree_txt']}"
        })

        # Action Space
        user_msgs.append({
            "type": "text",
            "text": f"# Action Space\n\n{self.action_set.describe(with_long_description=False, with_examples=True)}"
        })

        # Past Action History
        if self.action_history:
            logger.info(f"📜 Adding {len(self.action_history)} past actions...")
            history_text = "# History of Past Actions\n\n"
            for i, (action, error) in enumerate(self.action_history):
                history_text += f"Step {i+1}:\n- Action: {action}\n"
                history_text += f"- Result: {'Error - ' + error if error else 'Success'}\n"

            # Explicit last action success instruction
            last_action, last_error = self.action_history[-1]
            last_success = last_error is None or last_error == ""
            history_text += (
                f"\n# Last Action Status\n- Last action: {last_action}\n"
                f"- Success: {last_success}\n"
                "Instruction: If the last action succeeded, do not repeat it."
            )
            user_msgs.append({"type": "text", "text": history_text})

        # Summarized Memory
        if self.memory_text:
            logger.info(f"🧠 Adding summarized memory ({len(self.memory_text)} steps)...")
            memory_summary = "# Summarized Memory of Past Actions\n\n" + "\n".join(self.memory_text)
            user_msgs.append({"type": "text", "text": memory_summary})

        return user_msgs

    # ------------------------
    # Model Query
    # ------------------------

    def _query_model(self, system_msg: str, user_msgs: list[dict]) -> str:
        """Send messages to Anthropic LLM and return its response."""
        logger.info("💬 Querying Anthropic model...")
        response = self.client.messages.create(
            model=self.model_name,
            system=system_msg,
            messages=[{"role": "user", "content": user_msgs}],
            max_tokens=1024,
            temperature=0.0,
        )
        text_response = response.content[0].text
        logger.debug(f"Raw model response (first 200 chars): {text_response[:200]}...")
        return text_response

    # ------------------------
    # Action Selection
    # ------------------------

    def _summarize_element_context(self, action: str, axtree_txt: str) -> str:
        """Extract short description of element targeted by an action."""
        match = re.search(r'role=(\w+).*?name="([^"]+)"', axtree_txt)
        if match:
            role, label = match.groups()
            return f'{role} labeled "{label}"'
        return f"element related to `{action}`"

    def get_action(self, obs: dict) -> tuple[str, dict]:
        """
        Main agent loop:
        - Build LLM prompt.
        - Query model.
        - Extract action from response.
        - Update history & memory.
        - Return action.
        """
        logger.info("🤖 Generating next action...")

        # Prompt LLM
        system_msg = self._build_system_message()
        user_msgs = self._build_user_messages(obs)
        llm_response = self._query_model(system_msg, user_msgs)

        # Parse action from Python code block
        action_match = re.search(r"```(?:python)?\n(.*?)\n```", llm_response, re.DOTALL)
        if action_match:
            action = action_match.group(1).strip()
            logger.info(f"🛠️ Parsed action: {action}")
        else:
            action = "think('Could not parse an action. Must output inside a python code block.')"
            logger.warning(f"⚠️ Could not parse action from LLM response: {llm_response}")

        # Update history
        last_error = obs.get("last_action_error")
        self.action_history.append((action, last_error))

        # Update summarized memory
        step_num = len(self.action_history)
        goal_text = obs['goal_object'][0]['text']
        element_summary = self._summarize_element_context(action, obs.get("axtree_txt", ""))
        mem_entry = (
            f"Step {step_num}: I attempted `{action}` on {element_summary} "
            f"while pursuing \"{goal_text}\". Result: {'Error - ' + last_error if last_error else 'Success'}."
        )
        self.memory_text.append(mem_entry)
        if len(self.memory_text) > self.max_memory_steps:
            self.memory_text.pop(0)

        # Update screenshot memory
        self.last_screenshot = obs["screenshot"]

        return action, {"raw_response": llm_response, "memory_entry": mem_entry}


# ------------------------
# CLI + Runner
# ------------------------

@dataclasses.dataclass
class DemoAgentArgs(AbstractAgentArgs):
    """Arguments wrapper for DemoAgent."""
    agent_name: str = "DemoAgent"
    model_name: str = "claude-3-sonnet-20240229"
    use_screenshot: bool = True

    def make_agent(self):
        logger.info("🔧 Creating DemoAgent instance...")
        return DemoAgent(self.model_name, self.use_screenshot)


def run_demo_agent(model="claude-3-sonnet-20240229", task=None, headless=False, leaderboard=False, run_id=None):
    """Run DemoAgent on a browsergym task."""
    logger.info(f"▶️ Starting DemoAgent run with model={model}, task={task}, headless={headless}")
    agent_args = DemoAgentArgs(model_name=model, use_screenshot=True)

    harness = REAL.harness(
        agentargs=agent_args,
        task_name=task,
        task_type="omnizon",
        headless=headless,
        max_steps=55,
        use_axtree=True,
        use_screenshot=agent_args.use_screenshot,
        leaderboard=leaderboard,
        run_id=run_id,
        use_cache=True,
    )

    logger.info("🏃 Running task...")
    results = harness.run()
    logger.info(f"🏁 Task completed. Results: {results}")
    return results


# ------------------------
# Entry Point
# ------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DemoAgent on browsergym tasks")
    parser.add_argument("--model", type=str, default="claude-3-sonnet-20240229",
                        help="Model to use (e.g., claude-3-sonnet-20240229)")
    parser.add_argument("--task", type=str, default=None,
                        help="Task to run (e.g., 'webclones.omnizon-1'). Runs all tasks if not specified.")
    parser.add_argument("--headless", type=str2bool, default=False,
                        help="Run headless (default: False)")
    parser.add_argument("--run_id", type=str, default=None,
                        help="Run ID for leaderboard submission")
    parser.add_argument("--leaderboard", type=str2bool, default=False,
                        help="Submit results to leaderboard (default: False)")

    args = parser.parse_args()
    run_demo_agent(
        model=args.model,
        task=args.task,
        headless=args.headless,
        leaderboard=args.leaderboard,
        run_id=args.run_id
    )
