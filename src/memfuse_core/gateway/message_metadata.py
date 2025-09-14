"""Message metadata interpreter for API -> Gateway boundary.

Parses message-level metadata to produce routing/trigger decisions for write paths
(e.g., add_messages). Keeps API thin: API collects payload and delegates parsing here.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class MessageAddDecision:
    def __init__(
        self,
        trigger_m3: bool,
        workflow_name: Optional[str],
        user_goal: Optional[str],
        history_task: Optional[str],
    ) -> None:
        self.trigger_m3 = bool(trigger_m3)
        self.workflow_name = workflow_name
        self.user_goal = user_goal
        self.history_task = history_task


class MessageMetadataInterpreter:
    """Interprets messages metadata to produce add/write decisions."""

    def __init__(self) -> None:
        pass

    def interpret_add_messages(
        self,
        messages: List[Dict[str, Any]],
        tag: Optional[str] = None,
        legacy_tag_trigger: bool = False,
    ) -> MessageAddDecision:
        """Interpret metadata for add_messages.

        Rules:
        - If there is a user message with metadata.task_eos == true, trigger M3.
        - Use metadata.task as workflow_name. Use the last EOS user message content as user_goal.
        - If legacy_tag_trigger is enabled and tag/body tag == 'm3', trigger with workflow_name 'm3_legacy' when absent.
        """
        eos_msgs: List[Dict[str, Any]] = [
            m for m in messages
            if isinstance(m, dict)
            and str(m.get("role", "")) == "user"
            and isinstance(m.get("metadata"), dict)
            and bool(m.get("metadata", {}).get("task_eos", False))
        ]
        legacy_triggered = False
        if not eos_msgs and legacy_tag_trigger:
            body_tag_msgs = [
                m for m in messages
                if isinstance(m, dict)
                and str(m.get("role","")) == "user"
                and isinstance(m.get("metadata"), dict)
                and str(m.get("metadata", {}).get("tag", "")).lower() == "m3"
            ]
            if body_tag_msgs or str(tag or "").lower() == "m3":
                legacy_triggered = True
                eos_msgs = body_tag_msgs or [m for m in messages if str(m.get("role","")) == "user"]

        if not eos_msgs:
            return MessageAddDecision(False, None, None, None)

        last = eos_msgs[-1]
        user_goal = str(last.get("content") or "").strip() or None
        md = last.get("metadata") or {}
        workflow_name = str(md.get("task") or "").strip() or None
        if legacy_triggered and not workflow_name:
            workflow_name = "m3_legacy"
        return MessageAddDecision(True, workflow_name, user_goal, workflow_name)