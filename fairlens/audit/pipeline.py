from __future__ import annotations

from typing import Dict, Any


class AuditPipeline:
    """
    Pipeline responsible for executing each stage
    of a fairness audit.
    """

    def __init__(self):

        self.steps = []

    def register_step(self, name, callable_step):

        self.steps.append(
            (name, callable_step)
        )

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute all registered steps.
        """

        for name, step in self.steps:

            context[name] = step(context)

        return context