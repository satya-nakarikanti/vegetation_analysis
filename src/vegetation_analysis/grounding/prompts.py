"""Text prompts and prompt construction for Grounding DINO detection.

All default prompt strings and label constants are imported from
:mod:`vegetation_analysis.grounding.constants` so that values are never
duplicated across the package.

The detector receives prompts as arguments at call time rather than embedding
them internally.  This module provides the tools to build those prompts.
"""

from __future__ import annotations

from vegetation_analysis.grounding.constants import (
    DEFAULT_VEGETATION_PROMPT,
    LABEL_POLE,
    LABEL_TREE,
    POLE_PROMPT_VARIANTS,
    TREE_PROMPT_VARIANTS,
    VEGETATION_LABELS,
)

# Re-export so callers can import from either `prompts` or `constants`.
__all__ = [
    "DEFAULT_VEGETATION_PROMPT",
    "LABEL_POLE",
    "LABEL_TREE",
    "POLE_PROMPT_VARIANTS",
    "TREE_PROMPT_VARIANTS",
    "VEGETATION_LABELS",
    "PromptBuilder",
    "build_vegetation_prompt",
]


class PromptBuilder:
    """Construct and validate Grounding DINO text prompts.

    Grounding DINO expects a single text string where each object category is
    separated by `` . `` and the string ends with `` . ``.  This class
    provides a structured interface for building and validating such prompts.

    Example::

        prompt = PromptBuilder().add("tree").add("utility pole").build()
        # Returns: "tree . utility pole ."
    """

    def __init__(self) -> None:
        self._labels: list[str] = []

    def add(self, label: str) -> PromptBuilder:
        """Append an object category label to the prompt.

        Args:
            label: A single object category name, for example ``"tree"`` or
                ``"utility pole"``. Leading and trailing whitespace is stripped.

        Returns:
            This :class:`PromptBuilder` instance to enable method chaining.

        Raises:
            ValueError: If the label is empty after stripping whitespace.
        """

        cleaned = label.strip()
        if not cleaned:
            raise ValueError("Prompt label must not be empty.")
        self._labels.append(cleaned)
        return self

    def build(self) -> str:
        """Return the assembled Grounding DINO prompt string.

        Returns:
            A formatted prompt string with labels separated by `` . `` and
            terminated by `` . ``.

        Raises:
            ValueError: If no labels have been added before calling
                :meth:`build`.
        """

        if not self._labels:
            raise ValueError(
                "At least one label must be added before building a prompt."
            )
        return " . ".join(self._labels) + " ."

    @staticmethod
    def validate(prompt: str) -> bool:
        """Return whether the given string is a valid Grounding DINO prompt.

        A valid prompt is non-empty, contains at least one label, and ends
        with `` .``.

        Args:
            prompt: The prompt string to validate.

        Returns:
            ``True`` if the prompt is structurally valid, ``False`` otherwise.
        """

        stripped = prompt.strip()
        return bool(stripped) and stripped.endswith(".")


def build_vegetation_prompt(
    tree_label: str = LABEL_TREE,
    pole_label: str = LABEL_POLE,
) -> str:
    """Build a two-label vegetation prompt from reusable prompt variants.

    Args:
        tree_label: Tree prompt text. Recommended values are exposed through
            :data:`TREE_PROMPT_VARIANTS`.
        pole_label: Pole prompt text. Recommended values are exposed through
            :data:`POLE_PROMPT_VARIANTS`.

    Returns:
        A Grounding DINO prompt such as ``"tree . utility pole ."``.
    """

    return PromptBuilder().add(tree_label).add(pole_label).build()
