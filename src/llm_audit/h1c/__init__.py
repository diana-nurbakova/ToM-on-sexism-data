"""H1c — persona prompting (selective shift toward the NLPercep human pattern).

The full gender × age persona grid (``prompts-h1c-persona.md``), mirroring the EXIST
annotator stratification: female/male × {18-22, 23-45, 46+}. The persona replaces the
E0 system prompt; the user turn is the unchanged E0 base (bare 1.1 / defined 1.3). The
test is two-fold (§4.2): the female-minus-male model shift should (i) point the same
way as the female-minus-male human annotator shift, and (ii) concentrate on the
affective categories (objectification, sexual-violence, misogyny; misogyny sharpest)
while staying negligible on the cognitive categories and on detection. Refusal /
disclaimer cells are excluded from the shift and reported as a coverage finding via
the §6.2 parse-status taxonomy. Runs via the E0 ``Runner`` through its prompt hook.

  * ``sampling`` — the persona-grid sub-sample, the same items as E3 (``prompts-
    h1c-persona.md`` §2.5); delegates to ``e3.sampling``.
"""

from . import sampling  # noqa: F401

__all__ = ["sampling"]
