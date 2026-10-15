# Copyright (C) 2026 Microsoft Corporation.
# Licensed under the MIT License

"""Tests for GraphExtractor response handling.

Validates that _process_document joins gleaning responses so that
records at the join are not fused, and that _process_result tolerates
the completion delimiter arriving attached to the final record, as the
shipped extraction prompt's own example demonstrates.
"""

from dataclasses import dataclass

import pandas as pd
import pytest
from graphrag.index.operations.extract_graph.graph_extractor import (
    COMPLETION_DELIMITER,
    RECORD_DELIMITER,
    TUPLE_DELIMITER,
    GraphExtractor,
)

# The tail of the shipped prompt's Example 1 output: the last record is not
# terminated by a record delimiter, and the completion delimiter follows it on
# the next line.
BASE_RESPONSE = (
    '("entity"<|>CENTRAL INSTITUTION<|>ORGANIZATION<|>The Central Institution is'
    " the Federal Reserve of Verdantis)\n"
    "##\n"
    '("relationship"<|>MARTIN SMITH<|>CENTRAL INSTITUTION<|>Martin Smith is the'
    " Chair of the Central Institution<|>9)\n"
    "<|COMPLETE|>"
)

# A gleaning response in the same shape. ALICE CHEN appears nowhere else.
GLEANING_RESPONSE = (
    '("entity"<|>ALICE CHEN<|>PERSON<|>Alice Chen is a policy analyst at the'
    " Central Institution)\n"
    "##\n"
    '("relationship"<|>ALICE CHEN<|>CENTRAL INSTITUTION<|>Alice Chen analyses'
    " policy for the Central Institution<|>7)\n"
    "<|COMPLETE|>"
)


@dataclass
class _StubResponse:
    """Minimal stand-in for LLMCompletionResponse."""

    content: str


class _StubModel:
    """Completion model that replays a fixed list of responses in order."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls = 0

    async def completion_async(self, **_kwargs) -> _StubResponse:
        """Return the next canned response."""
        content = self._responses[self.calls]
        self.calls += 1
        return _StubResponse(content)


def _extractor(responses: list[str], max_gleanings: int = 1) -> GraphExtractor:
    """Build a GraphExtractor driven by canned model responses."""
    return GraphExtractor(
        model=_StubModel(responses),  # type: ignore[arg-type]
        prompt="{input_text} {entity_types}",
        max_gleanings=max_gleanings,
    )


async def _extract(
    responses: list[str], max_gleanings: int = 1
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a full extraction over canned responses."""
    extractor = _extractor(responses, max_gleanings)
    return await extractor("some text", ["ORGANIZATION", "PERSON"], "tu1")


class TestGleaningJoin:
    """Tests for how gleaning responses are combined before parsing."""

    @pytest.mark.asyncio
    async def test_entity_only_in_gleaning_survives(self):
        """An entity introduced by a gleaning is not lost at the join."""
        entities, _ = await _extract([BASE_RESPONSE, GLEANING_RESPONSE])

        assert "ALICE CHEN" in set(entities["title"])

    @pytest.mark.asyncio
    async def test_base_response_entity_description_intact(self):
        """The final record of the base response keeps its own description."""
        entities, _ = await _extract([BASE_RESPONSE, GLEANING_RESPONSE])

        central = entities[entities["title"] == "CENTRAL INSTITUTION"].iloc[0]
        assert "Alice Chen" not in central["description"]

    @pytest.mark.asyncio
    async def test_relationship_only_in_gleaning_survives(self):
        """A relationship introduced by a gleaning is not lost at the join."""
        _, relationships = await _extract([BASE_RESPONSE, GLEANING_RESPONSE])

        pairs = set(zip(relationships["source"], relationships["target"], strict=True))
        assert ("ALICE CHEN", "CENTRAL INSTITUTION") in pairs

    @pytest.mark.asyncio
    async def test_no_gleanings_is_unchanged(self):
        """With gleanings disabled the base response parses as before."""
        entities, relationships = await _extract([BASE_RESPONSE], max_gleanings=0)

        assert set(entities["title"]) == {"CENTRAL INSTITUTION"}
        assert len(relationships) == 1

    @pytest.mark.asyncio
    async def test_already_terminated_response_adds_no_empty_entity(self):
        """A response that does terminate its last record yields no stray rows."""
        terminated = BASE_RESPONSE.replace("9)\n", "9)\n##\n")
        entities, relationships = await _extract([terminated, GLEANING_RESPONSE])

        assert len(entities) == 2
        assert len(relationships) == 2


class TestCompletionDelimiterHandling:
    """Tests for records that carry a trailing completion delimiter."""

    @pytest.mark.asyncio
    async def test_relationship_weight_is_not_corrupted(self):
        """The weight of the last relationship survives the completion delimiter."""
        _, relationships = await _extract([BASE_RESPONSE], max_gleanings=0)

        assert relationships.iloc[0]["weight"] == pytest.approx(9.0)

    @pytest.mark.asyncio
    async def test_entity_description_excludes_completion_delimiter(self):
        """A trailing completion delimiter is not absorbed into a description."""
        response = (
            '("entity"<|>CENTRAL INSTITUTION<|>ORGANIZATION<|>The Central'
            " Institution is the Federal Reserve of Verdantis)\n"
            "<|COMPLETE|>"
        )
        entities, _ = await _extract([response], max_gleanings=0)

        assert COMPLETION_DELIMITER not in entities.iloc[0]["description"]

    @pytest.mark.asyncio
    async def test_delimiter_on_its_own_record_is_skipped(self):
        """A completion delimiter behind a record delimiter adds no rows."""
        response = (
            '("entity"<|>CENTRAL INSTITUTION<|>ORGANIZATION<|>desc)\n##\n<|COMPLETE|>'
        )
        entities, _ = await _extract([response], max_gleanings=0)

        assert len(entities) == 1


class TestDelimiterConstants:
    """Guards on the delimiters the parser and the prompt must agree on."""

    def test_constants_match_prompt(self):
        """Parser delimiters match those the shipped prompt instructs the model to use."""
        assert TUPLE_DELIMITER == "<|>"
        assert RECORD_DELIMITER == "##"
        assert COMPLETION_DELIMITER == "<|COMPLETE|>"
