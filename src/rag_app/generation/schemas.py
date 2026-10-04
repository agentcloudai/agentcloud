"""Output schemas. The LLM must return JSON matching these models.

Shape is UI-first: a crisp one-line ``summary`` plus ordered, individually-cited
``claims`` (the steps/points). Every claim carries the source ids that support it,
so a UI can render the answer and make each point's citations clickable.
"""
from pydantic import BaseModel, Field


class Claim(BaseModel):
    text: str = Field(description="One clear, concise sentence: a single step or point of the answer.")
    source_ids: list[str] = Field(
        min_length=1, description="IDs like 'S1', 'S2' of the sources that support this sentence."
    )


class Answer(BaseModel):
    summary: str = Field(
        default="", description="A crisp 1-2 sentence direct answer to the question."
    )
    claims: list[Claim] = Field(
        default_factory=list, description="The key points or ordered steps, each individually cited."
    )
    insufficient_context: bool = Field(
        default=False, description="True if the sources do not contain the answer."
    )
