"""Prompts are versioned here so every answer can be traced to the prompt that made it."""
from llama_index.core import PromptTemplate

QA_PROMPT_VERSION = "qa-v5"

QA_PROMPT = PromptTemplate(
    "You are a precise technical assistant. Answer the question using ONLY the sources below.\n\n"
    "Return:\n"
    "- summary: a crisp, direct 1-2 sentence answer (no preamble, no fluff).\n"
    "- claims: the COMPLETE ordered set of steps/points. For a how-to, include EVERY required step "
    "found in the sources — from prerequisites to the final step — in logical order. Do not stop at "
    "2-3 steps if the sources describe more. Each claim is ONE short clear sentence.\n\n"
    "Rules:\n"
    "- Be thorough: a reader should be able to complete the task from your steps alone.\n"
    "- Every claim must be SUPPORTED BY and CITE the specific source id(s) it comes from (e.g. S1). "
    "The cited source must actually contain that information — never attach a citation to a claim it "
    "does not support.\n"
    "- Combine multiple sources for a multi-service task, but only state what the sources say.\n"
    "- Do NOT add facts from outside the sources. If part of the question is not covered by any source, "
    "omit it rather than inventing it.\n"
    "- Only set insufficient_context=true if NONE of the sources are relevant to the question.\n\n"
    "Sources:\n{context}\n\n"
    "Question: {question}\n"
)
