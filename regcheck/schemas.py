"""Data that crosses module boundaries or comes back from an LLM."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Status = Literal["compliant", "partial", "non_compliant"]


class Section(BaseModel):
    """Parent document: one regulation article or one SRS section."""

    id: str  # "gdpr:art32", "e-store:s4"
    doc: str
    ref: str  # "Article 32 - Security of processing"
    text: str


class Chunk(BaseModel):
    """Child: one self-contained statement, the unit that gets embedded."""

    id: str  # "gdpr:art32:3"
    doc: str
    parent_id: str
    text: str
    meta: dict = {}


class Finding(BaseModel):
    """One CC agent's verdict on one requirement. Field order puts reasoning before the verdict."""

    requirement_id: str
    clauses: list[str]  # Section ids from the supplied context
    rationale: str
    status: Status
    suggestion: str = ""  # rewritten requirement; empty when compliant
    evidence: list[str] = []  # retrieved policy Chunk ids; filled by code, never by the LLM

    @property
    def flagged(self) -> bool:
        return self.status != "compliant"


class Step(BaseModel):
    question: str
    answer: str
    sources: list[str]  # Section or Chunk ids retrieved for this step


class Answer(BaseModel):
    question: str
    text: str
    contexts: list[str]  # every retrieved passage, as shown to the LLM (RAGAs input)
    steps: list[Step]


class Report(BaseModel):
    agent: str
    findings: list[Finding]
    plan: list[str] = []  # the agent's CoT sub-queries; refinement reuses them


class Dispute(BaseModel):
    """A requirement flagged by exactly one CC agent."""

    requirement_id: str
    flagged_by: str
    missed_by: str
    rationale: str  # the flagging agent's rationale
    clauses: list[str] = []  # the flagging agent's clauses
    valid: bool | None = None  # RA's judgement of that rationale
    message: str = ""  # RA's revision request to the agent that must re-evaluate


class Verdict(BaseModel):
    """RA's judgement of one dispute."""

    requirement_id: str
    valid: bool
    message: str


class Resolution(BaseModel):
    """RA's final call on one flagged requirement. Compliant ones are left out of the report."""

    requirement_id: str
    clauses: list[str]
    rationale: str
    status: Status
    suggestion: str = ""
    confidence: float = Field(ge=0, le=1)


class Issue(Resolution):
    status: Literal["partial", "non_compliant"]
    requirement: str
    positions: dict[str, Finding] = {}  # each CC agent's final finding; filled by code
    evidence: list[str] = []  # filled by code


class FinalReport(BaseModel):
    summary: str
    issues: list[Issue]


class Feedback(BaseModel):
    """One human review decision."""

    requirement: str
    regulation: str
    clauses: list[str]
    status: str  # what the system said
    decision: Literal["accept", "reject", "escalate"]
    justification: str
    ts: datetime = Field(default_factory=datetime.now)
