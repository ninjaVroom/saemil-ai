"""Request/response schemas: validation happens here, at the trust boundary .

Strict types: no string-to-number or bool-to-int coercion; NaN/Infinity (which Python's JSON parser accepts) rejected.
The heads re-validate; their ValueError becomes a 422 in app.py. Tabular per-feature upper caps depend on the
loaded artifact, so app.py checks them after this schema.
"""
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from heads import names as names_head

Pixel = Annotated[int, Field(strict=True, ge=0, le=255)]
Count = Annotated[int, Field(strict=True, ge=0)]
Seconds = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VisionRequest(_Strict):
    pixels: Annotated[list[Pixel], Field(min_length=784, max_length=784)]


class DigitProb(BaseModel):
    digit: int
    prob: float


class VisionDigit(BaseModel):
    blank: Literal[False]
    digit: int
    top3: list[DigitProb]


class VisionBlank(BaseModel):
    blank: Literal[True]
    message: str


VisionResponse = VisionDigit | VisionBlank  # the head's own shapes, unchanged (D-003, D-015)


class NamesRequest(_Strict):
    count: Annotated[int, Field(strict=True, ge=names_head.COUNT_RANGE[0], le=names_head.COUNT_RANGE[1])]
    temperature: Annotated[float, Field(strict=True, ge=names_head.TEMP_RANGE[0], le=names_head.TEMP_RANGE[1],
                                        allow_inf_nan=False)] = 0.85
    seed: Annotated[int, Field(strict=True, ge=0, le=2**63 - 1)] | None = None


class NamesResponse(BaseModel):
    names: list[str]
    accepted: int
    requested: int
    attempts: int


class TabularRequest(_Strict):
    """Raw whole-session totals in heads.tabular.FEATURES order: counts are whole numbers, durations seconds."""
    Administrative: Count
    Administrative_Duration: Seconds
    Informational: Count
    Informational_Duration: Seconds
    ProductRelated: Count
    ProductRelated_Duration: Seconds


class TabularResponse(BaseModel):
    probability: float
    source_id: str
    source_kind: Literal["real", "synthetic"]


class ReportRequest(_Strict):
    mode: Literal["offline", "cli", "cloud"] = "offline"
    vendor: Literal["claude", "gemini", "openai", "deepseek", "ollama", "llm"] | None = None
    vision: dict | None = None   # head outputs; the bridge keeps only permitted, well-formed fields
    names: dict | None = None
    tabular: dict | None = None


class ReportResponse(BaseModel):
    text: str
    provider_used: str
    mode_used: str
    fallback_reason: str | None
    latency_ms: int


class TabularFeature(BaseModel):
    name: str
    kind: Literal["count", "duration"]
    median: float
    percentiles: dict[str, float]  # "p0".."p100", raw train units
    input_max: float               # route cap: values above it get 422


class HeadStatus(BaseModel):
    ready: bool
    source_id: str | None = None
    source_kind: str | None = None
    error: str | None = None
    features: list[TabularFeature] | None = None  # tabular only (D-013)


class StatusResponse(BaseModel):
    ui_built: bool
    heads: dict[str, HeadStatus]
    providers: dict
