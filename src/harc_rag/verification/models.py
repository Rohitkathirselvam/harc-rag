from dataclasses import dataclass
from dataclasses import field


@dataclass
class VerificationResult:

    original_answer: str

    verified_answer: str

    is_verified: bool

    confidence: float

    reason: str = ""

    verdict: str = "UNSUPPORTED"

    claim_results: list[dict[str, str]] = field(
        default_factory=list
    )