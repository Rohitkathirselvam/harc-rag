from dataclasses import dataclass


@dataclass
class VerificationResult:

    original_answer: str

    verified_answer: str

    is_verified: bool

    confidence: float

    reason: str = ""