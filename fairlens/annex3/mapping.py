"""
EU AI Act Annex III Risk Classification

This module provides a lightweight rule engine to classify
whether an AI system belongs to one of the Annex III
high-risk categories defined in the EU AI Act.

Reference:
EU AI Act Annex III
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


# ==========================================================
# Risk Category
# ==========================================================

@dataclass(frozen=True)
class RiskCategory:
    """
    Represents one Annex III category.
    """

    id: int
    name: str
    description: str
    keywords: List[str]


# ==========================================================
# Annex III Registry
# ==========================================================

ANNEX3_REGISTRY: Dict[int, RiskCategory] = {

    1: RiskCategory(
        id=1,
        name="Biometric Identification",
        description="Remote biometric identification and categorization.",
        keywords=[
            "face recognition",
            "biometric",
            "fingerprint",
            "iris",
            "voice authentication",
        ],
    ),

    2: RiskCategory(
        id=2,
        name="Critical Infrastructure",
        description="AI systems managing critical infrastructure.",
        keywords=[
            "power grid",
            "electricity",
            "water supply",
            "transport",
            "railway",
            "airport",
        ],
    ),

    3: RiskCategory(
        id=3,
        name="Education",
        description="Admissions, examinations and student evaluation.",
        keywords=[
            "student",
            "exam",
            "admission",
            "grading",
            "education",
        ],
    ),

    4: RiskCategory(
        id=4,
        name="Employment",
        description="Recruitment, promotion and employee evaluation.",
        keywords=[
            "resume",
            "recruitment",
            "hiring",
            "job",
            "promotion",
            "employee",
            "candidate",
        ],
    ),

    5: RiskCategory(
        id=5,
        name="Essential Services",
        description="Credit scoring, insurance, healthcare and public services.",
        keywords=[
            "loan",
            "credit",
            "insurance",
            "bank",
            "healthcare",
            "hospital",
        ],
    ),

    6: RiskCategory(
        id=6,
        name="Law Enforcement",
        description="Crime prediction and policing.",
        keywords=[
            "crime",
            "police",
            "recidivism",
            "criminal",
            "investigation",
        ],
    ),

    7: RiskCategory(
        id=7,
        name="Migration & Border Control",
        description="Visa, asylum and border management.",
        keywords=[
            "visa",
            "passport",
            "border",
            "migration",
            "immigration",
            "asylum",
        ],
    ),

    8: RiskCategory(
        id=8,
        name="Justice & Democratic Processes",
        description="Judicial decision support and elections.",
        keywords=[
            "judge",
            "court",
            "legal",
            "election",
            "voting",
            "justice",
        ],
    ),
}


# ==========================================================
# Classifier
# ==========================================================

class Annex3Classifier:
    """
    Rule-based Annex III classifier.
    """

    def classify(
        self,
        use_case: str,
    ) -> Dict:

        use_case = use_case.lower()

        matched = []

        for category in ANNEX3_REGISTRY.values():

            if any(
                keyword in use_case
                for keyword in category.keywords
            ):
                matched.append(category)

        if not matched:

            return {
                "high_risk": False,
                "categories": [],
                "recommendation":
                    "No Annex III category detected.",
            }

        return {
            "high_risk": True,
            "categories": [
                {
                    "id": c.id,
                    "name": c.name,
                    "description": c.description,
                }
                for c in matched
            ],
            "recommendation":
                "Perform comprehensive fairness, transparency "
                "and risk assessment before deployment.",
        }


# ==========================================================
# Convenience Function
# ==========================================================

def classify_annex3_risk(
    use_case: str,
):
    """
    Functional API.
    """

    classifier = Annex3Classifier()

    return classifier.classify(use_case)