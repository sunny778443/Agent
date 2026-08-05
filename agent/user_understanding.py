"""
User Understanding, Emotion, and Skill Profiling module.
Estimates frustration, confusion, urgency, excitement, satisfaction, and skill level
to dynamically adjust communication styles and planning granularity.
"""
from typing import Any


class UserUnderstandingModel:
    """
    Profiles incoming user requests to estimate emotion (frustration, confusion, urgency,
    excitement, satisfaction) and skill levels (beginner, expert) to dynamically
    calibrate plan steps explanation depths and communication styles.
    """
    def __init__(self) -> None:
        pass

    def profile_user_request(self, text: str) -> dict[str, Any]:
        """
        Analyzes task text strings from scratch using heuristic matching,
        returning an estimated emotion dictionary and communication style guide.
        """
        text_lower = text.lower()

        # 1. Frustration heuristic
        frustration = 0.1
        if any(w in text_lower for w in ["fail", "error", "broken", "stop", "stupid", "wrong", "blank", "unusable"]):
            frustration = 0.8
        elif "please" in text_lower or "help" in text_lower:
            frustration = 0.4

        # 2. Confusion heuristic
        confusion = 0.1
        if any(w in text_lower for w in ["why", "how", "what", "where", "confused", "cannot find", "blank", "blank white"]):
            confusion = 0.7

        # 3. Urgency heuristic
        urgency = 0.1
        if any(w in text_lower for w in ["asap", "now", "urgent", "quick", "fast", "immediately", "blocker", "emergency"]):
            urgency = 0.9

        # 4. Excitement heuristic
        excitement = 0.1
        if any(w in text_lower for w in ["awesome", "great", "nice", "cool", "spectacular", "love"]):
            excitement = 0.8

        # 5. Satisfaction score
        satisfaction = 0.5
        if excitement > 0.5:
            satisfaction = 0.9
        elif frustration > 0.5:
            satisfaction = 0.2

        # 6. Skill Profiling (beginner vs expert)
        skill = "standard"
        if any(w in text_lower for w in ["how to run", "what is", "beginner", "newbie", "help me learn", "step by step"]):
            skill = "beginner"
        elif any(w in text_lower for w in ["ast", "gnn", "backprop", "gitpython", "refactor", "complexity", "optimization", "neural", "embeddings"]):
            skill = "expert"

        # Determine communication style instructions
        if skill == "beginner":
            style_guide = "Provide detailed step-by-step guidance and explain basic programming terms carefully."
        elif skill == "expert":
            style_guide = "Skip basic explanations. Highlight advanced architectural patterns, memory matches, and performance metrics."
        else:
            style_guide = "Standard professional communication style."

        if frustration > 0.6:
            style_guide += " Focus on short, precise explanations to resolve issues immediately."

        return {
            "frustration": frustration,
            "confusion": confusion,
            "urgency": urgency,
            "excitement": excitement,
            "satisfaction": satisfaction,
            "skill_level": skill,
            "style_guide": style_guide
        }
