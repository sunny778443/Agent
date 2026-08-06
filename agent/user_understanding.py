"""
User Understanding, Emotion, and Skill Profiling module.
Estimates a complete multi-dimensional spectrum of human emotions:
Anger, Frustration, Disappointment, Curiosity, Skepticism, Anxiety, Gratitude,
Happiness, Excitement, Impatience, Urgency, Boredom, Pride, and overall Satisfaction.
"""
from typing import Any


class UserUnderstandingModel:
    """
    Profiles incoming user requests to estimate an exhaustive range of human emotions
    to dynamically calibrate communication styles and planning granularity.
    """
    def __init__(self) -> None:
        pass

    def profile_user_request(self, text: str) -> dict[str, Any]:
        """
        Analyzes task text strings from scratch using heuristic matching,
        returning an estimated emotion dictionary and communication style guide.
        """
        text_lower = text.lower()

        # 1. Anger / Frustration
        anger = 0.05
        if any(w in text_lower for w in ["fail", "error", "broken", "stop", "stupid", "wrong", "blank", "unusable", "hate", "bad"]):
            anger = 0.8
        frustration = anger

        # 2. Disappointment
        disappointment = 0.05
        if any(w in text_lower for w in ["sad", "unfortunate", "pity", "disappointed", "regret", "alas"]):
            disappointment = 0.75

        # 3. Curiosity / Inquisitiveness
        curiosity = 0.05
        if any(w in text_lower for w in ["why", "how", "what", "where", "explain", "learn", "wonder"]):
            curiosity = 0.85

        # 4. Skepticism / Doubt
        skepticism = 0.05
        if any(w in text_lower for w in ["doubt", "unsure", "maybe", "really", "verify", "check", "confirm", "proof"]):
            skepticism = 0.7

        # 5. Anxiety / Panic
        anxiety = 0.05
        if any(w in text_lower for w in ["panic", "scared", "worried", "anxious", "fear", "afraid", "leak", "security"]):
            anxiety = 0.8

        # 6. Gratitude / Appreciation
        gratitude = 0.05
        if any(w in text_lower for w in ["thank", "thanks", "appreciate", "kind", "helpful", "good job", "perfect"]):
            gratitude = 0.9

        # 7. Happiness / Excitement
        happiness = 0.05
        if any(w in text_lower for w in ["awesome", "great", "nice", "cool", "spectacular", "love", "happy", "joy"]):
            happiness = 0.85
        excitement = happiness

        # 8. Impatience / Urgency
        impatience = 0.05
        if any(w in text_lower for w in ["asap", "now", "urgent", "quick", "fast", "immediately", "blocker", "emergency", "hurry"]):
            impatience = 0.9
        urgency = impatience

        # 9. Boredom / Indifference
        boredom = 0.05
        if any(w in text_lower for w in ["whatever", "meh", "bored", "slow", "tedious"]):
            boredom = 0.7

        # 10. Pride / Confidence
        pride = 0.05
        if any(w in text_lower for w in ["expert", "master", "achieved", "proud", "triumph"]):
            pride = 0.75

        # 11. Overall Satisfaction score
        satisfaction = 0.5
        if happiness > 0.5 or gratitude > 0.5:
            satisfaction = 0.9
        elif anger > 0.5 or disappointment > 0.5:
            satisfaction = 0.2

        # Skill Profiling (beginner vs expert)
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

        if anger > 0.6:
            style_guide += " Focus on short, precise explanations to resolve issues immediately."

        return {
            "anger": anger,
            "frustration": frustration,
            "disappointment": disappointment,
            "curiosity": curiosity,
            "skepticism": skepticism,
            "anxiety": anxiety,
            "gratitude": gratitude,
            "happiness": happiness,
            "excitement": excitement,
            "impatience": impatience,
            "urgency": urgency,
            "boredom": boredom,
            "pride": pride,
            "satisfaction": satisfaction,
            "skill_level": skill,
            "style_guide": style_guide
        }
