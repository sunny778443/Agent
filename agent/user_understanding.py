"""
User Understanding, Emotion-Estimation, and Skill Profiling module.
Estimates the likelihood of different human emotional states (Anger, Frustration,
Disappointment, Curiosity, Skepticism, Anxiety, Gratitude, Happiness, Excitement,
Impatience, Urgency, Boredom, Pride, and Satisfaction) and skill levels.
NOTE: This is a heuristic linguistic emotion-estimation system, NOT true human-level
affective or cognitive emotion understanding.
"""
import re
import math
from typing import Any


class UserUnderstandingModel:
    """
    Implements a transparent linguistic signal processing pipeline:
        text -> linguistic features -> signal detection -> emotion hypothesis -> confidence -> style recommendation
    Correctly handles negation, signal density, and contradictory indicators from scratch.
    """
    def __init__(self) -> None:
        # Define signal dictionary with keywords
        self.signals = {
            "anger": ["fail", "error", "broken", "stop", "stupid", "wrong", "blank", "unusable", "hate", "bad", "angry", "furious"],
            "disappointment": ["sad", "unfortunate", "pity", "disappointed", "regret", "alas", "disappointment"],
            "curiosity": ["why", "how", "what", "where", "explain", "learn", "wonder", "curious"],
            "skepticism": ["doubt", "unsure", "maybe", "really", "verify", "check", "confirm", "proof", "skeptical"],
            "anxiety": ["panic", "scared", "worried", "anxious", "fear", "afraid", "leak", "security", "leakage"],
            "gratitude": ["thank", "thanks", "appreciate", "kind", "helpful", "good job", "perfect", "gratitude"],
            "happiness": ["awesome", "great", "nice", "cool", "spectacular", "love", "happy", "joy", "excited"],
            "impatience": ["asap", "now", "urgent", "quick", "fast", "immediately", "blocker", "emergency", "hurry", "impatience"],
            "boredom": ["whatever", "meh", "bored", "slow", "tedious", "boring"],
            "pride": ["expert", "master", "achieved", "proud", "triumph", "pride"]
        }
        self.negation_words = {"not", "never", "no", "dont", "cannot", "wont", "neither", "nor"}

    def tokenize_and_clean(self, text: str) -> list[str]:
        """Cleans and splits input string into standard alphabetic tokens."""
        cleaned = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        return cleaned.split()

    def detect_negated_signals(self, text: str) -> dict[str, list[str]]:
        """
        Processes text to identify which signal keywords appear and if they are
        negated by preceding negation tokens within a window of 2 words.
        """
        tokens = self.tokenize_and_clean(text)
        detected_signals: dict[str, list[str]] = {category: [] for category in self.signals}
        negated_signals: dict[str, list[str]] = {category: [] for category in self.signals}

        for idx, token in enumerate(tokens):
            # Check which category this token belongs to
            for category, keywords in self.signals.items():
                if token in keywords:
                    # Check previous 2 words for negation tokens
                    is_negated = False
                    start_look = max(0, idx - 2)
                    for look_idx in range(start_look, idx):
                        if tokens[look_idx] in self.negation_words:
                            is_negated = True
                            break

                    if is_negated:
                        negated_signals[category].append(token)
                    else:
                        detected_signals[category].append(token)

        return {
            "active": detected_signals,
            "negated": negated_signals
        }

    def profile_user_request(self, text: str) -> dict[str, Any]:
        """
        Runs the complete, transparent linguistic pipeline.
        Returns estimated emotion probabilities, active evidence, and confidence.
        """
        # 1. Feature Extraction & Signal Detection
        signal_maps = self.detect_negated_signals(text)
        active_signals = signal_maps["active"]
        negated_signals = signal_maps["negated"]

        # 2. Emotion Hypothesis Scoring
        emotions = {}
        evidence = []
        for category in self.signals:
            base_score = 0.05
            active_hits = active_signals[category]
            negated_hits = negated_signals[category]

            if active_hits:
                # Log logarithmic scaling for repeated signals
                base_score = min(0.95, 0.5 + 0.15 * math.log(len(active_hits) + 1))
                evidence.extend([f"Active segment: '{hit}'" for hit in active_hits])

            if negated_hits:
                # Log negation evidence
                evidence.extend([f"Negated segment: 'NOT {hit}'" for hit in negated_hits])

            emotions[category] = base_score

        # Map aliases for backward compatibility
        emotions["frustration"] = emotions["anger"]
        emotions["excitement"] = emotions["happiness"]
        emotions["urgency"] = emotions["impatience"]

        # 3. Overall Satisfaction scoring
        satisfaction = 0.5
        if emotions["happiness"] > 0.5 or emotions["gratitude"] > 0.5:
            satisfaction = 0.9
        elif emotions["anger"] > 0.5 or emotions["disappointment"] > 0.5:
            satisfaction = 0.2
        emotions["satisfaction"] = satisfaction

        # 4. Confidence Calibration
        # Confidence is high if signals are consistent, and penalized if contradictory
        # (e.g. active anger and active happiness together indicates confusion/ambiguity)
        active_categories = [cat for cat in active_signals if active_signals[cat]]
        num_signals = sum(len(active_signals[cat]) for cat in active_signals)

        contradiction_penalty = 0.0
        if "anger" in active_categories and "happiness" in active_categories:
            contradiction_penalty = 0.4
        if "anger" in active_categories and "gratitude" in active_categories:
            contradiction_penalty = 0.4

        confidence = 0.5
        if num_signals > 0:
            confidence = min(0.99, max(0.1, 0.6 + 0.1 * num_signals - contradiction_penalty))
        else:
            confidence = 0.50 # baseline neutral confidence

        # 5. Skill Profiling (beginner vs expert)
        text_lower = text.lower()
        skill = "standard"
        if any(w in text_lower for w in ["how to run", "what is", "beginner", "newbie", "help me learn", "step by step"]):
            skill = "beginner"
        elif any(w in text_lower for w in ["ast", "gnn", "backprop", "gitpython", "refactor", "complexity", "optimization", "neural", "embeddings"]):
            skill = "expert"

        # 6. Communication Recommendation Formulation
        if skill == "beginner":
            recommendation = "Provide detailed step-by-step guidance and explain basic programming terms carefully."
        elif skill == "expert":
            recommendation = "Skip basic explanations. Highlight advanced architectural patterns, memory matches, and performance metrics."
        else:
            recommendation = "Standard professional communication style."

        if emotions["anger"] > 0.6:
            recommendation += " Focus on short, precise explanations to resolve issues immediately."

        # Package complete pipeline output
        result = {
            "skill_level": skill,
            "style_guide": recommendation,
            "confidence": round(confidence, 2),
            "evidence": evidence,
            "dominant_emotion": max(emotions, key=emotions.get) if num_signals > 0 else "neutral"
        }
        # Merge all emotion scores in
        result.update(emotions)
        return result
