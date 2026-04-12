from env.graders.grader_classify import ClassificationGrader
from env.graders.grader_itc import ITCGrader
from env.graders.grader_filing import FilingGrader

__all__ = ["ClassificationGrader", "ITCGrader", "FilingGrader"]

# Composite episode weights (Task 1: 20%, Task 2: 35%, Task 3: 45%)
TASK_WEIGHTS = {1: 0.20, 2: 0.35, 3: 0.45}


def compute_episode_score(task_scores: dict[int, float]) -> float:
    """Weighted composite score ∈ [0.0, 1.0]."""
    return round(sum(TASK_WEIGHTS[t] * s for t, s in task_scores.items()), 4)
