from dataclasses import dataclass, field


@dataclass()
class TrainingParams:
    criterion: str = field(default="gini")
    n_estimators: int = field(default=100)
    max_depth: int = field(default=4)
    random_state: int = field(default=13)
