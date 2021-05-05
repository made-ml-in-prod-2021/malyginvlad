from dataclasses import dataclass, field


@dataclass()
class PreprocessingParams:
    scaler: str = field(default='StandardScaler')


@dataclass()
class SplittingParams:
    test_size: float = field(default=0.2)
    random_state: int = field(default=13)
