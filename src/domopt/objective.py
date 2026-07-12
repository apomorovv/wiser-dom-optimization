from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectiveBreakdown:
    fulfilled_value: float
    penalty_cost: float
    shipping_cost: float

    @property
    def objective_value(self) -> float:
        return (self.fulfilled_value - self.penalty_cost - self.shipping_cost)
