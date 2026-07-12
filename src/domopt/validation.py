from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    is_feasible: bool
    assignment_violations: list[str] = field(default_factory=list)
    demand_violations: list[str] = field(default_factory=list)
    inventory_violations: list[str] = field(default_factory=list)
    eligibility_violations: list[str] = field(default_factory=list)

    @property
    def violations(self) -> list[str]:
        return (
            self.assignment_violations
            + self.demand_violations
            + self.inventory_violations
            + self.eligibility_violations
        )
