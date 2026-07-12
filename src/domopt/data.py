from dataclasses import dataclass
from typing import Hashable

OrderId = Hashable
SkuId = Hashable
DcId = Hashable


@dataclass(frozen=True)
class OrderLine:
    order_id: OrderId
    sku_id: SkuId
    demand_cases: int
    value_per_case: float
    penalty_per_unfilled_case: float


@dataclass(frozen=True)
class AssignmentCandidate:
    order_id: OrderId
    dc_id: DcId
    pgi_date: str
    shipping_cost: float
    is_default: bool
    eligible: bool = True


@dataclass(frozen=True)
class InventoryRecord:
    dc_id: DcId
    sku_id: SkuId
    pgi_date: str
    available_cases: int
