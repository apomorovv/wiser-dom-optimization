from domopt.synthetic import make_synthetic_problem


def test_synthetic_generator_is_reproducible() -> None:
    first = make_synthetic_problem(order_count=8, seed=3)
    second = make_synthetic_problem(order_count=8, seed=3)
    assert first.orders.equals(second.orders)
    assert first.order_lines.equals(second.order_lines)
    assert first.candidates.equals(second.candidates)
