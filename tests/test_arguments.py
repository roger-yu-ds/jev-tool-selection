"""The rule-based argument check: schema, made-up identifiers, order numbers carried, cents."""

from toolpick.arguments import check

ORDER_STATUS = "shopify_customer_account.get_order_status"


def test_an_order_number_from_the_message_passes():
    out = check(ORDER_STATUS, {"order_number": "ORD-475206"}, "status of order ORD-475206", "track_order")
    assert out["ok"] and out["carried"]


def test_a_made_up_order_number_fails():
    out = check(ORDER_STATUS, {"order_number": "12345"}, "where is my order", "track_order")
    assert not out["ok"] and not out["grounded"]


def test_an_order_number_left_out_fails():
    out = check("shopify_ucp.get_order", {"id": "unknown"}, "track order ORD-475206", "track_order")
    assert not out["ok"] and out["carried"] is False


def test_stripe_refund_amounts_are_cents():
    text = "refund $181 on pi_3abc please"
    assert check("stripe.create_refund", {"payment_intent": "pi_3abc", "amount": 18100}, text, "get_refund")["ok"]
    assert not check("stripe.create_refund", {"payment_intent": "pi_3abc", "amount": 181}, text, "get_refund")["ok"]


def test_arguments_that_break_the_schema_fail():
    out = check(ORDER_STATUS, {}, "status of order ORD-475206", "track_order")
    assert not out["valid"] and not out["ok"]
    assert not check(ORDER_STATUS, {"_unparsed": "{oops"}, "x", "track_order")["ok"]
