"""The request builder used for every native call (no network)."""

import toolpick as tp
from toolpick.live import MODELS, native_request, native_tools
from toolpick.prompts import POLICY, native_system


def test_function_names_have_no_dots_and_map_back():
    fn_tools, back = native_tools(tp.tools(customer_words=True))
    assert len(fn_tools) == 91
    assert all("." not in t["function"]["name"] for t in fn_tools)
    assert set(back.values()) == {t["name"] for t in tp.tools()}


def test_native_request_is_the_final_setup():
    kwargs, _ = native_request("sonnet", "where is my order", tp.tools(customer_words=True), native_system(True))
    assert kwargs["model"] == MODELS["sonnet"]
    assert kwargs["temperature"] == 0 and kwargs["tool_choice"] == "auto"
    assert POLICY.strip() in kwargs["messages"][0]["content"][0]["text"]
    assert kwargs["messages"][1] == {"role": "user", "content": "where is my order"}
