"""The prompts. Two conditions (base, policy) for each of the two ways an LLM can pick a tool."""

from __future__ import annotations

# The LLM as a router: the tool definitions are pasted into the system prompt as JSON and the
# model answers with one tool name. Jev gets JEV_INSTRUCTIONS with the names and descriptions as
# the options of a `choice` question.
ROUTER_SYSTEM = (
    "You are the tool router for an online store's customer-support assistant. The store runs on "
    "Shopify, takes payments through Stripe and uses Intercom as its helpdesk; its staff also use "
    "GitHub and Notion. Pick the ONE tool that should be called first for the customer's message. "
    "If none of the tools can do what the customer asks, pick hand_off_to_human. Reply with only the "
    "tool's exact name, nothing else.\n\n"
)
ROUTER_TOOLS_HEADER = "Tools (MCP tools/list JSON):\n"
JEV_INSTRUCTIONS = (
    "Which one tool should an online store's customer-support assistant call first for this customer "
    "message? Pick hand_off_to_human if no tool can do what the customer asks."
)

# Native tool calling: a plain assistant prompt; the tools go in the API's `tools` parameter.
NATIVE_SYSTEM = (
    "You are the customer-support assistant for an online store. The store runs on Shopify, takes "
    "payments through Stripe and uses Intercom as its helpdesk; its staff also use GitHub and Notion. "
    "Help the customer using the tools available to you. If none of the tools can do what the "
    "customer asks, call hand_off_to_human."
)

# The routing policy: three rules that say what the labels assume. The `policy` condition adds
# this to every prompt above.
POLICY = (
    "Routing policy:\n"
    "1. The assistant serves the store's customers and may use the store's Shopify and Stripe tools to look up "
    "or act on that customer's orders, payments, refunds and invoices. If a tool can do what the customer wants, "
    "call it, even when the message is phrased as 'how do I ...?'.\n"
    "2. Search the store's policies and FAQs only for questions about store policy: refunds, delivery, payment "
    "methods, fees, contact details.\n"
    "3. If no tool can do what the customer wants (for example cancelling or editing a placed order, account "
    "changes, complaints, newsletters, reviews), pick hand_off_to_human. Don't search the FAQ instead.\n\n"
)


def router_system(tools_json: str, policy: bool) -> str:
    return ROUTER_SYSTEM + (POLICY if policy else "") + ROUTER_TOOLS_HEADER + tools_json


def jev_instructions(policy: bool) -> str:
    return (POLICY + JEV_INSTRUCTIONS) if policy else JEV_INSTRUCTIONS


def native_system(policy: bool) -> str:
    return NATIVE_SYSTEM + ("\n\n" + POLICY.rstrip() if policy else "")
