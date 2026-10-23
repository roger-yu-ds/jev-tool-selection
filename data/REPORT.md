# Real MCP tool catalogue for an e-commerce support assistant

Retrieved 2026-10-22. Files: `catalog.json` (90 tools), `intent_map.json` (27 Bitext intents), raw captures in `src/`, scripts in `scripts/`.

## Servers in the catalogue

| server | tools | official? | how obtained | exactness | licence |
|---|---|---|---|---|---|
| shopify_storefront (`https://{shop}/api/mcp`) | 1 | Yes (Shopify-hosted on every store) | live `tools/list` on allbirds.com; same on kith, skims, brooklinen | exact name, description, inputSchema | Proprietary (Shopify terms); no open licence |
| shopify_ucp (`https://{shop}/api/ucp/mcp`, UCP 2026-08-25) | 15 | Yes (Shopify implementation of the Universal Commerce Protocol) | live `tools/list` on allbirds.com (found through `/.well-known/ucp`) | exact; outputSchema dropped | Tool text is Shopify's (proprietary); the UCP spec is Apache-2.0 |
| shopify_customer_account (`/customer/api/mcp`) | 4 | Yes | live `tools/list` without auth on accounts.allbirds.com; kith.com returns the same 4 | exact | Proprietary |
| stripe (`@stripe/mcp@0.2.5`, agent-toolkit 0.7.12) | 22 | Yes, but a **legacy** version | ran the official npm package locally over stdio with a placeholder key string, then `tools/list` | exact for that version | MIT |
| intercom (`mcp.intercom.com`) | 14 | Yes (Intercom-hosted) | official docs page developers.intercom.com/docs/guides/mcp | **docs summary only**: live description text not verified, input_schema = null | No licence stated |
| github (`github-mcp-server` v2.0.3, `issues` toolset) | 10 | Yes | official docker image (rev b98d115, 2026-10-21) with `--network none` and a placeholder token, then `tools/list` | exact | MIT |
| notion (`@notionhq/notion-mcp-server@2.5.2`, local) | 24 | Yes (makenotion org) | ran the official npm package locally with a placeholder token, then `tools/list` | exact | MIT |

No real API keys were used anywhere. The placeholder strings `sk_test_PLACEHOLDER_NOT_A_KEY` and `placeholder_not_a_token` only got past the startup format checks. The only live network calls were unauthenticated `tools/list` POSTs and `.well-known` GETs against public Shopify stores.

## Important caveats

- **Shopify catalog and cart tools moved.** The old Storefront MCP tools (`search_shop_catalog`, `get_cart`, `update_cart`, `get_product_details`) were removed from `/api/mcp`. Only `search_shop_policies_and_faqs` is left there. The replacements now live on the UCP endpoint, according to shopify.dev and confirmed live.
- **The UCP tool set varies by store.** `search_locations` and `lookup_locations` appear on allbirds.com but not on kith.com.
- **Customer Account MCP was listed without auth.** It may expose more tools once a customer is authenticated, which we could not check. shopify.dev documents no tool list for it and says to use `tools/list`.
- **The Stripe toolset is legacy.** The current official server is the hosted `https://mcp.stripe.com` (it needs auth and returned 401 to us). Per docs.stripe.com/mcp it now exposes generic tools: `stripe_api_search`, `stripe_api_details`, `stripe_api_read`, `stripe_api_write`, `get_stripe_account_info`, `search_stripe_documentation`, `stripe_implementation_planner`, `send_stripe_feedback`, plus Atlas, Analytics and Treasury tools. Refunds, invoices and similar actions are reached through the generic read/write tools.
  - Those docs-table descriptions are summaries, not `tools/list` text, so they are excluded from the catalogue.
  - We used the last published per-tool package instead. Its source was deleted from the repo in commit ed67c1f on 2026-02-05.
  - If the eval must reflect the current Stripe server, per-intent tool selection against Stripe collapses to `stripe_api_read` or `stripe_api_write`.
- **Stripe text quirks are kept verbatim.** `create_refund` says it "takes three arguments" (including `reason`), but its schema has only `payment_intent` and `amount`. `create_invoice_item` contains a stray `'}` in its description.
- **Intercom descriptions are the docs' summary lines.** The real `tools/list` descriptions are probably longer and could differ. Intercom also lists 5 conditional Fin Operator tools, which we excluded.
- **Notion descriptions are thin.** They are auto-generated from the OpenAPI spec (e.g. "Notion | Search by title ..."). Notion's hosted remote server (`mcp.notion.com`) has different tools but needs auth. The local server is the one we could reproduce exactly.
- **GitHub is a subset.** We used only the `issues` toolset. The full server has about 140 tools (see `pkg/github/__toolsnaps__`).

## Servers checked but not included

| vendor | finding |
|---|---|
| Zendesk | No shipped general-purpose official MCP server found. A server was announced at Relate in May 2026. Zendesk has an MCP *client* and a ChatGPT-channel server in early access. Every Zendesk MCP server found is third-party (Workato, community repos). |
| Gorgias | No official server found; community only. |
| Freshdesk | Only a third-party blog claims an Early Access official server. Not verified. |
| HubSpot | An official remote server exists (`mcp.hubspot.com`, returned 401). Tool names and descriptions are not published in docs or public repos (HubSpot/mcp-server has no licence and is the older developer server; hubspot-mcp-plugins holds skills, not tool definitions). |
| Shopify Admin | No official merchant Admin MCP server found. `@shopify/dev-mcp` is a developer-docs server. |
| Salesforce | Official `salesforcecli/mcp` is a developer (DX) server, not Service Cloud support. Not pursued. |
| Slack (`mcp.slack.com`), Linear, Atlassian, Sentry (`mcp.sentry.dev`) | All require auth (401). Slack's docs give feature prose and tool names, not tool descriptions. Sentry's repo is open (FSL-1.1, Apache-2.0 future licence), but descriptions are assembled from arrays across many files and it was not run. These could be added later by running Sentry's stdio server or with authenticated `tools/list` calls. |

## Intent coverage (see intent_map.json)

Each intent lists `acceptable` (clean fit), `partial` (defensible but imperfect) and `near_miss_distractors`.

- **12 of 27 intents have a clean real tool:**
  - check_cancellation_fee, check_payment_methods, check_refund_policy, contact_customer_service and delivery_options: `search_shop_policies_and_faqs`
  - check_invoice and get_invoice: `stripe.list_invoices`
  - delivery_period: policy tool or order-status tools
  - get_refund: `request_return` or `stripe.create_refund`
  - payment_issue: `stripe.list_payment_intents`
  - place_order: `create_checkout` or `create_cart`
  - track_order: `get_order_status`, `get_most_recent_order_status` or `get_order`
- **15 intents have no clean tool:**
  - cancel_order, change_order, change_shipping_address: post-purchase order edits do not exist in any official customer-side server. UCP checkout tools only cover the pre-purchase case.
  - complaint, contact_human_agent: no official helpdesk server lets the agent create a ticket or hand off. Intercom's MCP is agent-side and read-mostly; `add_internal_note` is the only write.
  - track_refund: partial only.
  - create_account, delete_account, edit_account, recover_password, registration_problems, switch_account, set_up_shipping_address: no account-management tools.
  - newsletter_subscription, review: no server for these.
- **Recommendation:** treat the 15 gap intents as "no suitable tool / abstain or hand off" in the eval, or score them separately. Do not force a mapping.

## Unverified points

- Whether authenticated Customer Account MCP sessions expose more tools (e.g. address or profile editing).
- Whether `get_order_status` or `get_order` surface refund status for real orders.
- The exact live Intercom tool descriptions and schemas.
- The current `mcp.stripe.com` `tools/list` text.
- Whether Zendesk, Gorgias or Freshdesk have shipped official servers after the sources I found (the latest dated about September 2026).
