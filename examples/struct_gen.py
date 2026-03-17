import json
import logging
import sglang as sgl

logging.basicConfig(level=logging.INFO)

sgl.set_default_backend(sgl.RuntimeEndpoint("http://127.0.0.1:30000"))

PRODUCT_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "brand": {"type": "string"},
        "price": {"type": "number"},
        "currency": {"type": "string"},
        "features": {"type": "array", "items": {"type": "string"}},
        "in_stock": {"type": "boolean"},
    },
    "required": ["title", "brand", "price", "currency"],
}

PRODUCT_SCHEMA_STR = json.dumps(PRODUCT_SCHEMA, sort_keys=True, ensure_ascii=False)


@sgl.function
def extract_product(s, description: str):
    s += "You are an assistant that extracts structured product metadata.\n"
    s += f"Product description:\n{description}\n\n"
    s += "Output the product metadata as JSON that exactly matches the required schema.\n"
    s += "JSON:\n"
    s += sgl.gen(
        "product_json", max_tokens=256, json_schema=PRODUCT_SCHEMA_STR, temperature=0.0
    )
    return s


def demo_single():
    desc = (
        "The Acme UltraVac 3000 is a lightweight cordless vacuum from Acme Corp. "
        "It comes with a 60-minute battery, HEPA filter, and three detachable heads. "
        "Price: 199.99 USD. Available now."
    )

    state = extract_product.run(description=desc)
    raw = state["product_json"]
    print("===# SGLang Output #===")
    print(raw)
    print()

    data = json.loads(raw)
    print("===# Parsed JSON #===")
    print(json.dumps(data, indent=2, ensure_ascii=False))


def demo_batch():
    descriptions = [
        "EcoBrew Portable is a mini espresso maker from BrewCo. 89 EUR. Comes with two filters. In stock.",
        "NexSound X1 bluetooth speaker by NexSound — 49.5 GBP, IPX7 waterproof, battery 20h.",
    ]

    states = extract_product.run_batch(
        [{"description": d} for d in descriptions], progress_bar=False
    )

    for i, st in enumerate(states, start=1):
        raw = st["product_json"]
        print(f"===# item {i} #===")
        parsed = json.loads(raw)
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
        print()


if __name__ == "__main__":
    demo_single()
    print("\n--- BATCHES DEMO ---\n")
    demo_batch()
