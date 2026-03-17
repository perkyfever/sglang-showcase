import sglang as sgl
import re
import logging

logging.basicConfig(level=logging.INFO)
sgl.set_default_backend(sgl.RuntimeEndpoint("http://127.0.0.1:30000"))


@sgl.function
def rag_multihop(s, question: str, docs: list, top_k: int = 3):
    s += "You are an assistant that answers user questions using provided evidence documents.\n"
    s += "For each document extract 1-3 concise assertions relevant to the question and then a relevance score 0-100.\n\n"
    s += f"QUESTION: {question}\n\n"

    forks = s.fork(len(docs))
    for i, f in enumerate(forks):
        f += f"--- DOCUMENT #{i + 1} ---\n{docs[i]}\n\n"
        f += "Extract 1-3 short assertions (one per line). Then on a new line output 'RELEVANCE: <integer 0-100>'.\n\n"
        f += sgl.gen(f"assertions_{i}", max_tokens=80, temperature=0.0)
        f += sgl.gen(f"relevance_{i}", max_tokens=4, temperature=0.0, regex=r"\d{1,3}")

    s += "\n=== EVIDENCE FROM DOCUMENTS (auto-extracted) ===\n"
    for i in range(len(docs)):
        s += f"\nDocument #{i + 1} assertions:\n"
        s += forks[i][f"assertions_{i}"]
        s += "\nRelevance score: "
        s += forks[i][f"relevance_{i}"]
        s += "\n"

    s += (
        "\nNow, based only on the 'Relevance score' lines above, output the TOP DOCUMENT INDICES "
        f"separated by commas (highest relevance first). Limit to top {top_k}. "
        "Output exactly like: TOP_DOCS: 2,1,4 (no extra text on the line).\n"
    )
    s += sgl.gen(
        "top_docs", max_tokens=20, temperature=0.0, regex=r"TOP_DOCS:\s*[0-9,\s]+"
    )
    s += (
        "\nUsing only the assertions from the TOP_DOCS above, synthesize a concise final answer "
        "(1-3 sentences). Then output exactly the following block once:\n"
        "BEGIN_FINAL_ANSWER\n"
        "ANSWER: <one-line answer>\n"
        "SOURCES: <comma-separated indices>\n"
        "CONFIDENCE: <low|medium|high>\n"
        "END_FINAL_ANSWER\n"
        "Do not print anything else outside the BEGIN...END block.\n"
    )
    s += sgl.gen("final_answer", max_tokens=220, temperature=0.0)

    return s


def dedupe_adjacent_lines(text: str) -> str:
    if not text:
        return text or ""
    lines = text.splitlines()
    out = []
    prev = None
    for ln in lines:
        ln_s = ln.rstrip()
        if ln_s == prev:
            continue
        out.append(ln_s)
        prev = ln_s
    return "\n".join(out)


def extract_top_docs_from_string(s: str):
    if not s:
        return None

    m = re.search(r"TOP_DOCS:\s*([0-9,\s]+)", s, flags=re.IGNORECASE)
    if m:
        raw = m.group(1)
        nums = [int(x) for x in re.findall(r"\d+", raw)]
        return nums

    m2 = re.search(r"SOURCES?:\s*([0-9,\s]+)", s, flags=re.IGNORECASE)
    if m2:
        nums = [int(x) for x in re.findall(r"\d+", m2.group(1))]
        return nums

    return None


def extract_final_from_block(s: str):
    if not s:
        return None

    m = re.search(
        r"BEGIN_FINAL_ANSWER(.*?)END_FINAL_ANSWER", s, flags=re.IGNORECASE | re.S
    )
    if m:
        block = m.group(1).strip()
        block = dedupe_adjacent_lines(block)
        return block

    m2 = re.search(
        r"(ANSWER:.*?SOURCES?:.*?(?:CONFIDENCE:.*)?)", s, flags=re.IGNORECASE | re.S
    )
    if m2:
        return dedupe_adjacent_lines(m2.group(1).strip())

    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    return "\n".join(lines[:6]) if lines else None


def demo():
    question = "What is the battery life and waterproof rating of the NexSound X1, and what is the charging time?"
    docs = [
        "Product spec: NexSound X1 bluetooth speaker (2024). Battery: up to 20 hours playtime. IPX7 waterproof rating. Charger: USB-C, typical charge in 3 hours.",
        "Retail listing: NexSound X1 — battery life advertised as 'around 18-22 hours depending on volume'. Waterproof rating shown as IPX7 in product shots. Charging time not specified.",
        "Forum post: I used NexSound X1 for 30 minutes in a pool and it survived, seller page mentioned IPX6 for some regions.",
        "Quick start guide excerpt: Battery 20h nominal, charging 3 hours. Waterproof: IPX7 (do not submerge for long periods).",
    ]

    state = rag_multihop.run(question=question, docs=docs, top_k=3)
    top_raw = state["top_docs"]
    top_raw = top_raw.strip()

    top_nums = extract_top_docs_from_string(top_raw)
    final_raw = state["final_answer"]

    if not top_nums and final_raw:
        top_nums = extract_top_docs_from_string(final_raw)

    cleaned_final = extract_final_from_block(final_raw or "")

    print("\n===# TOP_DOCS #===")
    print(top_nums)

    print("\n===# FINAL CLEAN ANSWER #===")
    print(cleaned_final)


if __name__ == "__main__":
    demo()
