#!/usr/bin/env python3
"""FTEC5660 HW1 student starter: build a chain for supermarket receipts."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


QUERY_1 = "How much money did I spend in total for these bills?"
QUERY_2 = "How much would I have had to pay without the discount?"
QUERIES = (QUERY_1, QUERY_2)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
DUMMY_RESPONSE = "please design your chain to answer these two queries."


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def image_files(folder: Path) -> list[Path]:
    """Return supported images directly inside *folder*, sorted by filename."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_data_url(path: Path) -> str:
    """Encode a local image in the format accepted by a multimodal prompt."""
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_chain() -> Any:
    """Create and return your LangChain chain once.

    Suggested imports:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_deepseek import ChatDeepSeek

    Use the vision-capable DeepSeek Flash model named
    ``deepseek-v4-flash-vision-exp``. The API key is loaded from .env.
    """
    ### YOUR CODE HERE
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_deepseek import ChatDeepSeek

    model = ChatDeepSeek(
        model="deepseek-v4-flash-vision-exp",
        temperature=0,
        max_retries=2,
    )

    system_prompt = (
        "You are a precise receipt-reading assistant. You are shown ONE "
        "supermarket receipt image from Hong Kong. Read it carefully and reply "
        "with ONLY a single JSON object (no prose, no markdown, no code fence) "
        "containing exactly these three keys:\n"
        '  "amount_paid_after_rounding": the final amount the customer actually '
        "paid, i.e. the value on the payment line (OCTOPUS / EPS / CASH / VISA / "
        "etc.) AFTER the ROUNDING line has been applied.\n"
        '  "subtotal": the value printed on the SUBTOTAL line, i.e. after every '
        "discount has been subtracted but BEFORE rounding.\n"
        '  "discount_total": the sum of the absolute values of EVERY discount, '
        "promotion, coupon, member, app or packaging-damage line on the receipt "
        "(all the negative lines above SUBTOTAL). Use 0 when there are none.\n"
        "All values are plain numbers in HKD with two decimals: no currency "
        "symbols, no thousands separators, no percent signs.\n"
        "Rules:\n"
        "- Do NOT add the ROUNDING line back into subtotal or discount_total.\n"
        "- If the SUBTOTAL line is missing, use the sum of the item prices minus "
        "the discounts.\n"
        "- If a value is genuinely absent, use 0.\n"
        '- Example: items 10.00 + 36.90 + 60.80, "5% OFF -5.39", '
        '"SUBTOTAL 102.31", "ROUNDING -0.01", "OCTOPUS 102.30" -> '
        '{{"amount_paid_after_rounding": 102.30, "subtotal": 102.31, '
        '"discount_total": 5.39}}.\n'
        "Reply with the JSON object only."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                [
                    {
                        "type": "text",
                        "text": "Read this receipt and return the JSON object.",
                    },
                    {"type": "image_url", "image_url": {"url": "{image_url}"}},
                ],
            ),
        ]
    )

    return prompt | model


def answer_queries(chain: Any, images: list[Path]) -> dict[str, Any]:
    """Run your chain and return one response for each exact query string.

    ``images`` contains every receipt in the selected folder. A valid return
    value looks like:

        {QUERY_1: "HK$123.40", QUERY_2: "HK$150.00"}

    Use the provided ``image_data_url(path)`` helper to put local images in
    multimodal human messages. LangChain's ``batch`` method is one simple way
    to process independent receipt-extraction prompts in parallel.
    """
    ### YOUR CODE HERE
    inputs = [{"image_url": image_data_url(path)} for path in images]
    responses = chain.batch(inputs) if inputs else []

    def _to_amount(raw: Any) -> Decimal:
        """Turn "HK$1,234.50" / 1234.5 / "1234.50" into a Decimal."""
        text = str(raw).replace("HK", "").replace("$", "").replace(",", "")
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        if not match:
            return Decimal("0")
        try:
            return Decimal(match.group(0))
        except InvalidOperation:
            return Decimal("0")

    def _parse_receipt(value: Any) -> dict[str, Decimal]:
        """Extract the three fields from one model response, tolerating noise."""
        text = response_text(value)
        data: dict[str, Any] = {}
        block = re.search(r"\{.*\}", text, re.DOTALL)
        if block:
            try:
                parsed = json.loads(block.group(0))
                if isinstance(parsed, dict):
                    data = parsed
            except json.JSONDecodeError:
                data = {}
        if not data:
            for key in (
                "amount_paid_after_rounding",
                "subtotal",
                "discount_total",
            ):
                field = re.search(rf'{key}"?\s*:\s*"?([-\d.,]+)', text)
                if field:
                    data[key] = field.group(1)
        return {
            "amount_paid_after_rounding": _to_amount(
                data.get("amount_paid_after_rounding", 0)
            ),
            "subtotal": _to_amount(data.get("subtotal", 0)),
            "discount_total": _to_amount(data.get("discount_total", 0)),
        }

    # Query 1: sum of the final payments after rounding.
    # Query 2: sum of (SUBTOTAL + every discount line added back positively).
    total_paid = Decimal("0")
    total_without_discount = Decimal("0")
    for value in responses:
        receipt = _parse_receipt(value)
        total_paid += receipt["amount_paid_after_rounding"]
        total_without_discount += receipt["subtotal"] + receipt["discount_total"]

    total_paid = total_paid.quantize(Decimal("0.01"))
    total_without_discount = total_without_discount.quantize(Decimal("0.01"))

    return {
        QUERY_1: f"HK${total_paid:.2f}",
        QUERY_2: f"HK${total_without_discount:.2f}",
    }


# Everything below is provided runner/scoring code. No edits are needed.

_MONEY_RE = re.compile(
    r"(?<![\w.])(?:HK\$|\$)?\s*(-?\d[\d,]*(?:\.\d+)?)(?![\w.])",
    re.IGNORECASE,
)


def response_text(value: Any) -> str:
    """Convert common LangChain response shapes to text for results.csv."""
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content).strip()


def parse_single_amount(text: str) -> Decimal | None:
    """Accept a response only when it contains exactly one numeric amount."""
    matches = _MONEY_RE.findall(text)
    if len(matches) != 1:
        return None
    try:
        return Decimal(matches[0].replace(",", "")).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def read_ground_truth(folder: Path) -> dict[str, Decimal]:
    """Read aggregate answers from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    answers = data.get("answers", data)
    return {query: Decimal(str(answers[query])).quantize(Decimal("0.01")) for query in QUERIES}


def correctness_text(response: str, expected: Decimal | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if expected is None:
        return "not graded: ground_truth.json is missing"
    predicted = parse_single_amount(response)
    if predicted == expected:
        return "correct"
    shown = f"HK${predicted:.2f}" if predicted is not None else repr(response)
    return f"incorrect: expected HK${expected:.2f}, predicted {shown}"


def write_results(responses: dict[str, Any], truth: dict[str, Decimal]) -> Path:
    """Write the required three-column results.csv file."""
    output = Path("results.csv")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query", "model_response", "correctness"])
        for query in QUERIES:
            text = response_text(responses.get(query, "<missing response>"))
            writer.writerow([query, text, correctness_text(text, truth.get(query))])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW1 on receipt images")
    parser.add_argument(
        "--image-folder",
        required=True,
        type=Path,
        help="folder containing supermarket receipt images",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image_folder.is_dir():
        raise SystemExit(f"not a folder: {args.image_folder}")

    images = image_files(args.image_folder)
    if not images:
        raise SystemExit(f"no supported images found in {args.image_folder}")

    load_env_file()
    chain = build_chain()
    responses = answer_queries(chain, images)
    if not isinstance(responses, dict):
        raise TypeError("answer_queries() must return a dictionary")

    output = write_results(responses, read_ground_truth(args.image_folder))
    print(f"Processed {len(images)} receipt(s). Wrote {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
