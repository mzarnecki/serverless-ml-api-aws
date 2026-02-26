import json
import re
import numpy as np
import tensorflow as tf

# --- preprocessing replicated from the repo ---
INVALID_LETTERS_PATTERN = r"[^a-z0-9\s\'\-\.\&]"
MULTI_SPACES_PATTERN = r"\s+"
TOKENIZED_LENGTH = 120
PAD_CHAR_CODE = ord("¬")  # used in repo padding :contentReference[oaicite:2]{index=2}


def remove_invalid_characters(input_text: str) -> str:
    s = re.sub(INVALID_LETTERS_PATTERN, " ", input_text.lower())
    s = re.sub(MULTI_SPACES_PATTERN, " ", s).strip()
    return s


def encode_to_ord_list(s: str) -> list[int]:
    return [ord(ch) for ch in s]


def pad_tokens(tokenized: list[int], total_length: int) -> list[int]:
    diff = len(tokenized) - total_length
    if diff == 0:
        return tokenized
    if diff < 0:
        # repo pads at the front with '¬' :contentReference[oaicite:3]{index=3}
        return [PAD_CHAR_CODE] * (-diff) + tokenized
    # truncate from the end (keep last N) :contentReference[oaicite:4]{index=4}
    return tokenized[-total_length:]


def parse_text(event) -> str | None:
    # direct invoke: {"text": "..."}
    if isinstance(event, dict) and isinstance(event.get("text"), str):
        return event["text"]

    # API Gateway proxy: {"body": "{\"text\":\"...\"}"}
    if isinstance(event, dict) and event.get("body") is not None:
        body = event["body"]
        if isinstance(body, str):
            try:
                body = json.loads(body)
            except json.JSONDecodeError:
                return None
        if isinstance(body, dict) and isinstance(body.get("text"), str):
            return body["text"]

    return None


# --- model load once per container (cold start) ---
MODEL_PATH = "trained_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)


def lambda_handler(event, context):
    cors_headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Allow-Methods": "OPTIONS,POST",
        "Content-Type": "application/json",
    }

    # Preflight
    if isinstance(event, dict) and event.get("httpMethod") == "OPTIONS":
        return {"statusCode": 200, "headers": cors_headers, "body": json.dumps({"ok": True})}

    text = parse_text(event)
    if not text or not text.strip():
        return {"statusCode": 400, "headers": cors_headers,
                "body": json.dumps({"error": "Missing or empty 'text' field"})}

    cleaned = remove_invalid_characters(text)
    encoded = encode_to_ord_list(cleaned)
    padded = pad_tokens(encoded, TOKENIZED_LENGTH)

    x = np.array([padded], dtype=np.int32)  # shape (1, 120)
    p = float(model.predict(x, verbose=0)[0][0])  # sigmoid probability

    # Model meaning (per repo): 1 = business, 0 = individual :contentReference[oaicite:5]{index=5}
    label = "business" if p >= 0.5 else "individual"

    return {
        "statusCode": 200,
        "headers": cors_headers,
        "body": json.dumps({
            "input": text,
            "cleaned": cleaned,
            "classification": label,
            "business_probability": round(p, 4)
        })
    }
