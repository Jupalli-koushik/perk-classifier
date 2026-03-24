"""
inference_pipeline.py
=====================
What is an Inference Pipeline?
-------------------------------
An **inference pipeline** is the end-to-end sequence of steps that transforms
a raw input (image, text, audio, …) into a meaningful output (label, answer,
structured data, …) using a trained ML/LLM/vision model.

Think of it as an assembly line:

  Raw Input
     ↓  1. Input Acquisition      – receive/fetch the raw data
     ↓  2. Preprocessing          – resize, normalise, tokenise …
     ↓  3. Model Forward Pass     – run through the neural network
     ↓  4. Postprocessing         – decode logits, decode tokens …
     ↓  5. Thresholding           – apply confidence cut-offs
     ↓  6. Business Logic         – domain rules ("ambiguous" band)
     ↓  7. Output Formatting      – JSON, human-readable string, UI card …
     ↓  8. Logging / Monitoring   – record latency, prediction, confidence
     ↓  (optional) Batching       – process N inputs at once for throughput
     ↓  (optional) Streaming      – yield partial results as they arrive
  Final Output

The two sections below show this for the two models already present in this
repository:

  A. Image classifier  – YOLOv8 perk detector  (see imgclsfy.py)
  B. LLM / RAG         – entity categorisation  (see jfs.py)
"""

# ---------------------------------------------------------------------------
# Standard-library imports used only for the logging/timing helpers below.
# The actual model calls are shown as pseudo-code so the file can be read
# without installing every dependency.
# ---------------------------------------------------------------------------
import json
import logging
import time
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# A.  IMAGE CLASSIFIER INFERENCE PIPELINE  (YOLOv8 perk detector)
# ===========================================================================
#
# Corresponds to the live Gradio app in imgclsfy.py.
# Each stage is annotated so you can map it back to the raw code.

def image_classifier_pipeline(image_input: Any) -> dict:
    """
    Full inference pipeline for the YOLOv8 perk classifier.

    Parameters
    ----------
    image_input : PIL.Image | np.ndarray | str
        Raw image supplied by the user (Gradio passes a numpy array).

    Returns
    -------
    dict with keys: label, confidence, raw_probs, latency_ms
    """

    # ------------------------------------------------------------------
    # 1. INPUT ACQUISITION
    #    Receive the raw image from the caller (Gradio UI, API, batch
    #    job, …).  In imgclsfy.py this is the `inp` argument of
    #    classify_image().
    # ------------------------------------------------------------------
    # image_input = inp                       # already done by Gradio

    # ------------------------------------------------------------------
    # 2. PREPROCESSING
    #    YOLOv8 handles resizing (imgsz=640) and normalisation
    #    (pixel values → 0-1) internally before the forward pass.
    #    If you were using raw PyTorch you would do:
    #
    #      from torchvision import transforms
    #      transform = transforms.Compose([
    #          transforms.Resize((640, 640)),
    #          transforms.ToTensor(),           # 0-255 → 0.0-1.0
    #          transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                               std=[0.229, 0.224, 0.225]),
    #      ])
    #      tensor = transform(image_input).unsqueeze(0)  # add batch dim
    # ------------------------------------------------------------------
    t_start = time.perf_counter()

    # ------------------------------------------------------------------
    # 3. MODEL FORWARD PASS  (pseudo-code — requires ultralytics)
    #
    #    from ultralytics import YOLO
    #    model = YOLO('./runs/classify/train11/weights/best.pt')
    #    results = model(image_input)          # internally preprocesses
    #                                          # and runs the CNN
    # ------------------------------------------------------------------
    # --- Simulated output for illustration purposes -------------------
    simulated_probs = [0.12, 0.88]            # [no_perk, perk]
    simulated_names = {0: "no_perk", 1: "perk"}
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 4. POSTPROCESSING
    #    Extract the raw probability vector from the YOLOv8 results
    #    object.  In imgclsfy.py:
    #
    #      labels_dict = results[0].names          # {0: 'no_perk', 1: 'perk'}
    #      probs       = results[0].probs.data.tolist()   # [0.12, 0.88]
    # ------------------------------------------------------------------
    probs = simulated_probs
    labels_dict = simulated_names

    # ------------------------------------------------------------------
    # 5. THRESHOLDING
    #    Decide whether the model is confident enough to commit to a
    #    label.  Hard thresholds keep the business team in control.
    #
    #    0.75+ → definitive "perk"
    #    0.50–0.75 → "ambiguous" (needs human review)
    #    <0.50  → "no_perk"
    # ------------------------------------------------------------------
    perk_confidence = probs[1]
    if perk_confidence > 0.75:
        raw_label = labels_dict[1]          # "perk"
    elif perk_confidence > 0.50:
        raw_label = "ambiguous"
    else:
        raw_label = labels_dict[0]          # "no_perk"

    # ------------------------------------------------------------------
    # 6. BUSINESS LOGIC
    #    Layer domain rules on top of the model output.
    #    Example: flag near-threshold "ambiguous" results for human
    #    review; automatically approve very high-confidence predictions.
    # ------------------------------------------------------------------
    needs_review = raw_label == "ambiguous"

    # ------------------------------------------------------------------
    # 7. OUTPUT FORMATTING
    #    Return a structured result that downstream systems can consume.
    # ------------------------------------------------------------------
    t_end = time.perf_counter()
    latency_ms = round((t_end - t_start) * 1000, 2)

    output = {
        "label": raw_label,
        "confidence": round(perk_confidence, 4),
        "raw_probs": {labels_dict[i]: round(p, 4) for i, p in enumerate(probs)},
        "needs_review": needs_review,
        "latency_ms": latency_ms,
    }

    # ------------------------------------------------------------------
    # 8. LOGGING / MONITORING
    #    Record every prediction so you can track accuracy drift, spot
    #    edge cases, and audit decisions.
    # ------------------------------------------------------------------
    logger.info(
        "image_classifier | label=%s confidence=%.4f latency_ms=%.2f",
        output["label"],
        output["confidence"],
        output["latency_ms"],
    )

    return output


# ---------------------------------------------------------------------------
# Optional: BATCHING
# ---------------------------------------------------------------------------
def image_classifier_batch_pipeline(image_list: list) -> list:
    """
    Process multiple images in one call.

    YOLOv8 accepts a list of images and runs them as a single batch,
    improving GPU utilisation compared to calling the model in a loop.

    Pseudo-code:
        results = model(image_list)           # list of N images → N results
        return [postprocess(r) for r in results]
    """
    return [image_classifier_pipeline(img) for img in image_list]


# ===========================================================================
# B.  LLM / RAG INFERENCE PIPELINE  (entity categorisation)
# ===========================================================================
#
# Corresponds to jfs.py which calls a TGI (Text Generation Inference) server.

def llm_rag_pipeline(entity_name: str) -> dict:
    """
    Full inference pipeline for entity categorisation via an LLM.

    Parameters
    ----------
    entity_name : str
        Name of the company/entity to classify.

    Returns
    -------
    dict with keys: entity_name, industry, sub_category,
                    market_positioning, reason, latency_ms
    """

    # ------------------------------------------------------------------
    # 1. INPUT ACQUISITION
    #    Receive the entity name (from a CSV row, API call, user input…).
    #    In jfs.py:  entity_names = df["Entity Name"].tolist()
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 2. PREPROCESSING  (Prompt Construction / Retrieval)
    #    For an LLM the "preprocessing" step is building the prompt.
    #    For a RAG system it also includes:
    #      a) Embed the query            → query_vector
    #      b) Search a vector store      → top-k relevant documents
    #      c) Inject docs into prompt    → augmented_prompt
    #
    #    Pseudo-code (RAG extension):
    #      query_vector    = embed_model.encode(entity_name)
    #      retrieved_docs  = vector_store.search(query_vector, top_k=3)
    #      context         = "\n".join(retrieved_docs)
    #      augmented_prompt = base_template.format(
    #                             entity_name=entity_name,
    #                             context=context)
    # ------------------------------------------------------------------
    prompt = f"""
Task: Categorise the following entity.
Entity name: {entity_name}
Provide:
  - Industry
  - Sub Category
  - Market Positioning (economy / premium / luxury)
  - Brief reason (max 10 words)
Return ONLY valid JSON.
"""

    # ------------------------------------------------------------------
    # 3. MODEL FORWARD PASS  (LLM text generation)
    #    Send the prompt to the inference server (OpenAI, TGI, vLLM, …).
    #    In jfs.py this is the POST request to the TGI endpoint.
    #
    #    Pseudo-code:
    #      import requests
    #      response = requests.post(
    #          url="http://<tgi-host>/generate",
    #          json={"inputs": prompt,
    #                "parameters": {"max_new_tokens": 256}},
    #      )
    #      raw_text = response.json()["generated_text"]
    # ------------------------------------------------------------------
    t_start = time.perf_counter()

    # --- Simulated LLM response for illustration ----------------------
    raw_text = json.dumps([{
        "Entity name": entity_name,
        "Industry": "Retail",
        "Sub Category": "E-commerce",
        "Market Positioning": "premium",
        "Brief reason for classification": "Online marketplace targeting mid-high income",
    }])
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 4. POSTPROCESSING  (Parse structured output from free text)
    #    LLMs sometimes wrap JSON in markdown fences (```json … ```).
    #    Strip the fences before parsing.
    #
    #    In jfs.py:
    #      start = json_string.find('```') + 3
    #      end   = json_string.rfind('```')
    #      cleaned = json_string[start:end].strip()
    # ------------------------------------------------------------------
    # Strip optional markdown code fences
    text = raw_text.strip()
    if text.startswith("```"):
        text = text[text.find("\n") + 1:]
    if text.endswith("```"):
        text = text[: text.rfind("```")]

    parsed = json.loads(text.strip())
    entity_data = parsed[0] if isinstance(parsed, list) else parsed

    # ------------------------------------------------------------------
    # 5. THRESHOLDING / VALIDATION
    #    For LLMs "thresholding" means validating that the model
    #    returned the expected fields and that values fall within
    #    known categories.
    # ------------------------------------------------------------------
    valid_positions = {"economy", "premium", "luxury"}
    market_pos = entity_data.get("Market Positioning", "").lower()
    if market_pos not in valid_positions:
        market_pos = "unknown"

    # ------------------------------------------------------------------
    # 6. BUSINESS LOGIC
    #    Apply domain rules: e.g. route "unknown" classifications for
    #    human review, merge sub-category aliases, deduplicate, etc.
    # ------------------------------------------------------------------
    needs_review = market_pos == "unknown"

    # ------------------------------------------------------------------
    # 7. OUTPUT FORMATTING
    # ------------------------------------------------------------------
    t_end = time.perf_counter()
    latency_ms = round((t_end - t_start) * 1000, 2)

    output = {
        "entity_name": entity_data.get("Entity name", entity_name),
        "industry": entity_data.get("Industry", ""),
        "sub_category": entity_data.get("Sub Category", ""),
        "market_positioning": market_pos,
        "reason": entity_data.get("Brief reason for classification", ""),
        "needs_review": needs_review,
        "latency_ms": latency_ms,
    }

    # ------------------------------------------------------------------
    # 8. LOGGING / MONITORING
    # ------------------------------------------------------------------
    logger.info(
        "llm_rag_pipeline | entity=%s positioning=%s latency_ms=%.2f",
        output["entity_name"],
        output["market_positioning"],
        output["latency_ms"],
    )

    return output


# ---------------------------------------------------------------------------
# Optional: STREAMING  (yield tokens as they arrive from the LLM)
# ---------------------------------------------------------------------------
def llm_streaming_pipeline(prompt: str):
    """
    Yield partial text tokens as the LLM generates them.

    This lets a UI display text in real-time instead of waiting for the
    full response, improving perceived latency.

    Pseudo-code (using an OpenAI-compatible streaming API):

        import openai
        client = openai.OpenAI(base_url="http://<tgi-host>/v1")
        stream = client.chat.completions.create(
            model="tgi",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            yield token          # <- send to UI incrementally
    """
    # Simulated streaming for illustration
    tokens = ["Re", "tail", " /", " E-", "com", "mer", "ce"]
    for token in tokens:
        time.sleep(0.01)         # simulate network latency
        yield token


# ===========================================================================
# Quick demo – run `python inference_pipeline.py` to see both pipelines
# ===========================================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("A. IMAGE CLASSIFIER INFERENCE PIPELINE (YOLOv8 perk)")
    print("=" * 60)
    # Pass None as a stand-in for an actual image array
    result_img = image_classifier_pipeline(image_input=None)
    print(json.dumps(result_img, indent=2))

    print("\n" + "=" * 60)
    print("B. LLM / RAG INFERENCE PIPELINE (entity categorisation)")
    print("=" * 60)
    result_llm = llm_rag_pipeline(entity_name="Amazon")
    print(json.dumps(result_llm, indent=2))

    print("\n" + "=" * 60)
    print("C. LLM STREAMING (tokens arrive one at a time)")
    print("=" * 60)
    print("Streaming output: ", end="", flush=True)
    for tok in llm_streaming_pipeline("Classify: Amazon"):
        print(tok, end="", flush=True)
    print()
