# ETL Pipeline: Concepts and Application in This Repository

## Table of Contents
1. [What Is an ETL Pipeline?](#what-is-an-etl-pipeline)
2. [The Three Stages: Extract, Transform, Load](#the-three-stages-extract-transform-load)
3. [Common Patterns: Batch vs Streaming](#common-patterns-batch-vs-streaming)
4. [Python / pandas Examples](#python--pandas-examples)
5. [ETL vs ELT](#etl-vs-elt)
6. [ETL in This Repository](#etl-in-this-repository)

---

## What Is an ETL Pipeline?

An **ETL pipeline** is a data-engineering workflow that moves data from one or more source systems into a destination (such as a database, file, or dashboard) in a structured, repeatable way.

The acronym stands for:

| Letter | Stage | One-line description |
|--------|-------|----------------------|
| **E** | Extract | Pull raw data from a source |
| **T** | Transform | Clean, reshape, or enrich the data |
| **L** | Load | Write the processed data to a destination |

ETL pipelines are the backbone of data warehouses, machine-learning feature stores, reporting systems, and automated classification workflows.

---

## The Three Stages: Extract, Transform, Load

### Extract
**Extract** is the process of reading raw data from one or more sources. Sources can be:
- Flat files (CSV, JSON, XML)
- Relational databases (PostgreSQL, MySQL)
- REST or GraphQL APIs
- Message queues (Kafka, RabbitMQ)
- Cloud storage buckets (S3, GCS)

The goal of extraction is to get data into memory or a staging area **without changing it** — preserving the original values for auditing and reproducibility.

```python
import pandas as pd

# Extract: read raw entity data from a CSV file
raw_df = pd.read_csv("part3.csv")
print(raw_df.head())
```

---

### Transform
**Transform** is where the real work happens. Transformations can include:
- **Cleaning**: removing nulls, fixing encoding, standardising date formats
- **Filtering**: keeping only rows that meet certain criteria
- **Enriching**: joining with reference data or calling an external API
- **Aggregating**: grouping and summarising rows
- **Reshaping**: pivoting, melting, or normalising table structure

```python
import pandas as pd

def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Drop rows with no entity name
    df = df.dropna(subset=["Entity Name"])

    # Normalise text: strip whitespace, title-case names
    df["Entity Name"] = df["Entity Name"].str.strip().str.title()

    # Add a derived column for downstream use
    df["name_length"] = df["Entity Name"].str.len()

    return df

raw_df = pd.read_csv("part3.csv")
clean_df = transform(raw_df)
```

---

### Load
**Load** writes the transformed data to its destination. Destinations can be:
- Another CSV / Parquet file
- A SQL database table
- A data warehouse (BigQuery, Snowflake, Redshift)
- A REST API endpoint
- A message queue

```python
import pandas as pd

# Load: append the enriched row to the output CSV
def load(row: dict, output_path: str = "updated_entity_data.csv") -> None:
    output_df = pd.DataFrame([row])
    output_df.to_csv(
        output_path,
        mode="a",
        header=not pd.io.common.file_exists(output_path),
        index=False,
    )
    print(f"Row written to {output_path}")
```

---

## Common Patterns: Batch vs Streaming

### Batch Processing
Data is collected over a period, then processed all at once on a schedule (hourly, nightly, weekly).

- **Pros**: simple to implement, easy to debug, low infrastructure cost
- **Cons**: data is never fully real-time; latency equals the batch interval
- **Tools**: pandas, Apache Spark, dbt, cron jobs

```python
import pandas as pd

def run_batch_etl(input_path: str, output_path: str) -> None:
    # Extract
    df = pd.read_csv(input_path)

    # Transform
    df["Entity Name"] = df["Entity Name"].str.strip().str.title()
    df = df.dropna(subset=["Entity Name"])

    # Load
    df.to_csv(output_path, index=False)
    print(f"Batch complete: {len(df)} rows written to {output_path}")

run_batch_etl("part3.csv", "updated_entity_data.csv")
```

**When to use batch**: nightly reports, model retraining runs, end-of-day data aggregations.

---

### Streaming Processing
Each record (or micro-batch of records) is processed immediately as it arrives.

- **Pros**: near-real-time latency, useful for fraud detection, live dashboards
- **Cons**: more complex infrastructure, harder to debug, higher cost
- **Tools**: Apache Kafka + Faust, Apache Flink, Spark Structured Streaming, AWS Kinesis

```python
# Conceptual streaming example using a generator to simulate a live feed
import time

def stream_source(records: list) -> None:
    """Simulate a streaming source that emits one record per second."""
    for record in records:
        yield record
        time.sleep(1)

def process_stream(source) -> None:
    for record in source:
        # Transform: inline, per-record
        entity_name = record.get("Entity Name", "").strip().title()
        print(f"Processed in real-time: {entity_name}")
        # Load: write immediately (e.g., insert into a database)

records = [{"Entity Name": "acme corp"}, {"Entity Name": "globex"}]
process_stream(stream_source(records))
```

**When to use streaming**: live product-feed updates, real-time perk detection alerts, event-driven classification pipelines.

---

## Python / pandas Examples

### Full Self-Contained Batch ETL Example

```python
import pandas as pd
import json
import requests

# ── EXTRACT ────────────────────────────────────────────────────────────────────
def extract(path: str) -> pd.DataFrame:
    """Read entity names from a CSV file."""
    df = pd.read_csv(path)
    print(f"Extracted {len(df)} rows from {path}")
    return df


# ── TRANSFORM ──────────────────────────────────────────────────────────────────
def call_classification_api(entity_name: str, api_url: str) -> dict:
    """
    Call an LLM API to classify an entity by industry and market position.
    Returns a dict with classification fields.
    """
    prompt = f"Classify '{entity_name}' by industry and market positioning. JSON only."
    response = requests.post(
        api_url,
        json={"inputs": prompt, "parameters": {"max_new_tokens": 256}},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def transform(df: pd.DataFrame, api_url: str) -> pd.DataFrame:
    """Enrich each row with classification data from the API."""
    results = []
    for entity_name in df["Entity Name"]:
        try:
            api_response = call_classification_api(entity_name, api_url)
            classification = json.loads(api_response.get("generated_text", "{}"))
            results.append({
                "Entity Name": entity_name,
                "Industry": classification.get("Industry", ""),
                "Sub Category": classification.get("Sub Category", ""),
                "Market Positioning": classification.get("Market Positioning", ""),
                "Reason": classification.get("Brief reason for classification", ""),
            })
        except Exception as exc:
            print(f"  Skipping '{entity_name}': {exc}")
    return pd.DataFrame(results)


# ── LOAD ───────────────────────────────────────────────────────────────────────
def load(df: pd.DataFrame, output_path: str) -> None:
    """Append transformed data to a CSV file."""
    df.to_csv(output_path, mode="a", header=not pd.io.common.file_exists(output_path), index=False)
    print(f"Loaded {len(df)} rows into {output_path}")


# ── PIPELINE ───────────────────────────────────────────────────────────────────
def run_pipeline(input_csv: str, output_csv: str, api_url: str) -> None:
    raw = extract(input_csv)
    enriched = transform(raw, api_url)
    load(enriched, output_csv)

# run_pipeline("part3.csv", "updated_entity_data.csv", "http://your-api/generate")
```

---

## ETL vs ELT

| Aspect | ETL | ELT |
|--------|-----|-----|
| **Order** | Transform *before* loading | Load first, transform *inside* the destination |
| **Where transformation happens** | Dedicated processing layer (Python, Spark) | Inside the data warehouse (SQL, dbt) |
| **Best for** | Structured pipelines, legacy warehouses, sensitive data that must be masked before storage | Cloud data warehouses (BigQuery, Snowflake) with abundant compute |
| **Latency** | Slightly higher (transform step adds time) | Lower raw ingestion latency |
| **Flexibility** | Transform logic can use any language/library | Transform logic is usually SQL |
| **Example tools** | pandas, Apache Spark, Airflow | dbt, Fivetran, BigQuery scheduled queries |

### ETL (Classic)
```
[Source CSV] → [Python/pandas transform] → [Output CSV / Database]
```

### ELT (Modern Cloud Approach)
```
[Source CSV] → [Raw table in warehouse] → [SQL/dbt transform] → [Analytics table]
```

**Rule of thumb**: use ETL when you need to apply complex transformations (ML inference, API enrichment, privacy masking) *before* the data reaches storage. Use ELT when your warehouse has the compute to run SQL-based transformations at scale.

---

## ETL in This Repository

This repository implements **two complementary ETL workflows** that together form a data classification pipeline.

### Workflow 1 — Entity Classification (`jfs.py`)

```
┌──────────────────────┐     ┌────────────────────────────────┐     ┌──────────────────────────┐
│  EXTRACT             │     │  TRANSFORM                     │     │  LOAD                    │
│                      │     │                                │     │                          │
│  Read entity names   │────▶│  For each entity name:         │────▶│  Append classified row   │
│  from part3.csv      │     │  • Build LLM prompt            │     │  to updated_entity_      │
│  using pandas        │     │  • POST to /tgi/generate API   │     │  data.csv                │
│                      │     │  • Parse JSON response         │     │  (Industry, Sub Category,│
│  pd.read_csv(...)    │     │  • Retry on failure            │     │   Market Positioning,    │
└──────────────────────┘     └────────────────────────────────┘     │   Reason)                │
                                                                     └──────────────────────────┘
```

| ETL Stage | Code location | What happens |
|-----------|--------------|--------------|
| **Extract** | `generate()` → `pd.read_csv("part3.csv")` | Entity names are read from a CSV file into a pandas DataFrame |
| **Transform** | `generate()` loop → `get_response_until_success()` | Each entity name is sent to an LLM API; the JSON response is parsed to extract Industry, Sub Category, Market Positioning, and a brief reason |
| **Load** | `saving(response)` → `csv.writer` | The enriched record is appended row-by-row to `updated_entity_data.csv` |

This is a **batch pipeline**: all entities from `part3.csv` are processed in one run, and results accumulate in the output CSV.

---

### Workflow 2 — Image Perk Classification (`imgclsfy.py`)

```
┌──────────────────────┐     ┌────────────────────────────────┐     ┌──────────────────────────┐
│  EXTRACT             │     │  TRANSFORM                     │     │  LOAD                    │
│                      │     │                                │     │                          │
│  User uploads an     │────▶│  YOLOv8 fine-tuned model       │────▶│  Classification label    │
│  image via Gradio    │     │  runs inference:               │     │  ("perk", "no_perk", or  │
│  web interface       │     │  • Computes class probabilities│     │  "ambiguous") is         │
│                      │     │  • Applies confidence threshold│     │  returned to the user    │
│  gr.Image input      │     │  • Maps to human-readable label│     │  via Gradio UI           │
└──────────────────────┘     └────────────────────────────────┘     └──────────────────────────┘
```

| ETL Stage | Code location | What happens |
|-----------|--------------|--------------|
| **Extract** | `classify_image(inp)` via `gr.Interface` | An image is captured from the Gradio web UI and passed to the classification function |
| **Transform** | `model(inp)` → probability thresholding | The fine-tuned YOLOv8 model (`best.pt`) runs inference; probabilities are compared to thresholds (0.75, 0.50) to decide the label |
| **Load** | `return results[0].names[...]` | The human-readable label is written back to the Gradio output text box for the user to see |

This is a **streaming/event-driven pipeline**: each image upload triggers an independent, real-time inference run rather than a scheduled batch job.

---

### Summary

| | `jfs.py` | `imgclsfy.py` |
|--|----------|--------------|
| **Pattern** | Batch | Streaming / event-driven |
| **Extract source** | CSV file (`part3.csv`) | User-uploaded image |
| **Transform** | LLM API call + JSON parsing | YOLOv8 model inference |
| **Load destination** | CSV file (`updated_entity_data.csv`) | Gradio UI text output |
| **Trigger** | Manual script execution | HTTP request per image |
