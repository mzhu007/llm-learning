
import os
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import json
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.notebook import tqdm

load_dotenv(override=True)

# Connect to local Ollama
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    timeout=300.0,
    max_retries=3
)

MODEL = "gpt-oss:20b"

BATCHES_FOLDER = "batches"
OUTPUT_FOLDER = "output"
state = Path("batches.pkl")

SYSTEM_PROMPT = """Create a concise description of a product. Respond only in this format. Do not include part numbers.
Title: Rewritten short precise title
Category: eg Electronics
Brand: Brand name
Description: 1 sentence description
Details: 1 sentence on features"""


class Batch:
    # BATCH_SIZE = 1_000
    BATCH_SIZE = 20

    # Number of concurrent Ollama requests
    MAX_WORKERS = 10

    batches = []

    def __init__(self, items, start, end, lite):
        self.items = items
        self.start = start
        self.end = end
        self.filename = f"{start}_{end}.jsonl"
        self.file_id = None
        self.batch_id = None
        self.output_file_id = None
        self.done = False

        folder = Path("lite") if lite else Path("full")
        self.batches = folder / BATCHES_FOLDER
        self.output = folder / OUTPUT_FOLDER

        self.batches.mkdir(parents=True, exist_ok=True)
        self.output.mkdir(parents=True, exist_ok=True)

    def make_jsonl(self, item):
        body = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item.full},
            ],
        }

        line = {
            "custom_id": str(item.id),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }

        return json.dumps(line, ensure_ascii=False)

    def make_file(self):
        batch_file = self.batches / self.filename

        with batch_file.open("w", encoding="utf-8") as f:
            for item in self.items[self.start:self.end]:
                f.write(self.make_jsonl(item))
                f.write("\n")

    def process_item(self, item):
        """Process one item using the local Ollama model."""

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item.full},
            ],
            temperature=0
        )

        summary = response.choices[0].message.content

        if not summary:
            raise ValueError(f"Empty response for item {item.id}")

        return {
            "custom_id": str(item.id),
            "response": {
                "body": {
                    "choices": [
                        {
                            "message": {
                                "content": summary
                            }
                        }
                    ]
                }
            }
        }

    def read_completed_ids(self, output_file):
        """Find products already processed and saved."""

        completed = set()

        if not output_file.exists():
            return completed

        with output_file.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    result = json.loads(line)
                    completed.add(int(result["custom_id"]))
                except (json.JSONDecodeError, KeyError, ValueError):
                    # Ignore incomplete or invalid lines
                    continue

        return completed

    def process_local(self, max_workers=None):
        """Process pending items concurrently and save each result."""

        if max_workers is None:
            max_workers = self.MAX_WORKERS

        output_file = self.output / self.filename

        batch_items = self.items[self.start:self.end]

        # Skip products already processed in previous runs
        completed = self.read_completed_ids(output_file)

        pending = [
            item for item in batch_items
            if item.id not in completed
        ]

        print(
            f"Batch {self.start}_{self.end}: "
            f"{len(completed)} completed, "
            f"{len(pending)} remaining"
        )

        if not pending:
            self.apply_output()
            return

        errors = []

        # Main thread handles file writing.
        # Worker threads only make API requests.
        with output_file.open("a", encoding="utf-8") as f:

            with ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:

                futures = {
                    executor.submit(self.process_item, item): item
                    for item in pending
                }

                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Batch {self.start}_{self.end}"
                ):
                    item = futures[future]

                    try:
                        result = future.result()

                        # Save immediately after each successful request
                        f.write(json.dumps(result, ensure_ascii=False))
                        f.write("\n")
                        f.flush()

                    except Exception as e:
                        errors.append((item.id, str(e)))

                        print(
                            f"\nError processing item {item.id}: {e}"
                        )

        # Apply all successfully processed results
        self.apply_output()

        if errors:
            print(
                f"Batch {self.start}_{self.end}: "
                f"{len(errors)} failed. "
                "Run Batch.run() again to retry."
            )

    def apply_output(self):
        """Apply saved results to original items."""

        output_file = self.output / self.filename

        if not output_file.exists():
            return

        items_by_id = {
            item.id: item
            for item in self.items[self.start:self.end]
        }

        completed = set()

        with output_file.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    result = json.loads(line)

                    item_id = int(result["custom_id"])

                    if item_id not in items_by_id:
                        continue

                    summary = (
                        result["response"]["body"]
                        ["choices"][0]["message"]["content"]
                    )

                    if summary:
                        items_by_id[item_id].summary = summary
                        completed.add(item_id)

                except (json.JSONDecodeError, KeyError, ValueError):
                    continue

        expected = set(items_by_id.keys())

        self.done = expected.issubset(completed)

    @classmethod
    def create(cls, items, lite):
        cls.batches = []

        for start in range(0, len(items), cls.BATCH_SIZE):
            end = min(start + cls.BATCH_SIZE, len(items))

            batch = Batch(items, start, end, lite)
            cls.batches.append(batch)

        print(f"Created {len(cls.batches)} batches")

    @classmethod
    def run(cls):
        """Process all batches using concurrent Ollama requests."""

        for batch in tqdm(cls.batches):
            if not batch.done:
                batch.make_file()
                batch.process_local()

        finished = sum(batch.done for batch in cls.batches)

        print(
            f"Finished {finished} of {len(cls.batches)} batches"
        )

    @classmethod
    def fetch(cls):
        """Apply previously saved output files."""

        for batch in tqdm(cls.batches):
            if not batch.done:
                batch.apply_output()

        finished = sum(batch.done for batch in cls.batches)

        print(
            f"Finished {finished} of {len(cls.batches)} batches"
        )

    @classmethod
    def save(cls):
        if not cls.batches:
            print("No batches to save")
            return

        items = cls.batches[0].items

        try:
            for batch in cls.batches:
                batch.items = None

            with state.open("wb") as f:
                pickle.dump(cls.batches, f)

        finally:
            for batch in cls.batches:
                batch.items = items

        print(f"Saved {len(cls.batches)} batches")

    @classmethod
    def load(cls, items):
        with state.open("rb") as f:
            cls.batches = pickle.load(f)

        for batch in cls.batches:
            batch.items = items

        print(f"Loaded {len(cls.batches)} batches")