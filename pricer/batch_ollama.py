
import os
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import json
import pickle
from tqdm.notebook import tqdm

load_dotenv(override=True)

# Connect to local Ollama
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
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
    BATCH_SIZE = 1_000

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
        """Send one request to the local Ollama model."""

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item.full},
            ],
        )

        return response.choices[0].message.content

    def process_local(self):
        """Process this batch locally and save JSONL results."""

        output_file = self.output / self.filename

        with output_file.open("w", encoding="utf-8") as f:
            for item in tqdm(
                self.items[self.start:self.end],
                desc=f"Processing {self.start}_{self.end}"
            ):
                summary = self.process_item(item)

                result = {
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

                f.write(json.dumps(result, ensure_ascii=False))
                f.write("\n")

        self.apply_output()

    def apply_output(self):
        output_file = self.output / self.filename

        items_by_id = {item.id: item for item in self.items}

        with output_file.open("r", encoding="utf-8") as f:
            for line in f:
                json_line = json.loads(line)

                item_id = int(json_line["custom_id"])

                summary = (
                    json_line["response"]["body"]
                    ["choices"][0]["message"]["content"]
                )

                items_by_id[item_id].summary = summary

        self.done = True

    @classmethod
    def create(cls, items, lite):
        for start in range(0, len(items), cls.BATCH_SIZE):
            end = min(start + cls.BATCH_SIZE, len(items))

            batch = Batch(items, start, end, lite)
            cls.batches.append(batch)

        print(f"Created {len(cls.batches)} batches")

    @classmethod
    def run(cls):
        """Process all batches using local Ollama."""

        for batch in tqdm(cls.batches):
            if not batch.done:
                batch.make_file()
                batch.process_local()

        print(f"Finished {sum(b.done for b in cls.batches)} "
              f"of {len(cls.batches)} batches")

    @classmethod
    def fetch(cls):
        """Load any existing output files not yet applied."""

        for batch in tqdm(cls.batches):
            if not batch.done:
                output_file = batch.output / batch.filename

                if output_file.exists():
                    batch.apply_output()

        finished = [batch for batch in cls.batches if batch.done]

        print(f"Finished {len(finished)} of {len(cls.batches)} batches")

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