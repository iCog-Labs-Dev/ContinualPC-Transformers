from datasets import load_dataset
ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
print(len(ds), "examples")
print(repr(ds[0]["text"][:300]))