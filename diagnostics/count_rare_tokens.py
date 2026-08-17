import numpy as np
from collections import Counter

bpe_dir = "data/bpe_wikitext103_v8000" 

train = np.load(f"{bpe_dir}/train.npy")
test  = np.load(f"{bpe_dir}/test.npy")

train_counts = Counter(train.tolist())
test_types = set(test.tolist())

# Of the token *types* that appear in test, how rare are they in train?
unseen   = sum(1 for t in test_types if train_counts[t] == 0)
rare     = sum(1 for t in test_types if 0 < train_counts[t] < 5)
ok       = sum(1 for t in test_types if train_counts[t] >= 5)
n        = len(test_types)

print(f"distinct test token types: {n}")
print(f"  unseen in train (count 0):     {unseen:5d}  ({unseen/n:.1%})")
print(f"  rare in train (1-4):           {rare:5d}  ({rare/n:.1%})")
print(f"  well-supported (>=5):          {ok:5d}  ({ok/n:.1%})")

# Weighted by actual test occurrences (what eval perplexity really sees):
test_counts = Counter(test.tolist())
tok_unseen = sum(c for t, c in test_counts.items() if train_counts[t] == 0)
tok_rare   = sum(c for t, c in test_counts.items() if 0 < train_counts[t] < 5)
total_tok  = sum(test_counts.values())
print(f"\nby test token *occurrences* (what perplexity averages over):")
print(f"  from unseen types:  {tok_unseen/total_tok:.1%}")
print(f"  from rare types:    {tok_rare/total_tok:.1%}")