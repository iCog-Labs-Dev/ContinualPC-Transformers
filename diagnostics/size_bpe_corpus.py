import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
from fabricpc.utils.data import BpeDataLoader

DATASET = "wikitext103"   # routes to the HF WikiText-103 branch in load_split
VOCAB = 8000           # size the vocab to the data

# First construction trains BPE on WikiText-103 and encodes all splits.
# This is the slow one-time step; subsequent runs load from cache.
train = np.asarray(
    BpeDataLoader("train", seq_len=128, batch_size=1, dataset=DATASET, vocab_size=VOCAB).data
)
test = np.asarray(
    BpeDataLoader("test", seq_len=128, batch_size=1, dataset=DATASET, vocab_size=VOCAB).data
)
print(f"{DATASET} vocab={VOCAB}: train={len(train):,} tokens, test={len(test):,} tokens")


def coverage(train_ids, test_ids, n):
    train_ng = set(zip(*[train_ids[i:] for i in range(n)]))
    test_ng = list(zip(*[test_ids[i:] for i in range(n)]))
    return sum(1 for g in test_ng if g in train_ng) / len(test_ng)


print(f"\n{'train budget':>13} | {'2-gram':>7} | {'3-gram':>7}")
for B in (1_000_000, 2_000_000, 5_000_000, 10_000_000, len(train)):
    B = min(B, len(train))
    sub = train[:B]
    c2, c3 = coverage(sub, test, 2), coverage(sub, test, 3)
    flag = "  <- clears 95% bigram" if c2 >= 0.95 else ""
    print(f"{B:>13,} | {c2:>6.1%} | {c3:>6.1%}{flag}")