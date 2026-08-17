import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
from fabricpc.utils.data import CharDataLoader, BpeDataLoader


def coverage(train_ids, test_ids, n):
    train_ng = set(zip(*[train_ids[i:] for i in range(n)]))
    test_ng = list(zip(*[test_ids[i:] for i in range(n)]))
    seen = sum(1 for g in test_ng if g in train_ng)
    return seen / len(test_ng), len(train_ng)


for name, Loader in [("char", CharDataLoader), ("bpe", BpeDataLoader)]:
    tr = np.asarray(Loader("train", seq_len=128, batch_size=1).data)
    te = np.asarray(Loader("test", seq_len=128, batch_size=1).data)
    print(f"\n{name}: vocab~{int(tr.max()) + 1}, train_tokens={len(tr)}, test_tokens={len(te)}")
    for n in (1, 2, 3):
        cov, n_distinct = coverage(tr, te, n)
        print(f"  {n}-gram: {cov:6.1%} of test {n}-grams seen in train  ({n_distinct} distinct in train)")
