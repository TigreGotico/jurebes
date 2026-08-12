"""predict vs predict_batch.

Confirms that per-item predict and predict_batch return the same intents,
and reports the wall-clock latency of each approach.
"""

# %% setup
import time

from jurebes import IntentClassifier

clf = IntentClassifier()
clf.add_intent("greet", ["hello", "hi", "hey there", "good day"])
clf.add_intent("bye", ["goodbye", "bye", "see ya", "later"])
clf.fit()

utts = ["hello world", "see ya", "hi friend", "later then", "hey", "goodbye"] * 20

# %% per-item
t = time.perf_counter()
single = [clf.predict(u).intent for u in utts]
dt_single = time.perf_counter() - t

# %% batched
t = time.perf_counter()
batched = [r.intent for r in clf.predict_batch(utts)]
dt_batch = time.perf_counter() - t

print(f"single  : {dt_single*1000:.2f} ms")
print(f"batched : {dt_batch*1000:.2f} ms")
assert single == batched
print("results identical")
