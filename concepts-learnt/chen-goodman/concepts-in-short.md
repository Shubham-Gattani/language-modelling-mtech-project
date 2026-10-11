### Interpolation
* **Always Blends**: It combines predictions from higher-order and lower-order models using a weighted average every single time.
* **Constant Input**: Even if an n-gram appears frequently in training, lower-order distributions still contribute to the final probability.
* **Intuition**: Like consulting both a specialist and a generalist doctor for every diagnosis, regardless of how obvious the symptoms are.

---

### Backoff
* **Conditional Switch**: It relies **only** on the higher-order model whenever an n-gram has been seen in the training data.
* **Fallback Only**: It "backs off" to consult the lower-order model **strictly** when the higher-order n-gram was never observed (zero count).
* **Intuition**: Like calling a specialist first, and phoning a generalist **only** if the specialist is unavailable.