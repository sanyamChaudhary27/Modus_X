# Segment-scale archive retention

Source-removal traces showed that the model wrote a measurable perturbation at
the earlier occurrence, but almost all of that archive-specific trace vanished
before a distant target. The canonical archive multiplied the entire matrix by
a learned retention value at every byte. A retention value that looks gentle
once becomes aggressive when compounded hundreds of times.

The candidate changes only the archive clock:

```text
canonical: rho = sigmoid(z)
candidate: rho = sigmoid(z + log(512))
```

Current memory, delta writes, reads, feedback, routing, parameter count, and
state size stay unchanged. Three short screens preserved language quality and
improved source retention. Two longer endpoints improved dense validation.
Finally, erasing archive state on natural conflicting repeats made prediction
worse rather than better, so the retained archive was useful in the measured
stale-memory stress case.
