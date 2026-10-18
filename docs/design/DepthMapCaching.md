# Depth-map and CUDA photo caches

`DepthMapSnapshotCache` reuses decoded previous-pass maps during geometric estimation.
Readers hold immutable snapshots; consumers needing writable planes receive clones.
Concurrent misses coalesce. Eviction removes cache ownership without invalidating readers.

| DensifyPointCloud option | Default | Budget |
| --- | --- | --- |
| `--geometric-cache-memory` | 0, disabled | Decoded previous-pass snapshots, MiB |
| `--patch-match-cuda-image-cache-memory` | 0, disabled | Same configured MiB cap applied separately to host and device allocations |

The CUDA photo cache keys an immutable full-resolution image and deterministic pyramid scale.
Retained parent ownership prevents allocation-address reuse. Stream-ready events and leases
protect uploads and consumers. Depth priors never enter this cache. The current worker pool
uses one selected CUDA device.

Worker admission accounts for cache reservations, neighbor planes and decode scratch. Cache
capacities are allocation budgets, not process RSS limits. Both caches are native OpenMVS
implementations and can be configured independently.

Validation: `DepthMapSnapshotCacheTest` covers snapshot ownership, eviction, load failures and
concurrent misses.
