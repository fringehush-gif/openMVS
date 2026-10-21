# Depth-map and CUDA photo caches

## Purpose and implementation

`DepthMapSnapshotCache` (`libs/MVS/DepthMapSnapshotCache.h`) reuses decoded previous-pass maps
during geometric estimation. Readers hold immutable snapshots; writable reference planes receive
clones in `DepthMapsData::InitViews` (`SceneDensify.cpp`). Cache lookup uses `PairIdx(imageID,flags)`;
a loaded superset can serve a request for fewer planes. A shared future coalesces concurrent misses,
including recoverable file errors, while decoding runs outside the cache mutex. Failed reads are
not retained. Least-recently-used entries without reader leases can be evicted. When leases fill
the budget, a decoded map is returned without cache admission; readers never wait for capacity.

The CUDA photo cache (`PatchMatchCUDA.cpp`) keys an immutable full-resolution image and its
deterministic pyramid scale. Retained parent ownership prevents allocation-address reuse.
Stream-ready events and leases protect uploads and consumers. Depth priors never enter this cache.
The worker pool uses one selected CUDA device. Worker admission accounts for cache reservations,
neighbor planes and decode scratch.

## Parameters and defaults

| DensifyPointCloud option | Default | Budget |
| --- | --- | --- |
| `--geometric-cache-memory` | 0, disabled | Decoded previous-pass snapshots, MiB |
| `--patch-match-cuda-image-cache-memory` | 0, disabled | Same configured MiB cap applied separately to host and device allocations |

## Invariants

- One image ID identifies one immutable file generation until `Reset`; reset requires joined
  workers, no pending loads and no remaining reader leases.
- Neighbor planes may alias an immutable snapshot; reference planes must be cloned before writes.
- Snapshot requests include depth and exclude per-pixel view maps; optional corrupt planes do not
  invalidate a depth-only request.
- Cache capacities limit retained allocations, not process RSS or temporary in-flight decoding.
- Configure CUDA photo retention before creating the worker pool; stream completion precedes reuse.

## Validation of the shipped defaults

Both caches default to disabled. Native regressions exercise snapshot ownership and worker state;
the opt-in CUDA photo cache is also exercised by full densification replays.

| Native regression | Checked behavior |
| --- | --- |
| `DepthMapSnapshotCacheTest` | Metadata/plane round trips, immutable borrow versus writable clone, reset generations, leased-capacity eviction, missing/truncated files, eight concurrent readers |
| `PatchMatchTopKLifecycleTest` | Per-image state and configured-versus-active TopK transitions |

## Rejected alternatives

- Evicting leased snapshots or waiting for capacity can invalidate readers or deadlock overlapping neighborhoods.
- Reusing a mutable reference map would expose estimator writes to other workers.

## Open items

- Cache budgets remain opt-in; workload-dependent memory and timing decide whether to enable them.
