# Previous-pass depth-map cache

`DepthMapSnapshotCache` reuses decoded previous-pass maps during geometric estimation.
Readers hold immutable snapshots; consumers needing writable planes receive clones.
Concurrent misses coalesce. Eviction removes cache ownership without invalidating readers.

`DensifyPointCloud --geometric-cache-memory` sets the capacity in MiB; its default
is zero (disabled). Worker admission accounts for the reservation, neighbor planes
and decoder scratch. Capacity is an allocation budget, not a process RSS limit.

Validation: `DepthMapSnapshotCacheTest` covers ownership, eviction, failures and
concurrent misses. Current-baseline native and performance comparisons remain open.
