# Mesh Texturing

## 1. Purpose and scope

`TextureMesh` maps every face of a mesh to the image that should texture it, levels the colours
across the seams between faces taken from different images, and packs the image regions into one or
more texture atlases. Library code: `libs/MVS/SceneTexture.cpp` (`MeshTexture`, `Scene::TextureMesh`,
`Scene::ComputeVertexColors`); CLI driver: `apps/TextureMesh/TextureMesh.cpp`. Texture coordinates are
stored per face (three per face, `Mesh::faceTexcoords`) with one texture index per face
(`Mesh::faceTexindices`, empty for a single atlas).

## 2. Algorithm as implemented

### 2.1 Face-view data (`MeshTexture::ListCameraFaces`)

For every view (in parallel, one view per thread):

1. The image is loaded at the working resolution (`--resolution-level`, `--min-resolution`).
2. The faces inside the view frustum (mesh octree) are rasterized at the **full image resolution**
   into a face-id map and a depth map (closest face wins). Faces whose pixels fall on the image's
   invalid area (`ComputeValidityMask`, lens-distortion border / ignore-mask label) are marked
   invalid for the view. Every pixel a face covers is also counted before the depth test
   (`RasterMesh::projArea`), occluded or not.
3. The **quality** of a face in the view is the number of its visible pixels: the texture resolution
   the view offers it, which already falls with distance and obliqueness. The pixels are counted
   (and their colours summed) per thread; only the faces the view sees are appended to the shared
   face-view lists, inside the one critical section.
4. A view that sees under 90% of the face's projection, with more than 2 pixels missing, keeps a
   tenth of that quality: it would copy the occluder into the face's texture, so it is chosen only
   where nothing sees the face whole.
5. With `--outlier-threshold` > 0 the mean colour of the face in each view is kept and
   `FaceOutlierDetection` (Gaussian clamping in YCbCr) removes views whose colour disagrees with the
   others (e.g. a passer-by in one image). With OpenMP enabled, independent face lists are filtered in
   parallel after the view workers finish; other builds filter serially. Observation order
   within each face is preserved.

With `--texture-cuda true`, eligible views use `TextureScoring::PrepareTriangles` and
`CUDA::RunTextureCUDA` (`TextureScoring.h`, `TextureCUDA.cu`) for the same native-resolution
visibility counts and ordered BGR sums. CPU projection supplies triangle coordinates and traversal
ranks; a depth/rank key resolves nearest depth and first-rank ties. A serialized workspace retains
allocations on one device and restores the caller's device after each operation. CPU scoring remains
the default. Repeated explicit views, partial validity masks, non-contiguous images and unsupported
host arithmetic or projection ranges select CPU before device work starts. An attempted CUDA
operation that fails aborts the stage after workers join; it does not retry on CPU.

### 2.2 View selection (`MeshTexture::FaceViewSelection`)

A labeling over the face adjacency graph (label 0 = no view, label i+1 = view i), solved by loopy
belief propagation (`Math/LBP.h`):

- **Data cost** of view v for face f: `(1 - q_fv / max_v q_fv) * MaxEnergy * area_f/avgArea`, i.e.
  each face is scored relative to its own best view (cost 0); views below 0.2 of the best are not
  candidates. A face no view sees can only take label 0, at cost `MaxEnergy*(area_f/avgArea+0.5)`.
- **Smoothness**: Potts (0 for equal labels, `LBPMaxEnergy` otherwise), weighted per edge by
  `0.5 + edgeLength/avgEdgeLength`.
- `MaxEnergy = --cost-smoothness-ratio` (default 0.1) sets the data/smoothness balance.
- The Potts model lets each LBP message be computed in O(labels) (`LBPInference::SetSmoothCostPotts`):
  the sender's energy for a label is one of two values (same label: smoothness 0, otherwise the Potts
  cost), so the message of receiver label j is min(E0[j], min_k E1[k]). Only E0 is stored; E1 feeds
  a running minimum. Energies follow the generic loop's order. Each worker normalizes its completed
  message slice before the implicit OpenMP barrier and buffer swap; no separate normalization pass
  scans the flat buffers.
- Automatic LBP optimization accumulates unary energy while selecting each node's label,
  then reduces pairwise edge energy separately. Both sums retain float arithmetic.
  Fixed-iteration optimization only selects labels; standalone energy evaluation always
  reads the current labels and data costs rather than a cached total.
- The messages of all directed edges live in two flat buffers allocated when the optimization
  starts; each face's view list is released as soon as its data costs are in the inference, so
  the labeling never holds both (Ignatius peak 3.87 -> 3.44 GB, Truck 2.95 -> 2.73 GB, 4-5% faster).
- Remaining-face grouping uses `TextureFaceIndex.h` to select and erase ordered ranks without
  repeated linear searches or shifting; face order and RNG calls stay unchanged.
- `--virtual-face-images` > 0 runs the same labeling on virtual faces (coplanar groups of faces
  sharing at least that many views) instead of individual faces.

Connected faces with the same label form a texture patch; faces without a view go to the last
patch, textured with `--empty-color`.

### 2.3 Patch geometry and seam leveling (`GenerateTexture`)

- Each patch's faces are projected in its view; the patch rectangle is their bounding box plus a
  2 px border of real image content.
- **Global seam leveling** (`GlobalSeamLeveling`, `--global-seam-leveling`): one additive YCbCr
  correction per (vertex, patch), solved in the least-squares sense so the two sides of every seam
  vertex agree (colours sampled along the seam edges), with a Tikhonov term (lambda 0.1) keeping the
  corrections smooth inside a patch; each channel owns its conjugate-gradient solver and mutable
  iteration state; the correction is interpolated
  over each patch's triangles.
- **Local seam leveling** (`LocalSeamLeveling`, `--local-seam-leveling`): Poisson blending of a strip
  along each patch border so the border continues smoothly into the patch. `TextureSeamIndex.h`
  builds patch-to-seam incidence once while preserving seam order. The optional
  `--poisson-interior-ldlt` path factors only interior unknowns with prescribed border colors;
  the existing solver remains the default, and floating-point colors can differ.
- A patch whose rectangle lies inside another patch's rectangle of the same view joins it.
  Removal updates the shifted owner's slot and UV offsets; patch zero and face order remain valid.

### 2.4 Atlas

- Patches are grouped spatially (`SplitPatchesSpatially`: sorted along the longest axis of their
  centroids, cut where a trial rectangle packing stops fitting one `--max-texture-size` page).
- Each patch is packed by its **footprint**, not its rectangle: its faces rasterized in the patch
  rectangle (plus the pixels holding their corners) dilated by the 2 px border, packed by
  `halfmesh::PackFootprints` (`PackTexturePagesMasks`): bottom-left first fit on a 4x4-pixel block
  grid, each block row of a footprint described by the block columns it spans; every page row keeps
  its free intervals, and the patches, largest footprint first, each take the lowest row and then the
  leftmost column where all their row spans lie in free intervals, as they are or turned 90° counter-
  clockwise, whichever lands lower (a rotated patch is copied with `cv::rotate` and its pixel (x,y)
  maps to (y, w-1-x)). The page width is tried at half, one and two times the rounded square root of
  the footprint area; the width leaving the fewest patches over, then the smallest page, wins. The
  page is cropped to the extent its patches use, each dimension rounded up to a power of two (or
  `--texture-size-multiple`), so pages need not be square; patches over the `--max-texture-size`
  height open another page.
- Only the footprint pixels are copied into the page (rectangles overlap, footprints do not); the
  other texels are filled by push-pull over a box pyramid of the footprints (`PushPullFill`), so a
  mipmapped lookup near a patch border averages its own surroundings instead of the empty colour
  (which only the unseen faces' patch keeps).
- `--sharpness-weight` > 0 applies an unsharp mask (sigma 1.5) to the pages; disabled by default.

### 2.5 Vertex colours (`ComputeVertexColors`, `--vertex-colors`)

Same view selection; each vertex takes the weighted average of its colour in the views texturing the
faces around it; images are released as soon as their faces are consumed.

## 3. Parameters and defaults

| option | default | meaning |
|---|---|---|
| `--resolution-level` / `--min-resolution` | 0 / 640 | working image resolution |
| `--outlier-threshold` | 0.06 | colour-outlier view removal (0 disables) |
| `--cost-smoothness-ratio` | 0.1 | data vs. smoothness balance of the labeling |
| `--texture-cuda` | false | opt in to native-grid CUDA visibility/scoring |
| `--cuda-device` | -1 | automatic CUDA selection; -2 CPU, nonnegative exact device; used with `--texture-cuda` |
| `--virtual-face-images` | 0 | label virtual faces of coplanar triangles (0 disables) |
| `--global-seam-leveling` / `--local-seam-leveling` | 1 / 1 | seam colour leveling |
| `--poisson-interior-ldlt` | false | interior-only factorization with prescribed border colors; floating-point results can change |
| `--texture-size-multiple` | 0 | page size multiple (0 = power of two) |
| `--max-texture-size` | 8192 | page size cap; more pages when exceeded |
| `--export-texture-png-compression` | 6 | PNG compression level 0-9 for lossless PLY textures |
| `--export-texture-png-buffered` | false | encode PNG before writing; requires additional RAM |
| `--empty-color` | 0x00FF7F27 | colour of faces no view sees |
| `--sharpness-weight` | 0 | unsharp-mask weight |
| `--remove-unseen-faces` | 0 | drop faces seen by fewer than N images before texturing |

Explicit PNG options use `TImage::Save(fileName, compression, buffered)`; the existing one-argument
image save keeps its format defaults. Native unit tests cover all eight combinations of compression
levels 0, 2, 6 and 9 with buffering on/off and exact decoded BGR pixels.

Fixed: full-resolution raster, partial-visibility threshold 90% / 2 px and factor 0.1, per-face
candidate cut-off 0.2, LBP min edge weight 0.5, global leveling lambda 0.1.

## 4. Invariants

- Every face has three texture coordinates; library code never uses a per-vertex layout.
- A face's data cost exists only for views within 0.2 of its best; the best view of every seen face
  has cost 0, so a face is never forced into label 0 while some view sees it.
- The visible-pixel count of a face never exceeds its projected-pixel count (asserted).

## 5. Validation of the shipped defaults

Hold-out bench `bench/eval_texture.py`: the refined mesh of each scene textured with every 4th view
left out, the textured mesh ray-cast into the held-out cameras and compared with the photos (T&T:
only the surface inside the evaluation crop; faces a run leaves unseen are excluded for every arm).

| current-default metric (hold-out) | HJ8 | F11 | HJ25 | Ignatius |
|---|---|---|---|---|
| PSNR / SSIM | 26.55 / .9222 | 28.80 / .9451 | 24.52 / .8989 | 17.47 / .7527 |
| footprint atlas | 32 MT | 32 MT | 32 MT | 32 MT |
| triangle fill | 0.34 | 0.36 | 0.57 | 0.26 |

Why the current defaults win: the visible-pixel count is the texture resolution a view actually
delivers, so it ranks views by distance and obliqueness without a separate gradient or angle term.
Scoring each face against its own best view, instead of one global normalization, gives that view
cost 0 on every seen face, so a face that every view sees poorly still gets its best view rather
than whatever the smoothness term prefers. The partial-visibility penalty stops a view
that sees most of a face from copying the occluder into the rest of it. The raster runs at the
resolution the texture is sampled from; the 2x raster only raised the peak RAM (below).

TextureMesh peak RAM (HJ25 hold-out): 2.5-3.0 GB.


Footprints halve the atlas because a patch's bounding rectangle is mostly empty for any patch that is
not a compact blob, and footprints may interlock where rectangles may not. Same PSNR at full
resolution, +0.01..+0.03 dB at mip 3 (push-pull fill instead of the empty colour around the
patches); the atlas stage takes the same time (HJ25 9.0 s vs 9.3 s, Ignatius 8.7 s vs 7.8 s).

## 6. Rejected alternatives

- Gradient-magnitude data term (blurred Sobel x cos^2, global 95th-percentile normalization):
  lower hold-out quality than visible-pixel scoring (§5).
- Crease-aware seam cost (edge weight x (1+n_a.n_b)/2): HJ8 -0.08 dB.
- Multiplicative, ratio-gated global seam leveling in log-RGB: HJ8 +0.05, F11 -0.09, HJ25 -0.2 dB.
- Merging same-view patches whose rectangles overlap (not only contained ones): -4% patch pixels but
  the page size never changed and the atlas stage was 20% slower.
- Unsharp mask by default: lowered PSNR and SSIM on every scene.
- Horizon ("Tetris") packing of the footprints: 44-58% of the page used, no smaller page than rectangles.
- 1 px footprint border: footprint area -2..-5% (the block grid dominates), no page smaller, mip 3 -0.01..-0.04 dB.
- 2x2-pixel blocks: footprint area -3..-14%, no page smaller on any scene, atlas stage +0.5..+1.7 s.

## 7. Open items

- Atlas fill: what the power-of-two rounding of the page dimensions leaves empty (a page is 50-90%
  used); `--texture-size-multiple` rounds finer.
- The images stay loaded from view selection to atlas generation (every image textures some patch on
  the scenes measured); the peak is now the face-view listing (images, per-thread face/depth rasters
  and the growing face-view lists).
- Faces no view sees get the empty colour; a colour diffused from the neighbouring faces would be
  closer to the surface's appearance.
