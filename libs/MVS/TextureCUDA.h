// Current native-grid texture counts and ordered BGR sums; opt-in only.
#pragma once
#include "Common.h"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace MVS { namespace CUDA {
struct TextureTriangle
{
	float x[3], y[3], z[3], invArea;
	int minX, minY, maxX, maxY;
	uint32_t face, rank;
};
struct TextureCount
{
	uint32_t projected, visible;
	float color[3];
};
struct TextureTiming
{
	double totalMs, uploadMs, rasterMs, scoreMs, downloadMs;
	uint64_t deviceBytes;
};
// requested >= 0 selects that exact device; -1 selects one automatically.
// Selection errors are expected external failures, never silent CPU fallback.
MVS_API bool SelectTextureCUDADevice(int requested, int& selected);
// Native CPU projection supplies integer-centered coordinates and traversal ranks.
// All triangles must pass the native working-image border and winding checks.
// Only an empty/fully-valid mask is eligible; color is optional (null: count only).
MVS_API bool RunTextureCUDA(int width, int height, int imageWidth, int imageHeight,
	const std::vector<TextureTriangle>& triangles, const unsigned char* bgr,
	std::vector<TextureCount>& counts, TextureTiming& timing,
	std::vector<uint64_t>* visibility = nullptr, int device = 0);

namespace detail {
// Shared allocation primitive, parameterized on the actual runtime operations.
// Group capacities become valid only after every member succeeds. This is also
// exercised by native Tests with a deterministic failing allocator, without a
// test switch or error-injection API in the production launcher.
template<class Allocator> struct TextureStorage
{
	Allocator& allocator;
	TextureTriangle* triangles = nullptr;
	TextureCount* counts = nullptr;
	uint64_t* keys = nullptr;
	unsigned char* bgr = nullptr;
	size_t triangleCapacity = 0, pixelCapacity = 0, colorCapacity = 0;
	explicit TextureStorage(Allocator& a) : allocator(a) {}
	template<class T> bool Replace(T*& pointer, size_t count)
	{
		ASSERT(count > 0 && count <= std::numeric_limits<size_t>::max()/sizeof(T));
		if (pointer && !allocator.Free(pointer)) return false;
		pointer = nullptr;
		return allocator.Allocate(reinterpret_cast<void**>(&pointer), count*sizeof(T));
	}
	bool Ensure(size_t trianglesNeeded, size_t pixels, size_t colorBytes)
	{
		if (trianglesNeeded > triangleCapacity) {
			triangleCapacity = 0;
			if (!Replace(triangles, trianglesNeeded) || !Replace(counts, trianglesNeeded)) return false;
			triangleCapacity = trianglesNeeded;
		}
		if (pixels > pixelCapacity) {
			pixelCapacity = 0;
			if (!Replace(keys, pixels)) return false;
			pixelCapacity = pixels;
		}
		if (colorBytes > colorCapacity) {
			colorCapacity = 0;
			if (!Replace(bgr, colorBytes)) return false;
			colorCapacity = colorBytes;
		}
		return true;
	}
	void Release()
	{
		if (triangles) allocator.Free(triangles);
		if (counts) allocator.Free(counts);
		if (keys) allocator.Free(keys);
		if (bgr) allocator.Free(bgr);
		triangles = nullptr; counts = nullptr; keys = nullptr; bgr = nullptr;
		triangleCapacity = pixelCapacity = colorCapacity = 0;
	}
};
template<class Synchronize> struct TextureFailureDrain
{
	Synchronize synchronize;
	bool complete = false;
	explicit TextureFailureDrain(Synchronize sync) : synchronize(sync) {}
	~TextureFailureDrain() { if (!complete) synchronize(); }
};
inline bool TextureDeviceMatches(int bound, int requested)
{
	return bound < 0 || bound == requested;
}
} // namespace detail
} } // namespace MVS::CUDA
