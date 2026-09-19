/*
* TexturePatchMerge.h
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU Affero General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*/

#ifndef _MVS_TEXTUREPATCHMERGE_H_
#define _MVS_TEXTUREPATCHMERGE_H_

// The including translation unit must include Common.h first. This keeps the
// same include contract as the other internal MVS implementation headers.
#include <boost/geometry.hpp>
#include <boost/geometry/index/predicates.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <algorithm>
#include <cstdint>
#include <iterator>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace MVS {
namespace PATCHMERGE {

struct Patch {
	uint32_t label;
	cv::Rect rect;
};

struct Merge {
	uint32_t big;
	uint32_t small;
	bool operator==(const Merge& rhs) const {
		return big == rhs.big && small == rhs.small;
	}
};

struct Stats {
	size_t queries = 0;
	size_t spatialCandidates = 0;
	size_t containmentChecks = 0;
	size_t merges = 0;
};

struct Plan {
	std::vector<Merge> merges;
	std::vector<uint8_t> active;
	Stats stats;
};

namespace detail {
namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;
using Point = bg::model::point<int, 2, bg::cs::cartesian>;
using Box = bg::model::box<Point>;
using Value = std::pair<Box, uint32_t>;
using RTree = bgi::rtree<Value, bgi::quadratic<16>>;

inline Box ToBox(const cv::Rect& rect) {
	ASSERT(rect.width >= 0 && rect.height >= 0);
	return Box(Point(rect.x, rect.y), Point(rect.x+rect.width, rect.y+rect.height));
}

inline bool IsContainedIn(const cv::Rect& small, const cv::Rect& big) {
	ASSERT(small.width >= 0 && small.height >= 0 && big.width >= 0 && big.height >= 0);
	return small.x >= big.x && small.y >= big.y
		&& small.x+small.width <= big.x+big.width
		&& small.y+small.height <= big.y+big.height;
}
} // namespace detail

// Build a deterministic merge plan without mutating the caller's patch array.
// Stable patch order is the ownership rule: each active patch is considered as
// a container in original order and absorbs same-label contained peers in
// original order. The spatial index only narrows candidates; the exact,
// boundary-inclusive rectangle predicate remains authoritative.
inline Plan BuildPlan(const std::vector<Patch>& patches) {
	ASSERT(patches.size() <= std::numeric_limits<uint32_t>::max());
	Plan plan;
	plan.active.assign(patches.size(), uint8_t(1));
	if (patches.size() < 2)
		return plan;

	std::vector<detail::Box> boxes;
	boxes.reserve(patches.size());
	std::unordered_map<uint32_t, std::vector<detail::Value>> valuesByLabel;
	valuesByLabel.reserve(patches.size());
	for (uint32_t patchIdx = 0; patchIdx < patches.size(); ++patchIdx) {
		boxes.emplace_back(detail::ToBox(patches[patchIdx].rect));
		valuesByLabel[patches[patchIdx].label].emplace_back(boxes.back(), patchIdx);
	}

	std::unordered_map<uint32_t, detail::RTree> trees;
	trees.reserve(valuesByLabel.size());
	for (auto& labelValues : valuesByLabel)
		trees.emplace(labelValues.first, detail::RTree(labelValues.second.begin(), labelValues.second.end()));

	std::vector<detail::Value> spatialCandidates;
	std::vector<uint32_t> candidates;
	for (uint32_t bigIdx = 0; bigIdx < patches.size(); ++bigIdx) {
		if (!plan.active[bigIdx])
			continue;
		++plan.stats.queries;
		detail::RTree& tree(trees.at(patches[bigIdx].label));
		spatialCandidates.clear();
		tree.query(detail::bgi::intersects(boxes[bigIdx]), std::back_inserter(spatialCandidates));
		plan.stats.spatialCandidates += spatialCandidates.size();
		candidates.clear();
		candidates.reserve(spatialCandidates.size());
		for (const detail::Value& candidate : spatialCandidates)
			candidates.emplace_back(candidate.second);
		std::sort(candidates.begin(), candidates.end());
		candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
		for (const uint32_t smallIdx : candidates) {
			if (smallIdx == bigIdx || !plan.active[smallIdx])
				continue;
			++plan.stats.containmentChecks;
			if (!detail::IsContainedIn(patches[smallIdx].rect, patches[bigIdx].rect))
				continue;
			plan.merges.push_back({bigIdx, smallIdx});
			plan.active[smallIdx] = 0;
			++plan.stats.merges;
			const size_t numRemoved(tree.remove(detail::Value(boxes[smallIdx], smallIdx)));
			ASSERT(numRemoved == 1);
			(void)numRemoved;
		}
	}
	return plan;
}

} // namespace PATCHMERGE
} // namespace MVS

#endif // _MVS_TEXTUREPATCHMERGE_H_
