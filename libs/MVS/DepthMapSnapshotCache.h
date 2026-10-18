/*
* DepthMapSnapshotCache.h
*
* Copyright (c) 2014-2025 SEACAVE
*
* Author(s):
*
*      cDc <cdc.seacave@gmail.com>
*
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU Affero General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU Affero General Public License for more details.
*
* You should have received a copy of the GNU Affero General Public License
* along with this program.  If not, see <http://www.gnu.org/licenses/>.
*
*
* Additional Terms:
*
*      You are required to preserve legal notices and author attributions in
*      that material or in the Appropriate Legal Notices displayed by works
*      containing it.
*/

#pragma once
#ifndef _MVS_DEPTHMAPSNAPSHOTCACHE_H_
	#define _MVS_DEPTHMAPSNAPSHOTCACHE_H_

// I N C L U D E S /////////////////////////////////////////////////

	#include "DepthMap.h"

	#include <algorithm>
	#include <future>
	#include <list>
	#include <mutex>
	#include <unordered_map>

// S T R U C T S ///////////////////////////////////////////////////

namespace MVS {

struct DepthMapSnapshot
{
	String imageFileName;
	IIndexArr IDs;
	cv::Size imageSize;
	Camera camera;
	Depth dMin, dMax;
	DepthMap depthMap;
	NormalMap normalMap;
	ConfidenceMap confMap;
	bool bConfAdjusted = false;

	size_t GetMemorySize() const
	{
		ASSERT(!depthMap.empty());
		return depthMap.memory_size() + normalMap.memory_size() + confMap.memory_size();
	}
};

// A small, bounded cache for read-only snapshots shared by workers in one
// geometric-consistency pass. The mutable reference depth map must clone its
// data; neighbor depth maps may borrow the snapshot while holding a lease.
class DepthMapSnapshotCache
{
	public:
	typedef std::shared_ptr<const DepthMapSnapshot> SnapshotPtr;

	explicit DepthMapSnapshotCache(size_t memory) :
	    maxMemory(memory)
	{
		ASSERT(memory > 0);
	}

	// Called only after all workers have joined, before replacing the files.
	void Reset()
	{
		std::lock_guard<std::mutex> lock(mutex);
		ASSERT(loading.empty());
		ASSERT(std::all_of(entries.begin(), entries.end(), [](const Entry& entry) { return entry.snapshot.use_count() == 1; }));
		lookup.clear();
		entries.clear();
		usedMemory = 0;
	}

	SnapshotPtr Load(IIndex imageID, unsigned flags)
	{
		ASSERT(imageID != NO_ID);
		return Load(ComposeDepthFilePath(imageID, "dmap"), imageID, flags);
	}

	// An image ID identifies one immutable file until Reset; flags select the
	// depth-map planes this read needs.
	SnapshotPtr Load(const String& fileName, IIndex imageID, unsigned flags)
	{
		ASSERT(!fileName.empty());
		ASSERT(imageID != NO_ID);
		ASSERT((flags & HeaderDepthDataRaw::HAS_DEPTH) != 0);
		ASSERT((flags & HeaderDepthDataRaw::HAS_VIEWS) == 0);
		const uint64_t key(uint64_t(PairIdx(imageID, flags)));
		std::shared_future<SnapshotPtr> future;
		std::shared_ptr<std::packaged_task<SnapshotPtr()>> task;
		{
			std::lock_guard<std::mutex> lock(mutex);
			// A successfully loaded superset is usable without decoding again;
			// misses request exactly the required planes so a corrupt optional
			// plane cannot make a depth-only read fail.
			const unsigned bundles[] = {flags, flags | HeaderDepthDataRaw::HAS_NORMAL,
			                            flags | HeaderDepthDataRaw::HAS_NORMAL | HeaderDepthDataRaw::HAS_CONF};
			for (unsigned bundle : bundles) {
				const auto found(lookup.find(uint64_t(PairIdx(imageID, bundle))));
				if (found != lookup.end()) {
					entries.splice(entries.begin(), entries, found->second);
					return found->second->snapshot;
				}
			}
			const auto pending(loading.find(key));
			if (pending != loading.end()) {
				future = pending->second;
			} else {
				task = std::make_shared<std::packaged_task<SnapshotPtr()>>([fileName, flags]() -> SnapshotPtr {
					auto snapshot(std::make_shared<DepthMapSnapshot>());
					ViewsMap viewsMap;
					if (!ImportDepthDataRaw(fileName, snapshot->imageFileName,
					                        snapshot->IDs, snapshot->imageSize, snapshot->camera.K, snapshot->camera.R, snapshot->camera.C,
					                        snapshot->dMin, snapshot->dMax, snapshot->depthMap, snapshot->normalMap, snapshot->confMap,
					                        viewsMap, flags, &snapshot->bConfAdjusted))
						return SnapshotPtr();
					return snapshot;
				});
				future = task->get_future().share();
				loading.emplace(key, future);
			}
		}
		// Decode outside the cache lock. Concurrent readers of this image share
		// the result, including a missing-file failure or a load exception.
		if (task)
			(*task)();
		SnapshotPtr snapshot;
		try {
			snapshot = future.get();
		} catch (...) {
			if (task) {
				std::lock_guard<std::mutex> lock(mutex);
				loading.erase(key);
			}
			throw;
		}
		if (task) {
			std::lock_guard<std::mutex> lock(mutex);
			loading.erase(key);
			if (snapshot) {
				const size_t bytes(snapshot->GetMemorySize());
				if (bytes <= maxMemory) {
					for (auto entry = entries.end(); usedMemory + bytes > maxMemory && entry != entries.begin();) {
						--entry;
						if (entry->snapshot.use_count() == 1) {
							usedMemory -= entry->snapshot->GetMemorySize();
							lookup.erase(entry->key);
							entry = entries.erase(entry);
						}
					}
					// If all entries are leased, use this decoded map uncached;
					// waiting for admission could deadlock overlapping neighborhoods.
					if (usedMemory + bytes <= maxMemory) {
						entries.push_front({key, snapshot});
						lookup.emplace(key, entries.begin());
						usedMemory += bytes;
					}
				}
			}
		}
		return snapshot;
	}

	private:
	struct Entry
	{
		uint64_t key;
		SnapshotPtr snapshot;
	};
	typedef std::list<Entry> EntryList;
	EntryList entries;
	std::unordered_map<uint64_t, EntryList::iterator> lookup;
	std::unordered_map<uint64_t, std::shared_future<SnapshotPtr>> loading;
	std::mutex mutex;
	size_t maxMemory, usedMemory = 0;
};
/*----------------------------------------------------------------*/

} // namespace MVS

#endif // _MVS_DEPTHMAPSNAPSHOTCACHE_H_
