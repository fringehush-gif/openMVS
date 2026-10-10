/*
 * Copyright (c) 2014-2021 SEACAVE
 * Author: cDc <cdc.seacave@gmail.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version. See <http://www.gnu.org/licenses/>.
 *
 * Additional Terms: You are required to preserve legal notices and author
 * attributions in that material or in the Appropriate Legal Notices displayed
 * by works containing it.
 */

#ifndef _MVS_PATCHMATCHVIEWSTATE_H_
#define _MVS_PATCHMATCHVIEWSTATE_H_

#include "../Common/Config.h"
#include <algorithm>

namespace MVS {
namespace CUDA {

// Host-only per-image transition shared by production and CUDA-free native tests.
// defaultTopK comes from fresh worker defaults, never the previous image's K.
inline void SetPatchMatchViewCount(int numViews, int defaultTopK, int& currentNumViews, int& currentTopK)
{
	ASSERT(numViews > 0 && defaultTopK > 0);
	currentNumViews = numViews;
	currentTopK = std::min(defaultTopK, currentNumViews);
	ASSERT(currentTopK > 0 && currentTopK <= currentNumViews);
}

} // namespace CUDA
} // namespace MVS

#endif // _MVS_PATCHMATCHVIEWSTATE_H_
