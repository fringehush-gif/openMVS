/*
* TextureFaceIndex.h
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU Affero General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Affero General Public License for more details.
*
* You should have received a copy of the GNU Affero General Public License
* along with this program. If not, see <http://www.gnu.org/licenses/>.
*
* Additional Terms:
* You are required to preserve legal notices and author attributions in
* that material or in the Appropriate Legal Notices displayed by works
* containing it.
*/

#ifndef _MVS_TEXTUREFACEINDEX_H_
#define _MVS_TEXTUREFACEINDEX_H_

namespace MVS {

// Active dense face IDs in their original order.
// Counts form a Fenwick tree, supporting ordered selection and removal.
class RemainingFaceIndex
{
	Mesh::FaceIdxArr counts;
	size_t remaining;
	size_t selectStep;

	public:
	explicit RemainingFaceIndex(Mesh::FIndex size) :
	    remaining(size), selectStep(1)
	{
		counts.ResizeExact(size);
		for (size_t i = 1; i <= counts.size(); ++i)
			counts[i - 1] = Mesh::FIndex(i & (~i + 1));
		while (selectStep <= counts.size() / 2)
			selectStep <<= 1;
	}
	size_t GetSize() const { return remaining; }
	Mesh::FIndex Select(size_t rank) const
	{
		ASSERT(rank < remaining);
		size_t position(0);
		for (size_t step = selectStep; step; step >>= 1) {
			const size_t next(position + step);
			if (next <= counts.size() && counts[next - 1] <= rank) {
				position = next;
				rank -= counts[next - 1];
			}
		}
		ASSERT(position < counts.size());
		return Mesh::FIndex(position);
	}
	void Remove(Mesh::FIndex face)
	{
		ASSERT(face < counts.size() && remaining > 0);
		for (size_t i = size_t(face) + 1; i <= counts.size(); i += i & (~i + 1)) {
			ASSERT(counts[i - 1] > 0);
			--counts[i - 1];
		}
		--remaining;
	}
};

} // namespace MVS

#endif
