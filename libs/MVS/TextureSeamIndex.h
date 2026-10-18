/*
 * TextureSeamIndex.h
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
#ifndef _MVS_TEXTURESEAMINDEX_H_
#define _MVS_TEXTURESEAMINDEX_H_
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>
namespace MVS {
namespace TEXTURESEAMINDEX {
struct Index { std::vector<size_t> offsets; std::vector<uint32_t> vertices; };
template <typename SeamVertices>
inline Index Build(const SeamVertices& seamVertices, size_t numPatches)
{
	ASSERT(seamVertices.size() <= std::numeric_limits<uint32_t>::max());
	Index index;
	index.offsets.assign(numPatches + 1, 0);
	// BuildPatch uses SeamVertex::GetPatch, so each patch occurs at most once per seam.
	for (const auto& seamVertex : seamVertices)
		for (const auto& patch : seamVertex.patches) {
			ASSERT(patch.idxPatch < numPatches);
			++index.offsets[patch.idxPatch + 1];
		}
	for (size_t patch = 0; patch < numPatches; ++patch)
		index.offsets[patch + 1] += index.offsets[patch];
	index.vertices.resize(index.offsets.back());
	std::vector<size_t> write(index.offsets.begin(), index.offsets.end() - 1);
	for (uint32_t vertex = 0; vertex < seamVertices.size(); ++vertex)
		for (const auto& patch : seamVertices[vertex].patches)
			index.vertices[write[patch.idxPatch]++] = vertex;
	return index;
}
} // namespace TEXTURESEAMINDEX
} // namespace MVS
#endif // _MVS_TEXTURESEAMINDEX_H_
