/*
* SceneTexture.cpp
*
* Copyright (c) 2014-2015 SEACAVE
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
#include "Mesh.h"
#include "TextureCUDA.h"
#include <algorithm>
#include <climits>
#include <cmath>
#include <cfenv>
#if defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
#include <xmmintrin.h>
#endif

// Internal production oracle shared only by SceneTexture and native Tests.
// RasterMesh is shared by production scoring and native tests.
// Native test hooks stay in this non-installed implementation header.
namespace MVS {
MVS_API bool PoissonBlendingTest();
MVS_API bool TexturePatchContainmentTest();
}

namespace MVS { namespace TextureScoring {
typedef Mesh::FIndex FIndex;
typedef TImage<cuint32_t> FaceMap;
struct RasterMesh : TRasterMesh<RasterMesh>
{
	typedef TRasterMesh<RasterMesh> Base;
	FaceMap& faceMap;
	FIndex idxFace;
	Image8U mask; // valid pixels of the image at its working resolution (empty: all valid)
	bool validFace;
	const float scaleMaskX, scaleMaskY; // working resolution over rendering resolution
	const int border; // keeps the vertices at least 2 px inside the working image, as GenerateTexture requires

	RasterMesh(const Mesh::VertexArr& _vertices, const Camera& _camera, DepthMap& _depthMap, FaceMap& _faceMap, const cv::Size& imageSize)
		: Base(_vertices, _camera, _depthMap), faceMap(_faceMap),
		scaleMaskX((float)imageSize.width / _faceMap.cols), scaleMaskY((float)imageSize.height / _faceMap.rows),
		border(CEIL2INT(3.f / MINF(scaleMaskX, scaleMaskY)) - 1) {}
	inline bool ProjectVertex(const Point3f& pt, int v, Triangle& t)
	{
		return (t.ptc[v] = camera.TransformPointW2C(Cast<REAL>(pt))).z > 0 &&
			depthMap.isInsideWithBorder(t.pti[v] = camera.TransformPointC2I(t.ptc[v]), border);
	}
	inline void Clear()
	{
		Base::Clear();
		faceMap.memset((uint8_t)NO_ID);
	}
	uint32_t* projArea = nullptr; // when set, counts every pixel the face covers, occluded or not
	void Raster(const ImageRef& pt, const Triangle& t, const Point3f& bary)
	{
		if (projArea)
			++*projArea;
		const Point3f pbary(PerspectiveCorrectBarycentricCoordinates(t, bary));
		const Depth z(ComputeDepth(t, pbary));
		ASSERT(z > Depth(0));
		Depth& depth = depthMap(pt);
		if (depth == 0 || depth > z) {
			depth = z;
			faceMap(pt) = validFace && (validFace = (mask.empty() || mask((int)(pt.y * scaleMaskY), (int)(pt.x * scaleMaskX)) != 0)) ? idxFace : NO_ID;
		}
	}
};

typedef Point3f Color;
typedef CLISTDEF0IDX(uint32_t,Mesh::FIndex) Counts;
typedef CLISTDEF0IDX(Color,Mesh::FIndex) Colors;

inline bool UniqueExplicitViews(const IIndexArr& views)
{
	IIndexArr sorted(views);
	std::sort(sorted.begin(),sorted.end());
	return std::adjacent_find(sorted.begin(),sorted.end())==sorted.end();
}
inline bool HostPrecisionEligible()
{
#if defined(__FAST_MATH__) || defined(__FMA__) || defined(_M_FP_FAST) || defined(_M_FP_CONTRACT) || \
    (defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1930 && \
     defined(__AVX2__) && !defined(_M_FP_STRICT))
	return false;
#elif defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
	// Match explicit nearest, gradual-underflow device arithmetic without changing
	// the caller's rounding, flush-to-zero or denormals-are-zero controls.
	return std::fegetround()==FE_TONEAREST && (_mm_getcsr() & 0xe040u)==0;
#else
	return false; // CUDA parity is currently qualified only with observable MXCSR.
#endif
}
inline bool Eligible(int optionalGpu, bool uniqueViews, const Image8U& mask, const Image8U3& image, const cv::Size& nativeSize)
{
	ASSERT(optionalGpu>=-2 && !image.empty() && nativeSize.width>0 && nativeSize.height>0);
	// Eligibility runs before any device/context/stream/allocation/transfer.
	return optionalGpu!=-2 && uniqueViews && HostPrecisionEligible() && image.isContinuous() &&
		uint64_t(nativeSize.width)*nativeSize.height<=INT_MAX && image.total()<=INT_MAX &&
		(mask.empty() || size_t(cv::countNonZero(mask))==mask.total());
}
inline bool PrepareTriangles(RasterMesh& rasterer, const Mesh::FaceArr& faces,
		const Mesh::FaceIdxArr& cameraFaces, std::vector<CUDA::TextureTriangle>& triangles)
{
	ASSERT(cameraFaces.size()<UINT_MAX);
	triangles.clear(); triangles.reserve(cameraFaces.size());
	RasterMesh::Triangle triangle;
	for (uint32_t rank=0;rank<cameraFaces.size();++rank) {
		const auto idxFace=cameraFaces[rank];
		ASSERT(idxFace<faces.size());
		const auto& facet=faces[idxFace];
		bool inside=true;
		for (int v=0;v<3;++v)
			if (!rasterer.ProjectVertex(rasterer.vertices[facet[v]],v,triangle)) { inside=false; break; }
		if (!inside) continue;
		const float area=EdgeFunction(triangle.pti[0],triangle.pti[1],triangle.pti[2]);
		if (area<=0) continue; // native winding/degenerate rejection
		CUDA::TextureTriangle t{};
		t.invArea=1.f/area; t.face=idxFace; t.rank=rank;
		// Conservative range eligibility; unusual camera/depth precision stays CPU.
		if (!std::isfinite(area) || !std::isfinite(t.invArea) || t.invArea<=0.f) return false;
		for (int v=0;v<3;++v) {
			t.x[v]=triangle.pti[v].x; t.y[v]=triangle.pti[v].y; t.z[v]=float(triangle.ptc[v].z);
			if (!std::isfinite(t.x[v]) || !std::isfinite(t.y[v]) ||
					!std::isfinite(t.z[v]) || t.z[v]<1e-10f || t.z[v]>1e10f) return false;
		}
		const auto size=rasterer.Size();
		t.minX=std::max(0,FLOOR2INT(MINF3(t.x[0],t.x[1],t.x[2])));
		t.minY=std::max(0,FLOOR2INT(MINF3(t.y[0],t.y[1],t.y[2])));
		t.maxX=std::min(size.width-1,CEIL2INT(MAXF3(t.x[0],t.x[1],t.x[2])));
		t.maxY=std::min(size.height-1,CEIL2INT(MAXF3(t.y[0],t.y[1],t.y[2])));
		triangles.push_back(t);
	}
	return triangles.size()<INT_MAX;
}
inline void RasterCPU(RasterMesh& rasterer, const Mesh::FaceArr& faces,
		const Mesh::FaceIdxArr& cameraFaces, Counts& projAreas)
{
	ASSERT(projAreas.size()==faces.size());
	ASSERT(rasterer.faceMap.size()==rasterer.depthMap.size());
	RasterMesh::Triangle triangle;
	RasterMesh::TriangleRasterizer triangleRasterizer(triangle,rasterer);
	rasterer.Clear();
	for (Mesh::FIndex idxFace:cameraFaces) {
		ASSERT(idxFace<faces.size());
		rasterer.validFace=true;
		const Mesh::Face& facet=faces[idxFace];
		rasterer.idxFace=idxFace;
		rasterer.projArea=&projAreas[idxFace];
		rasterer.Project(facet,triangleRasterizer);
		rasterer.projArea=nullptr;
		if (!rasterer.validFace) rasterer.Project(facet,triangleRasterizer);
	}
}
template<bool WithColor> inline void FoldVisible(const FaceMap& faceMap, const DepthMap& depthMap,
		const Image8U3& image, Counts& areas, Colors& colors)
{
	ASSERT(!faceMap.empty() && faceMap.size()==depthMap.size() && !image.empty());
	if constexpr (WithColor)
		ASSERT(colors.size()==areas.size());
	const float scaleWeightX((float)image.cols / faceMap.cols);
	const float scaleWeightY((float)image.rows / faceMap.rows);
	for (int j=0;j<faceMap.rows;++j) {
		for (int i=0;i<faceMap.cols;++i) {
			const Mesh::FIndex idxFace=faceMap(j,i).idx;
			ASSERT(idxFace==NO_ID || depthMap(j,i)>0);
			if (idxFace==NO_ID) continue;
			ASSERT(idxFace<areas.size());
			if constexpr (WithColor) {
				const Color color(image((int)(j*scaleWeightY),(int)(i*scaleWeightX)));
				if (areas[idxFace]++==0) colors[idxFace]=color;
				else colors[idxFace]+=color;
			} else ++areas[idxFace];
		}
	}
}
template<bool WithColor> inline void ImportCounts(const std::vector<CUDA::TextureTriangle>& triangles,
		const std::vector<CUDA::TextureCount>& counts, Counts& projected, Counts& visible, Colors& colors)
{
	ASSERT(triangles.size()==counts.size() && projected.size()==visible.size());
	if constexpr (WithColor)
		ASSERT(colors.size()==visible.size());
	for (size_t i=0;i<triangles.size();++i) {
		const auto face=triangles[i].face;
		ASSERT(face<projected.size());
		projected[face]+=counts[i].projected; // same uint32 wrap as repeated CPU callbacks
		if (!counts[i].visible) continue;
		// A repeated identical face can win only at its first traversal rank.
		ASSERT(visible[face]==0 && counts[i].visible<=counts[i].projected);
		visible[face]=counts[i].visible;
		if constexpr (WithColor) colors[face]=Color(counts[i].color[0],counts[i].color[1],counts[i].color[2]);
	}
}
inline float Quality(uint32_t visible, uint32_t projected)
{
	ASSERT(visible<=projected);
	float quality=(float)visible;
	if (visible<0.9f*projected && projected-visible>2) quality*=0.1f;
	return quality;
}
} } // namespace MVS::TextureScoring
