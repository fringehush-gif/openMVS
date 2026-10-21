#include <cuda_runtime.h>
#include <cassert>
#include "TextureCUDA.h"
#include <chrono>
#include <climits>
#include <cmath>
#include <mutex>

namespace MVS { namespace CUDA {
namespace {
__device__ float Edge(float ax, float ay, float bx, float by, float px, float py)
{
	// cv::Point_<float> rounds the differences, then cross() widens to double.
	const double dx=double(__fsub_rn(px,ax)), dy=double(__fsub_rn(py,ay));
	const double ex=double(__fsub_rn(bx,ax)), ey=double(__fsub_rn(by,ay));
	return __double2float_rn(__dsub_rn(__dmul_rn(dx,ey),__dmul_rn(dy,ex)));
}
__device__ bool DepthAt(const TextureTriangle& t, int x, int y, float& depth)
{
	// Device assertions use CUDA's builtin; project ASSERT is host-only.
	assert(x>=t.minX && x<=t.maxX && y>=t.minY && y<=t.maxY);
	const float b0=__fmul_rn(Edge(t.x[1],t.y[1],t.x[2],t.y[2],x,y),t.invArea);
	if (b0<0) return false;
	const float b1=__fmul_rn(Edge(t.x[2],t.y[2],t.x[0],t.y[0],x,y),t.invArea);
	if (b1<0) return false;
	const float b2=__fmul_rn(Edge(t.x[0],t.y[0],t.x[1],t.y[1],x,y),t.invArea);
	if (b2<0) return false;
	const float p0=__fmul_rn(__fmul_rn(b0,t.z[1]),t.z[2]);
	const float p1=__fmul_rn(__fmul_rn(b1,t.z[0]),t.z[2]);
	const float p2=__fmul_rn(__fmul_rn(b2,t.z[0]),t.z[1]);
	const float inv=__fdiv_rn(1.f,__fadd_rn(__fadd_rn(p0,p1),p2));
	depth=__fadd_rn(__fadd_rn(__fmul_rn(__fmul_rn(inv,p0),t.z[0]),
	                         __fmul_rn(__fmul_rn(inv,p1),t.z[1])),
	                         __fmul_rn(__fmul_rn(inv,p2),t.z[2]));
	return true;
}
__global__ void Raster(const TextureTriangle* triangles, TextureCount* counts,
	uint64_t* keys, int width)
{
	const int i=blockIdx.x;
	const TextureTriangle t=triangles[i];
	const int boxWidth=t.maxX-t.minX+1;
	const uint32_t count=uint32_t(boxWidth)*(t.maxY-t.minY+1);
	uint32_t projected=0;
	// count <= INT_MAX; the final unsigned increment remains below UINT_MAX.
	for (uint32_t p=threadIdx.x;p<count;p+=blockDim.x) {
		const int x=t.minX+int(p%boxWidth), y=t.minY+int(p/boxWidth);
		float depth;
		if (!DepthAt(t,x,y,depth)) continue;
		// Every coverage callback counts, including occluded and repeated faces.
		++projected;
		assert(depth>0.f && isfinite(depth));
		// An ordered depth/rank key, not an unordered pair of object indices.
		const uint64_t key=(uint64_t(__float_as_uint(depth))<<32)|t.rank;
		atomicMin(reinterpret_cast<unsigned long long*>(keys+y*width+x),
			static_cast<unsigned long long>(key));
	}
	atomicAdd(&counts[i].projected,projected);
}
__global__ void Collect(const TextureTriangle* triangles, int count,
	const uint64_t* keys, int width, float scaleX, float scaleY, int imageWidth,
	const unsigned char* bgr, TextureCount* counts)
{
	const int i=blockIdx.x*blockDim.x+threadIdx.x;
	if (i>=count) return;
	const TextureTriangle t=triangles[i];
	uint32_t visible=0;
	float color[3]{};
	for (int y=t.minY;y<=t.maxY;++y) for (int x=t.minX;x<=t.maxX;++x) {
		if (uint32_t(keys[y*width+x])!=t.rank) continue;
		if (bgr) {
			const int imgY=int(__fmul_rn(float(y),scaleY));
			const int imgX=int(__fmul_rn(float(x),scaleX));
			const size_t source=(size_t(imgY)*imageWidth+imgX)*3;
			if (!visible) for (int c=0;c<3;++c) color[c]=float(bgr[source+c]);
			else for (int c=0;c<3;++c) color[c]=__fadd_rn(color[c],float(bgr[source+c]));
		}
		++visible;
	}
	counts[i].visible=visible;
	for (int c=0;c<3;++c) counts[i].color[c]=color[c];
}
bool Checked(cudaError_t result, const char* operation)
{
	ASSERT(operation);
	if (result==cudaSuccess) return true;
	VERBOSE("Texture CUDA %s: %s", operation, cudaGetErrorString(result));
	return false;
}
struct Allocator {
	bool Free(void* pointer) { ASSERT(pointer); return Checked(cudaFree(pointer),"free"); }
	bool Allocate(void** pointer, size_t bytes)
{
		ASSERT(pointer && bytes>0); return Checked(cudaMalloc(pointer,bytes),"allocate");
	}
};
struct Workspace {
	int device=-1;
	cudaStream_t stream=nullptr;
	cudaEvent_t events[5]{};
	Allocator allocator;
	detail::TextureStorage<Allocator> storage{allocator};
	~Workspace()
{
		if (device<0) return;
		// Never free device-owned state under a foreign current context.
		if (cudaSetDevice(device)!=cudaSuccess) return;
		if (stream) cudaStreamSynchronize(stream);
		storage.Release();
		for (auto e:events) if(e) cudaEventDestroy(e);
		if (stream) cudaStreamDestroy(stream);
	}
};
std::mutex workspaceMutex;
Workspace workspace;
using Clock=std::chrono::steady_clock;
} // namespace

bool SelectTextureCUDADevice(int requested, int& selected)
{
	ASSERT(requested>=-1);
	int count=0;
	if (!Checked(cudaGetDeviceCount(&count),"device count") || count==0) return false;
	if (requested>=count) return false;
	selected=-1;
	cudaDeviceProp best{};
	int64_t bestPerformance=0;
	for (int device=0;device<count;++device) {
		if (requested>=0 && device!=requested) continue;
		cudaDeviceProp props{};
		if (!Checked(cudaGetDeviceProperties(&props,device),"device properties")) return false;
		int clockRate=0, computeMode=0;
		if (!Checked(cudaDeviceGetAttribute(&clockRate,cudaDevAttrClockRate,device),"device clock rate")) return false;
		if (!Checked(cudaDeviceGetAttribute(&computeMode,cudaDevAttrComputeMode,device),"device compute mode")) return false;
		const int64_t performance=int64_t(props.multiProcessorCount)*clockRate;
		if (computeMode==cudaComputeModeProhibited || props.major<3 ||
			(props.major==3 && props.minor<5))
			{
			if (requested>=0) return false;
			continue;
		}
		if (selected<0 || props.major>best.major ||
			(props.major==best.major && props.minor>best.minor) ||
			(props.major==best.major && props.minor==best.minor &&
			 performance>bestPerformance))
			 {
			selected=device; best=props; bestPerformance=performance;
		}
	}
	return selected>=0;
}

bool RunTextureCUDA(int width, int height, int imageWidth, int imageHeight,
	const std::vector<TextureTriangle>& triangles, const unsigned char* bgr,
	std::vector<TextureCount>& counts, TextureTiming& timing,
	std::vector<uint64_t>* visibility, int device)
{
	ASSERT(width>0 && height>0 && imageWidth>0 && imageHeight>0 && device>=-1);
	ASSERT(uint64_t(width)*height<=INT_MAX && uint64_t(imageWidth)*imageHeight<=INT_MAX);
	ASSERT(triangles.size()<INT_MAX);
	for (const auto& t:triangles) {
		ASSERT(t.minX>=0 && t.minY>=0 && t.maxX<width && t.maxY<height);
		ASSERT(t.minX<=t.maxX && t.minY<=t.maxY && t.rank<UINT_MAX);
		ASSERT(t.invArea>0.f && std::isfinite(t.invArea));
		for (int v=0;v<3;++v) ASSERT(t.z[v]>0.f && std::isfinite(t.z[v]));
	}
	std::lock_guard<std::mutex> lock(workspaceMutex);
	timing={};
	const auto begin=Clock::now();
	auto& w=workspace;
	if (device<0 && w.device>=0) device=w.device;
	if (!detail::TextureDeviceMatches(w.device,device)) {
		VERBOSE("Texture CUDA workspace bound to %d, requested %d", w.device, device);
		return false;
	}
	int selected=-1;
	if (!SelectTextureCUDADevice(device,selected)) return false;
	int previous=-1;
	if (!Checked(cudaGetDevice(&previous),"get caller device") ||
		!Checked(cudaSetDevice(selected),"select texture device")) return false;
	struct RestoreDevice {
		int previous;
		~RestoreDevice() { cudaSetDevice(previous); }
	} restore{previous};
	w.device=selected;
	if (!w.stream && !Checked(cudaStreamCreateWithFlags(&w.stream,cudaStreamNonBlocking),"stream")) return false;
	// Both success and failure return only after queued caller-buffer transfers end.
	auto synchronize=[&]() { Checked(cudaStreamSynchronize(w.stream),"failure drain"); };
	detail::TextureFailureDrain<decltype(synchronize)> drain(synchronize);
	for (auto& event:w.events) if (!event && !Checked(cudaEventCreate(&event),"timing event")) return false;
	// Drain before any reallocation, including retries after grouped partial failure.
	if (!Checked(cudaStreamSynchronize(w.stream),"before workspace growth")) return false;
	const size_t pixels=size_t(width)*height;
	const size_t colorBytes=bgr ? size_t(imageWidth)*imageHeight*3 : 0;
	auto& s=w.storage;
	if (!s.Ensure(triangles.size(),pixels,colorBytes)) return false;
	counts.resize(triangles.size());
	timing.deviceBytes=s.pixelCapacity*8+s.colorCapacity+
		s.triangleCapacity*(sizeof(TextureTriangle)+sizeof(TextureCount));
	if (!Checked(cudaEventRecord(w.events[0],w.stream),"upload start") ||
		!Checked(cudaMemsetAsync(s.keys,0xff,pixels*8,w.stream),"clear visibility") ||
		(!triangles.empty() && (!Checked(cudaMemsetAsync(s.counts,0,triangles.size()*sizeof(TextureCount),w.stream),"clear counts") ||
		 !Checked(cudaMemcpyAsync(s.triangles,triangles.data(),triangles.size()*sizeof(TextureTriangle),cudaMemcpyHostToDevice,w.stream),"upload triangles"))) ||
		(bgr && !Checked(cudaMemcpyAsync(s.bgr,bgr,colorBytes,cudaMemcpyHostToDevice,w.stream),"upload BGR")) ||
		!Checked(cudaEventRecord(w.events[1],w.stream),"upload end")) return false;
	if (!triangles.empty()) Raster<<<triangles.size(),128,0,w.stream>>>(s.triangles,s.counts,s.keys,width);
	if (!Checked(cudaGetLastError(),"raster launch") || !Checked(cudaEventRecord(w.events[2],w.stream),"raster end")) return false;
	if (!triangles.empty()) Collect<<<(triangles.size()+127)/128,128,0,w.stream>>>(s.triangles,int(triangles.size()),s.keys,width,
		float(imageWidth)/width,float(imageHeight)/height,imageWidth,bgr?s.bgr:nullptr,s.counts);
	if (!Checked(cudaGetLastError(),"collect launch") || !Checked(cudaEventRecord(w.events[3],w.stream),"collect end")) return false;
	if (!counts.empty() && !Checked(cudaMemcpyAsync(counts.data(),s.counts,counts.size()*sizeof(TextureCount),cudaMemcpyDeviceToHost,w.stream),"download counts")) return false;
	if (visibility) {
		visibility->resize(pixels);
		if (!Checked(cudaMemcpyAsync(visibility->data(),s.keys,pixels*8,cudaMemcpyDeviceToHost,w.stream),"download oracle visibility")) return false;
	}
	if (!Checked(cudaEventRecord(w.events[4],w.stream),"download end") ||
		!Checked(cudaStreamSynchronize(w.stream),"download wait")) return false;
	double* phases[]={&timing.uploadMs,&timing.rasterMs,&timing.scoreMs,&timing.downloadMs};
	for (int i=0;i<4;++i) {
		float ms=0;
		if (!Checked(cudaEventElapsedTime(&ms,w.events[i],w.events[i+1]),"phase time")) return false;
		*phases[i]=ms;
	}
	timing.totalMs=std::chrono::duration<double,std::milli>(Clock::now()-begin).count();
	drain.complete=true;
	return true;
}
} } // namespace MVS::CUDA
