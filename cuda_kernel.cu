#ifndef RECONSTRUCTION_KERNEL_CU
#define RECONSTRUCTION_KERNEL_CU

#include <stdio.h>
#include <string.h>
#include <cuda_runtime_api.h>
#include "cutil_math.h"
#include "defines.h"
#include "tables.h"
#include "structdata.h"

//纹理参考保存查找表
texture<uint, 1, cudaReadModeElementType> edgeTex;
texture<uint, 1, cudaReadModeElementType> triTex;
texture<uint, 1, cudaReadModeElementType> numVertsTex;
texture<uchar, 1, cudaReadModeElementType> mtwoTex1;
texture<uchar, 1, cudaReadModeElementType> mtwoTex2;
texture<uchar, 1, cudaReadModeElementType> mtwoTex3;
texture<uchar, 1, cudaReadModeElementType> mtwoTex4;
texture<uchar, 1, cudaReadModeElementType> mtwoTex5;

extern "C" void allocateTexturesRef(uint **d_edgeTable, uint **d_triTable,  uint **d_numVertsTable,uchar **dimage1,uchar **dimage2,uchar **dimage3,uchar **dimage4,uchar **dimage5)
{
    cudaMalloc((void**) d_edgeTable, 256*sizeof(uint));
    cudaMemcpy((void *)*d_edgeTable, (void *)edgeTable, 256*sizeof(uint), cudaMemcpyHostToDevice);
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);
    cudaBindTexture(0, edgeTex, *d_edgeTable, channelDesc);

    cudaMalloc((void**) d_triTable, 256*16*sizeof(uint));
    cudaMemcpy((void *)*d_triTable, (void *)triTable, 256*16*sizeof(uint), cudaMemcpyHostToDevice);
    cudaBindTexture(0, triTex, *d_triTable, channelDesc);

    cudaMalloc((void**) d_numVertsTable, 256*sizeof(uint));
    cudaMemcpy((void *)*d_numVertsTable, (void *)numVertsTable, 256*sizeof(uint), cudaMemcpyHostToDevice);
    cudaBindTexture(0, numVertsTex, *d_numVertsTable, channelDesc);
    
    cudaMalloc((void**) dimage1, 640*512*sizeof(uchar));
    cudaMemset((void *)*dimage1,0,640*512);
    cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
    cudaBindTexture(0, mtwoTex1, *dimage1, channelDesc2);
    
    cudaMalloc((void**) dimage2, 640*512*sizeof(uchar));
    cudaMemset((void *)*dimage2,0,640*512);
    cudaBindTexture(0, mtwoTex2, *dimage2, channelDesc2);
    
    cudaMalloc((void**) dimage3, 640*512*sizeof(uchar));
    cudaMemset((void *)*dimage3,0,640*512);
    cudaBindTexture(0, mtwoTex3, *dimage3, channelDesc2);
    
    cudaMalloc((void**) dimage4, 640*512*sizeof(uchar));
    cudaMemset((void *)*dimage4,0,640*512);
    cudaBindTexture(0, mtwoTex4, *dimage4, channelDesc2);
    
    cudaMalloc((void**) dimage5, 640*512*sizeof(uchar));
    cudaMemset((void *)*dimage5,0,640*512);
    cudaBindTexture(0, mtwoTex5, *dimage5, channelDesc2);
}
//计算在各个成像平面上的纹理坐标
__device__ float2 calcTexCordl(float objx, float objy, float objz) 
{
	float2 result;
	float x,y,z;
	float temx,temy,temz;
	x=(-814.181) * objx +(-254.218) * objy +(-301.903) * objz +393436 *1;
	y=65.0634 * objx +440.248 * objy +(-732.481)* objz +209546*1;
	z=0.091213* objx +(-0.732029)* objy +(-0.67514)* objz +1273.78*1;
	temx=x/z;
	temy=y/z;
	result=make_float2((temx/639),((511-temy)/511));
	return result;
}
__device__ float2 calcTexCord2(float objx, float objy, float objz) 
{
	float2 result;
	float x,y,z;
	float temx,temy,temz;
	x=(-481.117) * objx +(767.8) * objy +(-41.382) * objz +268432 *1;
	y=243.437 * objx +(-9.67565) * objy +(-450.11)* objz +119411*1;
	z=(-1.45133)* objx  +908.513*1;
	temx=x/z;
	temy=y/z;
	result=make_float2((temx/639),((511-temy)/511));
	return result;
}
__device__ float2 calcTexCord3(float objx, float objy, float objz) 
{
	float2 result;
	float x,y,z;
	float temx,temy,temz;
	x=(866.417) * objx +(171.75) * objy +(-189.775) * objz +206639 *1;
	y=4.13581* objx +(-471.062) * objy +(-744.419)* objz +259510*1;
	z=0.051583* objx +(0.689672)* objy +(-0.722282)* objz +919.593*1;
	temx=x/z;
	temy=y/z;
	result=make_float2((temx/639),((511-temy)/511));
	return result;
}
__device__ float2 calcTexCord4(float objx, float objy, float objz) 
{
	float2 result;
	float x,y,z;
	float temx,temy,temz;
	x=(165.335) * objx +(883.286) * objy +(-170.112) * objz +296392 *1;
	y=838.353* objx +59.3904* objy +(340.12)* objz +234738*1;
	z=0.718938* objx +(0.132446)* objy +(-0.682339)* objz +979.866*1;
	temx=x/z;
	temy=y/z;
	result=make_float2((temx/639),((511-temy)/511));
	return result;
}
__device__ float2 calcTexCord5(float objx, float objy, float objz) 
{
	float2 result;
	float x,y,z;
	float temx,temy,temz;
	x=(-841.698) * objx +(-14.4216) * objy +(-334.349) * objz +489542 *1;
	y=8.54247 * objx +838.757* objy +(-263.983)* objz +332716*1;
	z=0.02968* objx +(-0.048834)* objy +(-0.998366)* objz +1393.59*1;
	temx=x/z;
	temy=y/z;
	result=make_float2((temx/639),((511-temy)/511));
	return result;
}

//计算体素位置
__device__ uint3 calcGridPos(uint i, uint3 gridSizeShift, uint3 gridSizeMask)
{
    uint3 gridPos;
    gridPos.x = i & gridSizeMask.x;
    gridPos.y = (i >> gridSizeShift.y) & gridSizeMask.y;
    gridPos.z = (i >> gridSizeShift.z) & gridSizeMask.z;
    return gridPos;
}

//利用可视外壳重建方法的原理并行计算体素的类型
__global__ void  visHullBasedClassifyVoxel(uint* voxelVerts, uint *voxelOccupied, uint numVoxels,
                              struct projNode *dproj1,struct projNode *dproj2,struct projNode *dproj3,struct projNode *dproj4,struct projNode *dproj5)
{
    uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
    uint i = __mul24(blockId, blockDim.x) + threadIdx.x;
    uint cubeindex=0;
	int ex,ey;
	unsigned char t;

    //根据八个顶点和前景轮廓的关系对体素进行分类
    //用纹理内存保存随机访问的二值化图象信息，避免了全局内存的对齐问题
	for(int iVertexTest= 0;iVertexTest<8;iVertexTest++)
	{	
		t=0;	
		ex=dproj1[i].x[iVertexTest];
		ey=dproj1[i].y[iVertexTest];
		t+=tex1Dfetch(mtwoTex1,ey*640+ex);
		
		ex=dproj2[i].x[iVertexTest];
		ey=dproj2[i].y[iVertexTest];
		t+=tex1Dfetch(mtwoTex2,ey*640+ex);
		
		ex=dproj3[i].x[iVertexTest];
		ey=dproj3[i].y[iVertexTest];
		t+=tex1Dfetch(mtwoTex3,ey*640+ex);

		ex=dproj4[i].x[iVertexTest];
		ey=dproj4[i].y[iVertexTest];
		t+=tex1Dfetch(mtwoTex4,ey*640+ex);

		ex=dproj5[i].x[iVertexTest];
		ey=dproj5[i].y[iVertexTest];
		t+=tex1Dfetch(mtwoTex5,ey*640+ex);
		
		if (t==5)		
			cubeindex |=1<<iVertexTest;
	 }
    //从纹理内存中读取交点个数
    uint numVerts = tex1Dfetch(numVertsTex,cubeindex);
    if (i < numVoxels) 
    {
        voxelVerts[i] = numVerts;
        voxelOccupied[i] = (numVerts > 0);
    }
}

extern "C" void parallelClassifyAllVoxels( dim3 grid, dim3 threads, uint* voxelVerts, uint *voxelOccupied, uint numVoxels,
									struct projNode *dproj1,struct projNode *dproj2,struct projNode *dproj3,struct projNode *dproj4,struct projNode *dproj5)
{
    // 启动内核
    visHullBasedClassifyVoxel<<<grid, threads>>>(voxelVerts,voxelOccupied,numVoxels,dproj1,dproj2,dproj3,dproj4,dproj5);
}

// 并行压缩体素数组
__global__ void compactVoxels(uint *compactedVoxelArray, uint *voxelOccupied, uint *voxelOccupiedScan, uint numVoxels) 
{
    uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
    uint i = __mul24(blockId, blockDim.x) + threadIdx.x;

    if (voxelOccupied[i] && (i < numVoxels)) 
    {
        compactedVoxelArray[ voxelOccupiedScan[i] ] = i;
    }
}

extern "C" void parallelCompactVoxels(dim3 grid, dim3 threads, uint *compactedVoxelArray, uint *voxelOccupied, uint *voxelOccupiedScan, uint numVoxels)
{
    compactVoxels<<<grid, threads>>>(compactedVoxelArray, voxelOccupied,voxelOccupiedScan, numVoxels);
}

//设备函数，取两个点之间的中点
__device__ float3 myvertexInterp(float3 p0, float3 p1)                   
{
	return lerp(p0, p1, 0.5);
} 

//利用行进立方体方法提取等值面
__global__ void generateTriangles(float3 *pos,float3 *normal,float2 *tex1,float2 *tex2,float2 *tex3,float2 *tex4,float2 *tex5, uint *compactedVoxelArray, uint *numVertsScanned,uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask,
                                  float3 voxelSize, uint activeVoxels, uint maxVerts,struct projNode *dproj1,struct projNode *dproj2,struct projNode *dproj3,struct projNode *dproj4,
                                  struct projNode *dproj5)
{
    uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
    uint i = __mul24(blockId, blockDim.x) + threadIdx.x;

    if (i > activeVoxels - 1) {
        i = activeVoxels - 1;
    }

#if SKIP_EMPTY_VOXELS
    uint voxel = compactedVoxelArray[i];
#else
    uint voxel = i;
#endif

    //计算体素的位置
    uint3 gridPos = calcGridPos(voxel, gridSizeShift, gridSizeMask);
	float3 norm,v0,v1,v2;
    float3 p;
    p.x = -300.0f + (gridPos.x * voxelSize.x);
    p.y = -300.0f + (gridPos.y * voxelSize.y);
    p.z = -300.0f + (gridPos.z * voxelSize.z);

    // 计算体素各个顶点的坐标
    float3 v[8];
    v[0] = p;
    v[1] = p + make_float3(voxelSize.x, 0, 0);
    v[2] = p + make_float3(voxelSize.x, voxelSize.y, 0);
    v[3] = p + make_float3(0, voxelSize.y, 0);
    v[4] = p + make_float3(0, 0, voxelSize.z);
    v[5] = p + make_float3(voxelSize.x, 0, voxelSize.z);
    v[6] = p + make_float3(voxelSize.x, voxelSize.y, voxelSize.z);
    v[7] = p + make_float3(0, voxelSize.y, voxelSize.z);

    // 计算体素类型
    uint cubeindex=0;
	int ex,ey;
	unsigned char t;
	for(int iVertexTest= 0;iVertexTest<8;iVertexTest++)
	{	
		t=0;	
		ex=dproj1[voxel].x[iVertexTest];
		ey=dproj1[voxel].y[iVertexTest];
		t+=tex1Dfetch(mtwoTex1,ey*640+ex);
		
		ex=dproj2[voxel].x[iVertexTest];
		ey=dproj2[voxel].y[iVertexTest];
		t+=tex1Dfetch(mtwoTex2,ey*640+ex);
		
		ex=dproj3[voxel].x[iVertexTest];
		ey=dproj3[voxel].y[iVertexTest];
		t+=tex1Dfetch(mtwoTex3,ey*640+ex);

		ex=dproj4[voxel].x[iVertexTest];
		ey=dproj4[voxel].y[iVertexTest];
		t+=tex1Dfetch(mtwoTex4,ey*640+ex);

		ex=dproj5[voxel].x[iVertexTest];
		ey=dproj5[voxel].y[iVertexTest];
		t+=tex1Dfetch(mtwoTex5,ey*640+ex);
		
		if (t==5)		
			cubeindex |=1<<iVertexTest;
	 }

//利用共享内存提高效率存储存储各边中点坐标	 
#if USE_SHARED   
	__shared__ float3 vertlist[12*NTHREADS];
	vertlist[threadIdx.x] = myvertexInterp(v[0], v[1]);
    vertlist[NTHREADS+threadIdx.x] = myvertexInterp(v[1], v[2]);
    vertlist[(NTHREADS*2)+threadIdx.x] = myvertexInterp(v[2], v[3]);
    vertlist[(NTHREADS*3)+threadIdx.x] = myvertexInterp(v[3], v[0]);
	vertlist[(NTHREADS*4)+threadIdx.x] = myvertexInterp(v[4], v[5]);
    vertlist[(NTHREADS*5)+threadIdx.x] = myvertexInterp(v[5], v[6]);
    vertlist[(NTHREADS*6)+threadIdx.x] = myvertexInterp(v[6], v[7]);
    vertlist[(NTHREADS*7)+threadIdx.x] = myvertexInterp(v[7], v[4]);
	vertlist[(NTHREADS*8)+threadIdx.x] = myvertexInterp(v[0], v[4]);
    vertlist[(NTHREADS*9)+threadIdx.x] = myvertexInterp(v[1], v[5]);
    vertlist[(NTHREADS*10)+threadIdx.x]= myvertexInterp(v[2], v[6]);
    vertlist[(NTHREADS*11)+threadIdx.x]=myvertexInterp(v[3], v[7]);
    __syncthreads();

//利用局部内存存储各边中点坐标
#else
	float3 vertlist[12];
    vertlist[0] = myvertexInterp(v[0], v[1]);
    vertlist[1] = myvertexInterp(v[1], v[2]);
    vertlist[2] = myvertexInterp(v[2], v[3]);
    vertlist[3] = myvertexInterp(v[3], v[0]);

	vertlist[4] = myvertexInterp(v[4], v[5]);
    vertlist[5] = myvertexInterp(v[5], v[6]);
    vertlist[6] = myvertexInterp(v[6], v[7]);
    vertlist[7] = myvertexInterp(v[7], v[4]);

	vertlist[8] = myvertexInterp(v[0], v[4]);
    vertlist[9] = myvertexInterp(v[1], v[5]);
    vertlist[10] = myvertexInterp(v[2], v[6]);
    vertlist[11] = myvertexInterp(v[3], v[7]);
#endif

    // 提取三角形
    uint numVerts = tex1Dfetch(numVertsTex, cubeindex);
    for(int i=0; i<numVerts; i+=3) 
    {
        uint index = numVertsScanned[voxel] + i;

        float3 *v[3];
        uint edge;
        edge = tex1Dfetch(triTex, (cubeindex*16) + i);
	#if USE_SHARED
		v[0] = &vertlist[(edge*NTHREADS)+threadIdx.x];
		v0=vertlist[(edge*NTHREADS)+threadIdx.x];
	#else
		v[0] = &vertlist[edge];
		v0=vertlist[(edge*NTHREADS)+threadIdx.x];
	#endif

		edge = tex1Dfetch(triTex, (cubeindex*16) + i + 1);
	#if USE_SHARED
		v[1] = &vertlist[(edge*NTHREADS)+threadIdx.x];
		v1=vertlist[(edge*NTHREADS)+threadIdx.x];
	#else
		v[1] = &vertlist[edge];
		v1=vertlist[(edge*NTHREADS)+threadIdx.x];
	#endif

        edge = tex1Dfetch(triTex, (cubeindex*16) + i + 2);
	#if USE_SHARED
		v[2] = &vertlist[(edge*NTHREADS)+threadIdx.x];
		v2=vertlist[(edge*NTHREADS)+threadIdx.x];
	#else
		v[2] = &vertlist[edge];
		v2=vertlist[(edge*NTHREADS)+threadIdx.x];
	#endif

        // 计算顶点坐标，法线向量及其在各个成像平面上的纹理坐标

        if (index < (maxVerts - 3)) 
        {  
            pos[index] = (*v[0]);
            tex1[index] = calcTexCordl(pos[index].x,pos[index].y,pos[index].z);
            tex2[index] = calcTexCord2(pos[index].x,pos[index].y,pos[index].z);
            tex3[index] = calcTexCord3(pos[index].x,pos[index].y,pos[index].z);
            tex4[index] = calcTexCord4(pos[index].x,pos[index].y,pos[index].z);
            tex5[index] = calcTexCord5(pos[index].x,pos[index].y,pos[index].z);

            pos[index+1] = (*v[1]);
            tex1[index+1] = calcTexCordl(pos[index+1].x,pos[index+1].y,pos[index+1].z);
            tex2[index+1] = calcTexCord2(pos[index+1].x,pos[index+1].y,pos[index+1].z);
            tex3[index+1] = calcTexCord3(pos[index+1].x,pos[index+1].y,pos[index+1].z);
            tex4[index+1] = calcTexCord4(pos[index+1].x,pos[index+1].y,pos[index+1].z);
            tex5[index+1] = calcTexCord5(pos[index+1].x,pos[index+1].y,pos[index+1].z);

            pos[index+2] = (*v[2]);
            tex1[index+2] = calcTexCordl(pos[index+2].x,pos[index+2].y,pos[index+2].z);
            tex2[index+2] = calcTexCord2(pos[index+2].x,pos[index+2].y,pos[index+2].z);
            tex3[index+2] = calcTexCord3(pos[index+2].x,pos[index+2].y,pos[index+2].z);
            tex4[index+2] = calcTexCord4(pos[index+2].x,pos[index+2].y,pos[index+2].z);
            tex5[index+2] = calcTexCord5(pos[index+2].x,pos[index+2].y,pos[index+2].z);
            
            norm.x=(v2.y-v0.y)*(v1.z-v0.z)-(v2.z-v0.z)*(v1.y-v0.y);
            norm.y=(v2.z-v0.z)*(v1.x-v0.x)-(v2.x-v0.x)*(v1.z-v0.z);
            norm.z=(v2.x-v0.x)*(v1.y-v0.y)-(v1.x-v0.x)*(v2.y-v0.y);
            
            normal[index]=norm;
            normal[index+1]=norm;
            normal[index+2]=norm;
            
        }
    }
}
            
extern "C" void parallelGenerateTriangles(dim3 grid, dim3 threads,float3 *pos,float3 *normal,float2 *tex1,float2 *tex2,float2 *tex3,float2 *tex4,float2 *tex5,
                                           uint *compactedVoxelArray, uint *numVertsScanned,uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask,float3 voxelSize,uint activeVoxels, 
                                            uint maxVerts,struct projNode *dproj1,struct projNode *dproj2,struct projNode *dproj3,struct projNode *dproj4,struct projNode *dproj5)
{
    generateTriangles<<<grid, NTHREADS>>>(pos, normal,tex1,tex2,tex3,tex4,tex5, compactedVoxelArray, numVertsScanned, 
                                            gridSize, gridSizeShift, gridSizeMask,voxelSize,activeVoxels, 
                                            maxVerts,dproj1,dproj2,dproj3,dproj4,dproj5);
}




#endif





