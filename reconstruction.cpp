
#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>  
#include <GL/glew.h>
#include <Cg/cg.h>    
#include <Cg/cgGL.h>
//#include <cutil_inline.h>
#include <cutil.h>
#include "defines.h"

#include "structdata.h"

#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <vector_types.h>
#include <vector_functions.h>

#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif
#include <GL/glaux.h>

#include "cudpp/cudpp.h"


#define Depth 64
#define DATA_SIZE 64*64*64
const unsigned int window_width = 512;
const unsigned int window_height = 512;



uint3 gridSizeLog2 = make_uint3(6,6,6);
uint3 gridSizeShift;
uint3 gridSize;
uint3 gridSizeMask;

float3 voxelSize;
uint numVoxels = 0;
uint maxVerts = 0;
uint activeVoxels = 0;
uint totalVerts = 0;


CUDPPHandle scanplan;

// device data
unsigned char *dimage1,*dimage2,*dimage3,*dimage4,*dimage5;

struct projNode *dproj1,*dproj2,*dproj3,*dproj4,*dproj5;
GLuint posVbo, normalVbo,texVbo1,texVbo2,texVbo3,texVbo4,texVbo5;

uchar *d_volume = 0;
uint  *d_voxelVerts = 0;
uint  *d_voxelVertsScan = 0;
uint  *d_voxelOccupied = 0;
uint  *d_voxelOccupiedScan = 0;
uint  *d_compactedVoxelArray;

// tables
uint *d_numVertsTable = 0;
uint *d_edgeTable = 0;
uint *d_triTable = 0;

//pre data
const double myPi = 3.14159265358979323846;
float myEyeAngle = 0;   
float myProjectionMatrix[16];
float translateMatrix[16], rotateMatrix[16],modelMatrix[16], viewMatrix[16],modelViewMatrix[16], modelViewProjMatrix[16];
//摄象机投影矩阵
double matrix1[12]={-814.181,-254.218,-301.903,393436,65.0634,440.248,-732.481,209546,0.091213,-0.732029,-0.67514,1273.78},
	   matrix2[12]={-481.117,767.8,-41.382,268432,243.437,-9.67565,-450.11,119411,-1.45133,0,0,908.513},
	   matrix3[12]={866.417,171.75,-189.775,206639,4.13581,-471.062,-744.419,259510,0.051583,0.689672,-0.722282,919.593},
	   matrix4[12]={165.335,883.286,-170.112,296392,838.353,59.3904,340.12,234738,0.718938,0.132446,-0.682339,979.866},
	   matrix5[12]={-841.698,-14.4216,-334.349,489542,8.54247,838.757,-263.983,332716,0.02968,-0.048834,-0.998366,1393.59};
//成像平面法向量
float mynormal1[3]={-0.0862021,0.762329,0.64142};
float mynormal2[3]={0.6427876,0.0,0.766044};
float mynormal3[3]={0.0,-0.6427876,0.766044};
float mynormal4[3]={-0.6427876,0.0,0.766044};
float mynormal5[3]={0.0,0.0,1.0};
//体素数据
struct voxelNode voxels[Depth*Depth*Depth];
struct projNode proj1[Depth*Depth*Depth],proj2[Depth*Depth*Depth],proj3[Depth*Depth*Depth],proj4[Depth*Depth*Depth],proj5[Depth*Depth*Depth];
//图像数据
GLuint tex[5]; 
unsigned char pixels1[512*640*3],pixels2[512*640*3],pixels3[512*640*3],pixels4[512*640*3],pixels5[512*640*3];
unsigned char m_two1[512*640],m_two2[512*640],m_two3[512*640],m_two4[512*640],m_two5[512*640];
//cg parameter
static CGcontext   myCgContext;
static CGprofile   myCgVertexProfile,
                   myCgFragmentProfile;
static CGprogram   myCgVertexProgram,
                   myCgFragmentProgram;
static CGparameter myCgVertexParam_modelViewProj,          //与uniform变量关联
                   myCgVertexParam_p1,
                   myCgVertexParam_p2,
                   myCgVertexParam_p3,
                   myCgVertexParam_p4,
                   myCgVertexParam_p5,
                   myCgFragmentParam_decal1,
                   myCgFragmentParam_decal2,
                   myCgFragmentParam_decal3,
                   myCgFragmentParam_decal4,
                   myCgFragmentParam_decal5,
                   myCgFragmentParam_normal1,
                   myCgFragmentParam_normal2,
                   myCgFragmentParam_normal3,
                   myCgFragmentParam_normal4,
                   myCgFragmentParam_normal5;
static const char  *myProgramName = "simpleGL",
                   *myVertexProgramFileName = "Vcg_TextureMap.cg",
				   *myVertexProgramName = "Vertex_TextureMap",
                   *myFragmentProgramFileName = "Fcg_TextureMap.cg",
				   *myFragmentProgramName = "Fragment_TextureMap";
// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float3 rotate = make_float3(0.0, 0.0, 0.0);
float3 translate= make_float3(0.0, 0.0, -3.0);

// toggles
bool wireframe = true;
bool render = true;
bool compute = true;

// forward declarations
void runTest(int argc, char** argv);

CUTBoolean initGL();
void createVBO(GLuint* vbo, unsigned int size);
void deleteVBO(GLuint* vbo);

void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void idle();
void reshape(int w, int h);
//内核调用函数
extern "C" void allocateTexturesRef(uint **d_edgeTable, uint **d_triTable,  uint **d_numVertsTable,uchar **dimage1,uchar **dimage2,uchar **dimage3,uchar **dimage4,uchar **dimage5);
extern "C" void parallelClassifyAllVoxels( dim3 grid, dim3 threads, uint* voxelVerts, uint *voxelOccupied, uint numVoxels,
									struct projNode *dproj1,struct projNode *dproj2,struct projNode *dproj3,struct projNode *dproj4,struct projNode *dproj5);            
extern "C" void parallelCompactVoxels(dim3 grid, dim3 threads, uint *compactedVoxelArray, uint *voxelOccupied, uint *voxelOccupiedScan, uint numVoxels);
extern "C" void parallelGenerateTriangles(dim3 grid, dim3 threads,float3 *pos,float3 *normal,float2 *tex1,float2 *tex2,float2 *tex3,float2 *tex4,float2 *tex5,uint *compactedVoxelArray, uint *numVertsScanned,
						uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask,float3 voxelSize,uint activeVoxels, uint maxVerts,
						struct projNode *dproj1,struct projNode *dproj2,struct projNode *dproj3,struct projNode *dproj4,struct projNode *dproj5);
void mainMenu(int i);

//矩阵求解函数
//计算透视投影矩阵
 void buildPerspectiveMatrix(double fieldOfView,double aspectRatio,double zNear, double zFar,float m[16])
{
  double sine, cotangent, deltaZ;
  double radians = fieldOfView / 2.0 * myPi / 180.0;
  
  deltaZ = zFar - zNear;
  sine = sin(radians);
  assert(deltaZ);
  assert(sine);
  assert(aspectRatio);
  cotangent = cos(radians) / sine;

  m[0*4+0] = cotangent / aspectRatio;
  m[0*4+1] = 0.0;
  m[0*4+2] = 0.0;
  m[0*4+3] = 0.0;
  
  m[1*4+0] = 0.0;
  m[1*4+1] = cotangent;
  m[1*4+2] = 0.0;
  m[1*4+3] = 0.0;
  
  m[2*4+0] = 0.0;
  m[2*4+1] = 0.0;
  m[2*4+2] = -(zFar + zNear) / deltaZ;
  m[2*4+3] = -2 * zNear * zFar / deltaZ;
  
  m[3*4+0] = 0.0;
  m[3*4+1] = 0.0;
  m[3*4+2] = -1;
  m[3*4+3] = 0;
}
 //计算视点变换矩阵
void buildLookAtMatrix(double eyex, double eyey, double eyez,double centerx, double centery, double centerz,double upx, double upy, double upz,float m[16])
{
  double x[3], y[3], z[3], mag;

  z[0] = eyex - centerx;
  z[1] = eyey - centery;
  z[2] = eyez - centerz;

  mag = sqrt(z[0]*z[0] + z[1]*z[1] + z[2]*z[2]);
  if (mag) {
    z[0] /= mag;
    z[1] /= mag;
    z[2] /= mag;
  }

  y[0] = upx;
  y[1] = upy;
  y[2] = upz;

  x[0] =  y[1]*z[2] - y[2]*z[1];
  x[1] = -y[0]*z[2] + y[2]*z[0];
  x[2] =  y[0]*z[1] - y[1]*z[0];

  y[0] =  z[1]*x[2] - z[2]*x[1];
  y[1] = -z[0]*x[2] + z[2]*x[0];
  y[2] =  z[0]*x[1] - z[1]*x[0];

  mag = sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
  if (mag) {
    x[0] /= mag;
    x[1] /= mag;
    x[2] /= mag;
  }

  mag = sqrt(y[0]*y[0] + y[1]*y[1] + y[2]*y[2]);
  if (mag) {
    y[0] /= mag;
    y[1] /= mag;
    y[2] /= mag;
  }

  m[0*4+0] = x[0];  m[0*4+1] = x[1];
  m[0*4+2] = x[2];  m[0*4+3] = -x[0]*eyex + -x[1]*eyey + -x[2]*eyez;

  m[1*4+0] = y[0];  m[1*4+1] = y[1];
  m[1*4+2] = y[2];  m[1*4+3] = -y[0]*eyex + -y[1]*eyey + -y[2]*eyez;

  m[2*4+0] = z[0];  m[2*4+1] = z[1];
  m[2*4+2] = z[2];  m[2*4+3] = -z[0]*eyex + -z[1]*eyey + -z[2]*eyez;

  m[3*4+0] = 0.0;   m[3*4+1] = 0.0;  m[3*4+2] = 0.0;  m[3*4+3] = 1.0;
}
//计算旋转矩阵
void makeRotateMatrix(float angle,float ax, float ay, float az,float m[16])
{
  float radians, sine, cosine, ab, bc, ca, tx, ty, tz;
  float axis[3];
  float mag;

  axis[0] = ax;
  axis[1] = ay;
  axis[2] = az;
  mag = sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]);
  if (mag) {
    axis[0] /= mag;
    axis[1] /= mag;
    axis[2] /= mag;
  }

  radians = angle * myPi / 180.0;
  sine = sin(radians);
  cosine = cos(radians);
  ab = axis[0] * axis[1] * (1 - cosine);
  bc = axis[1] * axis[2] * (1 - cosine);
  ca = axis[2] * axis[0] * (1 - cosine);
  tx = axis[0] * axis[0];
  ty = axis[1] * axis[1];
  tz = axis[2] * axis[2];

  m[0]  = tx + cosine * (1 - tx);
  m[1]  = ab + axis[2] * sine;
  m[2]  = ca - axis[1] * sine;
  m[3]  = 0.0f;
  m[4]  = ab - axis[2] * sine;
  m[5]  = ty + cosine * (1 - ty);
  m[6]  = bc + axis[0] * sine;
  m[7]  = 0.0f;
  m[8]  = ca + axis[1] * sine;
  m[9]  = bc - axis[0] * sine;
  m[10] = tz + cosine * (1 - tz);
  m[11] = 0;
  m[12] = 0;
  m[13] = 0;
  m[14] = 0;
  m[15] = 1;
}
//计算平移变换矩阵
 void makeTranslateMatrix(float x, float y, float z, float m[16])
{
  m[0]  = 1;  m[1]  = 0;  m[2]  = 0;  m[3]  = x;
  m[4]  = 0;  m[5]  = 1;  m[6]  = 0;  m[7]  = y;
  m[8]  = 0;  m[9]  = 0;  m[10] = 1;  m[11] = z;
  m[12] = 0;  m[13] = 0;  m[14] = 0;  m[15] = 1;
}

//计算两矩阵相乘
 void multMatrix(float dst[16],const float src1[16], const float src2[16])
{
  float tmp[16];
  int i, j;

  for (i=0; i<4; i++) {
    for (j=0; j<4; j++) {
      tmp[i*4+j] = src1[i*4+0] * src2[0*4+j] +
                   src1[i*4+1] * src2[1*4+j] +
                   src1[i*4+2] * src2[2*4+j] +
                   src1[i*4+3] * src2[3*4+j];
    }
  }
  for (i=0; i<16; i++)
    dst[i] = tmp[i];
}
 //矩阵和向量相乘
void multMatrixVector(float dst[3],const float srcM[12], const float srcV[4])
{
	for(int i=0;i<3;i++)
	{
		dst[i]=srcM[i*4+0] * srcV[0] +srcM[i*4+1] * srcV[1] +srcM[i*4+2] * srcV[2] +srcM[i*4+3] * srcV[3];
	}
}

////读取经过前景提取处理的bmp图像
void readBMPfile(char *Filename,unsigned char pixels[512*640*3])
{
	FILE *File=NULL;									
	AUX_RGBImageRec *pBitmap = NULL;
	if (!Filename)										
	{
		exit(0);									
	}
	File=fopen(Filename,"r");							
	if(File)											
	{
		fclose(File);									
		pBitmap = auxDIBImageLoad(Filename);			
	}				
	if(pBitmap == NULL)		
		exit(0);
	memcpy( pixels, pBitmap->data, 512*640*3 ); 				
}
//计算三维坐标点在成像平面上的投影
void projectWtoI(GLdouble objx,GLdouble objy,GLdouble objz,const GLdouble projMatrix[12],GLdouble *imagex,GLdouble *imagey,GLdouble *imagez)
{
	double x,y,z;
	double temx,temy,temz;
	x=projMatrix[0*4+0] * objx +projMatrix[0*4+1] * objy +projMatrix[0*4+2] * objz +projMatrix[0*4+3] * 1;
	y=projMatrix[1*4+0] * objx +projMatrix[1*4+1] * objy +projMatrix[1*4+2] * objz +projMatrix[1*4+3] * 1;
	z=projMatrix[2*4+0] * objx +projMatrix[2*4+1] * objy +projMatrix[2*4+2] * objz +projMatrix[2*4+3] * 1;
	temx=(double)x/(double)z;
	temy=(double)y/(double)z;
	*imagez=(double)z/(double)z;
	*imagex=temx;
	*imagey=511-temy;
}

// 初始化体素数据
void initVoxels()                                       
{
	int xMin=-300,xMax=300;
	int yMin=-300,yMax=300;
	int zMin=-300,zMax=300;
	double step=(double)(xMax-xMin)/(double)Depth;
	for(int k=0;k<Depth;k++)
	{
		for(int j=0;j<Depth;j++)
		{
			for (int i=0;i<Depth;i++)
			{
				voxels[k*Depth*Depth+j*Depth+i].xMin=xMin+i*step;
				voxels[k*Depth*Depth+j*Depth+i].xMax=xMin+(i+1)*step;
				voxels[k*Depth*Depth+j*Depth+i].yMin=yMin+j*step;
				voxels[k*Depth*Depth+j*Depth+i].yMax=yMin+(j+1)*step;
				voxels[k*Depth*Depth+j*Depth+i].zMin=zMin+k*step;
				voxels[k*Depth*Depth+j*Depth+i].zMax=zMin+(k+1)*step;
			}
		}
	}
	
}
//计算体素的投影
void Projection(GLdouble *M,struct voxelNode *pV,struct projNode *pJ)                
{	
	GLdouble winX[8], winY[8], winZ[8];
	for(int i=0;i<Depth*Depth*Depth;i++)
	{
		projectWtoI(pV[i].xMin, pV[i].yMin,pV[i].zMin, M,&winX[0],&winY[0],&winZ[0]);
		projectWtoI(pV[i].xMax, pV[i].yMin,pV[i].zMin, M,&winX[1],&winY[1],&winZ[1]);
		projectWtoI(pV[i].xMax, pV[i].yMax,pV[i].zMin, M,&winX[2],&winY[2],&winZ[2]);
		projectWtoI(pV[i].xMin, pV[i].yMax,pV[i].zMin, M,&winX[3],&winY[3],&winZ[3]);
		projectWtoI(pV[i].xMin, pV[i].yMin,pV[i].zMax, M,&winX[4],&winY[4],&winZ[4]);
		projectWtoI(pV[i].xMax, pV[i].yMin,pV[i].zMax, M,&winX[5],&winY[5],&winZ[5]);
		projectWtoI(pV[i].xMax, pV[i].yMax,pV[i].zMax, M,&winX[6],&winY[6],&winZ[6]);
		projectWtoI(pV[i].xMin, pV[i].yMax,pV[i].zMax, M,&winX[7],&winY[7],&winZ[7]);

		pJ[i].x[0]=winX[0],pJ[i].y[0]=winY[0];
		pJ[i].x[1]=winX[1],pJ[i].y[1]=winY[1];
		pJ[i].x[2]=winX[2],pJ[i].y[2]=winY[2];
		pJ[i].x[3]=winX[3],pJ[i].y[3]=winY[3];
		pJ[i].x[4]=winX[4],pJ[i].y[4]=winY[4];
		pJ[i].x[5]=winX[5],pJ[i].y[5]=winY[5];
		pJ[i].x[6]=winX[6],pJ[i].y[6]=winY[6];
		pJ[i].x[7]=winX[7],pJ[i].y[7]=winY[7];
	}
}
//对图像进行二值化
void imageTwo(unsigned char *a,unsigned char b[512*640])
{	
	int index;
	for(int i=0;i<512;i++)
	for(int j=0;j<640;j++)
	{
		index = i*640*3+j*3;
		if(a[index+0]==0&&a[index+1]==0&&a[index+2]==255)
			b[i*640+j]=0;
		else
			b[i*640+j]=1;
	}
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
    runTest(argc, argv);

    CUT_EXIT(argc, argv);
}

//////初始化
void initImageBasedRec()
{
    gridSize = make_uint3(1<<gridSizeLog2.x, 1<<gridSizeLog2.y, 1<<gridSizeLog2.z);
    gridSizeMask = make_uint3(gridSize.x-1, gridSize.y-1, gridSize.z-1);             
    gridSizeShift = make_uint3(0, gridSizeLog2.x, gridSizeLog2.x+gridSizeLog2.y);     
    printf("gridSizeMask: %d , %d , %d \n", gridSizeMask.x, gridSizeMask.y, gridSizeMask.z);
	printf("gridSizeShift: %d , %d , %d \n", gridSizeShift.x, gridSizeShift.y, gridSizeShift.z);

    numVoxels = gridSize.x*gridSize.y*gridSize.z;
    voxelSize = make_float3(600.0f / gridSize.x, 600.0f / gridSize.y, 600.0f / gridSize.z);
    maxVerts = gridSize.x*gridSize.y*100;
    printf("grid: %d x %d x %d = %d voxels\n", gridSize.x, gridSize.y, gridSize.z, numVoxels);
    printf("max verts = %d\n", maxVerts);
    
    // create VBOs
    createVBO(&posVbo, maxVerts*sizeof(float)*3);
    CUDA_SAFE_CALL(cudaGLRegisterBufferObject(posVbo));
    createVBO(&normalVbo, maxVerts*sizeof(float)*3);
    CUDA_SAFE_CALL(cudaGLRegisterBufferObject(normalVbo));
    createVBO(&texVbo1, maxVerts*sizeof(float)*2);
    CUDA_SAFE_CALL(cudaGLRegisterBufferObject(texVbo1));
    createVBO(&texVbo2, maxVerts*sizeof(float)*2);
    CUDA_SAFE_CALL(cudaGLRegisterBufferObject(texVbo2));
    createVBO(&texVbo3, maxVerts*sizeof(float)*2);
    CUDA_SAFE_CALL(cudaGLRegisterBufferObject(texVbo3));
    createVBO(&texVbo4, maxVerts*sizeof(float)*2);
    CUDA_SAFE_CALL(cudaGLRegisterBufferObject(texVbo4));
    createVBO(&texVbo5, maxVerts*sizeof(float)*2);
    CUDA_SAFE_CALL(cudaGLRegisterBufferObject(texVbo5));

	allocateTexturesRef(&d_edgeTable, &d_triTable, &d_numVertsTable,&dimage1,&dimage2,&dimage3,&dimage4,&dimage5);

    // 分配设备内存
    unsigned int memSize = sizeof(uint) * numVoxels;
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_voxelVerts, memSize));
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_voxelVertsScan, memSize));
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_voxelOccupied, memSize));
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_voxelOccupiedScan, memSize));
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_compactedVoxelArray, memSize));
    	
    CUDA_SAFE_CALL(cudaMalloc((void**)&dproj1, sizeof(struct projNode) * DATA_SIZE));
    CUDA_SAFE_CALL(cudaMalloc((void**)&dproj2, sizeof(struct projNode) * DATA_SIZE));
    CUDA_SAFE_CALL(cudaMalloc((void**)&dproj3, sizeof(struct projNode) * DATA_SIZE));
    CUDA_SAFE_CALL(cudaMalloc((void**)&dproj4, sizeof(struct projNode) * DATA_SIZE));
    CUDA_SAFE_CALL(cudaMalloc((void**)&dproj5, sizeof(struct projNode) * DATA_SIZE));

	//拷贝投影信息到设备内存
    cudaMemcpy(dproj1,proj1, sizeof(struct projNode) * DATA_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(dproj2,proj2, sizeof(struct projNode) * DATA_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(dproj3,proj3, sizeof(struct projNode) * DATA_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(dproj4,proj4, sizeof(struct projNode) * DATA_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(dproj5,proj5, sizeof(struct projNode) * DATA_SIZE, cudaMemcpyHostToDevice);
	
    // initialize CUDPP scan
    CUDPPConfiguration config;
    config.algorithm = CUDPP_SCAN;
    config.datatype = CUDPP_UINT;
    config.op = CUDPP_ADD;
    config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;
    cudppPlan(&scanplan, config, numVoxels, 1, 0);
}

//释放内存
void cleanup()
{
    deleteVBO(&posVbo);
    deleteVBO(&normalVbo);
    deleteVBO(&texVbo1);
    deleteVBO(&texVbo2);
    deleteVBO(&texVbo3);
    deleteVBO(&texVbo4);
    deleteVBO(&texVbo5);

    CUDA_SAFE_CALL(cudaFree(d_edgeTable));
    CUDA_SAFE_CALL(cudaFree(d_triTable));
    CUDA_SAFE_CALL(cudaFree(d_numVertsTable));
    CUDA_SAFE_CALL(cudaFree(d_voxelVerts));
    CUDA_SAFE_CALL(cudaFree(d_voxelVertsScan));
    CUDA_SAFE_CALL(cudaFree(d_voxelOccupied));
    CUDA_SAFE_CALL(cudaFree(d_voxelOccupiedScan));
    CUDA_SAFE_CALL(cudaFree(d_compactedVoxelArray));
    
    CUDA_SAFE_CALL(cudaFree(dproj1));
	CUDA_SAFE_CALL(cudaFree(dproj2));
	CUDA_SAFE_CALL(cudaFree(dproj3));
	CUDA_SAFE_CALL(cudaFree(dproj4));
	CUDA_SAFE_CALL(cudaFree(dproj5));
	CUDA_SAFE_CALL(cudaFree(dimage1));
	CUDA_SAFE_CALL(cudaFree(dimage2));
	CUDA_SAFE_CALL(cudaFree(dimage3));
	CUDA_SAFE_CALL(cudaFree(dimage4));
	CUDA_SAFE_CALL(cudaFree(dimage5));

    cudppDestroyPlan(scanplan);
}

void initMenus()
{
    glutCreateMenu(mainMenu);
    glutAddMenuEntry("Toggle rendering [r]", 'r');
    glutAddMenuEntry("Toggle wireframe [w]", 'w');
    glutAddMenuEntry("Quit (esc)", 'q');
    glutAttachMenu(GLUT_RIGHT_BUTTON);
}

void runTest(int argc, char** argv)
{
    CUT_DEVICE_INIT(argc, argv);

    // Create GL context
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(640, 640);
    glutCreateWindow("Image Based 3D Reconstruction");

    // initialize GL
    if(CUTFalse == initGL()) 
    {
        return;
    }
	//纹理对象
	glPixelStorei(GL_UNPACK_ALIGNMENT,1);
	glGenTextures(5,tex);
	glTexEnvf(GL_TEXTURE_ENV,GL_TEXTURE_ENV_MODE,GL_REPLACE);

	initVoxels();
	//读取图像数据
	readBMPfile("data/1.bmp", pixels1);
	readBMPfile("data/2.bmp", pixels2);
	readBMPfile("data/3.bmp", pixels3);
	readBMPfile("data/4.bmp", pixels4);
	readBMPfile("data/5.bmp", pixels5);
	//计算投影
    Projection(matrix1,voxels,proj1); 
    Projection(matrix2,voxels,proj2); 
    Projection(matrix3,voxels,proj3); 
    Projection(matrix4,voxels,proj4); 
    Projection(matrix5,voxels,proj5);

	initImageBasedRec();
    // 注册回调
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutReshapeFunc(reshape);
    initMenus();
    // cg context设置及顶点和片段程序的建立和加载
	myCgContext = cgCreateContext();
    cgGLSetDebugMode(CG_FALSE);
    cgSetParameterSettingMode(myCgContext, CG_DEFERRED_PARAMETER_SETTING);
    
    myCgVertexProfile = cgGLGetLatestProfile(CG_GL_VERTEX);
    cgGLSetOptimalOptions(myCgVertexProfile);
    myCgVertexProgram =cgCreateProgramFromFile(
                                myCgContext,              
							    CG_SOURCE,                
								myVertexProgramFileName,  
								myCgVertexProfile,        
								myVertexProgramName,     
								NULL);                    
    cgGLLoadProgram(myCgVertexProgram); 
    
    #define GET_VERTEX_PARAM(name)  myCgVertexParam_##name =cgGetNamedParameter(myCgVertexProgram, #name)
    
    GET_VERTEX_PARAM(modelViewProj);
 
    myCgFragmentProfile = cgGLGetLatestProfile(CG_GL_FRAGMENT);
    cgGLSetOptimalOptions(myCgFragmentProfile);
    myCgFragmentProgram =cgCreateProgramFromFile(
													myCgContext,              
													CG_SOURCE,                
													myFragmentProgramFileName,
													myCgFragmentProfile,      
													myFragmentProgramName,    
													NULL); 
    cgGLLoadProgram(myCgFragmentProgram);
    
    #define GET_FRAGMENT_PARAM(name) \
      myCgFragmentParam_##name = \
      cgGetNamedParameter(myCgFragmentProgram, #name); 
      
    GET_FRAGMENT_PARAM(decal1);
    GET_FRAGMENT_PARAM(decal2);
    GET_FRAGMENT_PARAM(decal3);
    GET_FRAGMENT_PARAM(decal4);
    GET_FRAGMENT_PARAM(decal5);
    
    GET_FRAGMENT_PARAM(normal1);
    GET_FRAGMENT_PARAM(normal2);
    GET_FRAGMENT_PARAM(normal3);
    GET_FRAGMENT_PARAM(normal4);
    GET_FRAGMENT_PARAM(normal5);

    //主循环
    glutMainLoop();
}

void modelReconstruction()
{
	//设定grid，block的维度
    int threads = 128;
    dim3 grid(numVoxels / threads, 1, 1);
    if (grid.x > 65535) {
        grid.y = grid.x / 32768;
        grid.x = 32768;
    }
    // 对体素分类
	parallelClassifyAllVoxels(grid, threads, d_voxelVerts, d_voxelOccupied, numVoxels,dproj1,dproj2,dproj3,dproj4,dproj5);

	#if SKIP_EMPTY_VOXELS

    // 利用cudpp库进行并行求和
		cudppScan(scanplan, d_voxelOccupiedScan, d_voxelOccupied, numVoxels);

    // 把非空体素数拷回主存
    {
        uint lastElement, lastScanElement;
        CUDA_SAFE_CALL(cudaMemcpy((void *) &lastElement, (void *) (d_voxelOccupied + numVoxels-1), sizeof(uint), cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy((void *) &lastScanElement, (void *) (d_voxelOccupiedScan + numVoxels-1), sizeof(uint), cudaMemcpyDeviceToHost));
        activeVoxels = lastElement + lastScanElement;
    }

    if (activeVoxels==0) {
        totalVerts = 0;
        return;
    }
      
    // 并行压缩
    parallelCompactVoxels(grid, threads, d_compactedVoxelArray, d_voxelOccupied, d_voxelOccupiedScan, numVoxels);
    
	#endif // SKIP_EMPTY_VOXELS

    //并行求和顶点数
    cudppScan(scanplan, d_voxelVertsScan, d_voxelVerts, numVoxels);

    // 把顶点数拷回主存
    {
        uint lastElement, lastScanElement;
        CUDA_SAFE_CALL(cudaMemcpy((void *) &lastElement,(void *) (d_voxelVerts + numVoxels-1),sizeof(uint), cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy((void *) &lastScanElement,(void *) (d_voxelVertsScan + numVoxels-1),sizeof(uint), cudaMemcpyDeviceToHost));
        totalVerts = lastElement + lastScanElement;
    }

    // 计算顶点坐标，法线及顶点纹理坐标，存储到缓存对象
    float3 *d_pos = 0;
    float3 *d_normal=0;
	float2 *d_tex1=0;
	float2 *d_tex2=0;
	float2 *d_tex3=0;
	float2 *d_tex4=0;
	float2 *d_tex5=0;
	CUDA_SAFE_CALL(cudaGLMapBufferObject((void**)&d_normal, normalVbo));
    CUDA_SAFE_CALL(cudaGLMapBufferObject((void**)&d_pos, posVbo));
    CUDA_SAFE_CALL(cudaGLMapBufferObject((void**)&d_tex1, texVbo1));
    CUDA_SAFE_CALL(cudaGLMapBufferObject((void**)&d_tex2, texVbo2));
    CUDA_SAFE_CALL(cudaGLMapBufferObject((void**)&d_tex3, texVbo3));
    CUDA_SAFE_CALL(cudaGLMapBufferObject((void**)&d_tex4, texVbo4));
    CUDA_SAFE_CALL(cudaGLMapBufferObject((void**)&d_tex5, texVbo5));

	#if SKIP_EMPTY_VOXELS
		dim3 grid2((int) ceil(activeVoxels / (float) NTHREADS), 1, 1);
	#else
		dim3 grid2((int) ceil(numVoxels / (float) NTHREADS), 1, 1);
	#endif
    while(grid2.x > 65535) {
        grid2.x/=2;
        grid2.y*=2;
    }
	parallelGenerateTriangles(grid2, NTHREADS,d_pos,d_normal,d_tex1,d_tex2,d_tex3,d_tex4,d_tex5, d_compactedVoxelArray, d_voxelVertsScan, gridSize, gridSizeShift,gridSizeMask,voxelSize,activeVoxels, maxVerts,dproj1,dproj2,dproj3,dproj4,dproj5);
   
	//解除映射
	CUDA_SAFE_CALL(cudaGLUnmapBufferObject(texVbo1));
    CUDA_SAFE_CALL(cudaGLUnmapBufferObject(texVbo2));
    CUDA_SAFE_CALL(cudaGLUnmapBufferObject(texVbo3));
    CUDA_SAFE_CALL(cudaGLUnmapBufferObject(texVbo4));
    CUDA_SAFE_CALL(cudaGLUnmapBufferObject(texVbo5));
    CUDA_SAFE_CALL(cudaGLUnmapBufferObject(normalVbo));
    CUDA_SAFE_CALL(cudaGLUnmapBufferObject(posVbo));
}




////////////////////////////////////////////////////////////////////////////////
//! Initialize OpenGL
////////////////////////////////////////////////////////////////////////////////
CUTBoolean initGL()
{
    // initialize necessary OpenGL extensions
    glewInit();
    if (! glewIsSupported("GL_VERSION_2_0 " 
		                  )) {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        return CUTFalse;
    }

    glDisable(GL_DEPTH_TEST); 
    glutReportErrors();
    return CUTTrue;
}

// Create VBO
void createVBO(GLuint* vbo, unsigned int size)
{
    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glutReportErrors();
}

// Delete VBO
void deleteVBO(GLuint* vbo)
{
    glBindBuffer(1, *vbo);
    glDeleteBuffers(1, vbo);
    CUDA_SAFE_CALL(cudaGLUnregisterBufferObject(*vbo));
    *vbo = 0;
}

// 绘制重建的模型
void renderReconstructedModel()
{
    glBindBuffer(GL_ARRAY_BUFFER, posVbo);
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_FLOAT, 0, 0);
    
    glBindBufferARB(GL_ARRAY_BUFFER_ARB, normalVbo);
    glEnableClientState(GL_NORMAL_ARRAY);
    glNormalPointer(GL_FLOAT, sizeof(float)*3, 0);


	glClientActiveTexture(GL_TEXTURE0);
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, texVbo1);
    glTexCoordPointer(2,GL_FLOAT, 0,0);
    
    glClientActiveTexture(GL_TEXTURE1);
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, texVbo2);
    glTexCoordPointer(2,GL_FLOAT, 0,0);
    
    glClientActiveTexture(GL_TEXTURE2);
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, texVbo3);
    glTexCoordPointer(2,GL_FLOAT, 0,0);
    
    glClientActiveTexture(GL_TEXTURE3);
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, texVbo4);
    glTexCoordPointer(2,GL_FLOAT, 0,0);
    
    glClientActiveTexture(GL_TEXTURE4);
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, texVbo5);
    glTexCoordPointer(2,GL_FLOAT, 0,0);
   
    glClientActiveTexture(GL_TEXTURE5);

    glColor3f(1.0, 0.0, 0.0);
    glDrawArrays(GL_TRIANGLES, 0, totalVerts);
    
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);
    glClientActiveTexture(GL_TEXTURE0);
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    glClientActiveTexture(GL_TEXTURE1);
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    glClientActiveTexture(GL_TEXTURE2);
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    glClientActiveTexture(GL_TEXTURE3);
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    glClientActiveTexture(GL_TEXTURE4);
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}


//Display callback
void display()
{	 
	unsigned int timer;
	CUT_SAFE_CALL(  cutCreateTimer(&timer)  );
    CUT_SAFE_CALL(  cutResetTimer(timer)    );
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
    CUT_SAFE_CALL( cutStartTimer(timer) );

	//二值化图像
    imageTwo(pixels1, m_two1);     
    imageTwo(pixels2, m_two2);      
    imageTwo(pixels3, m_two3);    
    imageTwo(pixels4, m_two4);   
    imageTwo(pixels5, m_two5);

	//启用多重纹理映射
    glActiveTexture(GL_TEXTURE0);
    glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D,tex[0]);
	glPixelStorei(GL_UNPACK_ALIGNMENT,1);
	glTexImage2D(GL_TEXTURE_2D,0,GL_RGB,640,512,0,GL_RGB,GL_UNSIGNED_BYTE,pixels1);
//	gluBuild2DMipmaps(GL_TEXTURE_2D, GL_RGB8,640, 512, GL_RGB, GL_UNSIGNED_BYTE, pixels1);
	glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP);
    glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP);
	glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
	
	glActiveTexture(GL_TEXTURE1);
    glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D,tex[1]);
	glPixelStorei(GL_UNPACK_ALIGNMENT,1);
	glTexImage2D(GL_TEXTURE_2D,0,GL_RGB,640,512,0,GL_RGB,GL_UNSIGNED_BYTE,pixels2);
	glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP);
    glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP);
	glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
	
	glActiveTexture(GL_TEXTURE2);
    glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D,tex[2]);
	glPixelStorei(GL_UNPACK_ALIGNMENT,1);
	glTexImage2D(GL_TEXTURE_2D,0,GL_RGB,640,512,0,GL_RGB,GL_UNSIGNED_BYTE,pixels3);
	glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP);
    glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP);
	glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
	
	glActiveTexture(GL_TEXTURE3);
    glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D,tex[3]);
	glPixelStorei(GL_UNPACK_ALIGNMENT,1);
	glTexImage2D(GL_TEXTURE_2D,0,GL_RGB,640,512,0,GL_RGB,GL_UNSIGNED_BYTE,pixels4);
	glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP);
    glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP);
	glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
	
	glActiveTexture(GL_TEXTURE4);
    glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D,tex[4]);
	glPixelStorei(GL_UNPACK_ALIGNMENT,1);
	glTexImage2D(GL_TEXTURE_2D,0,GL_RGB,640,512,0,GL_RGB,GL_UNSIGNED_BYTE,pixels5);
	
//	CUT_SAFE_CALL(cutStopTimer(timer));
//	printf("Time spent by Construction %.2f\n", cutGetTimerValue(timer) );
	glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP);
    glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP);
	glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);

//	CUT_SAFE_CALL(cutStopTimer(timer));
//	printf("Time spent by Construction %.2f\n", cutGetTimerValue(timer) );

	//cg shader   
	cgGLBindProgram(myCgVertexProgram);
    cgGLEnableProfile(myCgVertexProfile);
    double aspectRatio = 1.0;
    double fieldOfView = 60.0;
    buildPerspectiveMatrix(fieldOfView, aspectRatio,
                           0.1, 2000.0, 
                           myProjectionMatrix);
	buildLookAtMatrix(  -43.10105,581.164,520.71,   // eye position 
					   0, 0, 0,                     // view center
                       0, 0, 1,		               // up vector 
                       viewMatrix);
    makeRotateMatrix(0, 1, 1, 1, rotateMatrix);
    makeTranslateMatrix(0, 0, 0, translateMatrix);
    multMatrix(modelMatrix, translateMatrix, rotateMatrix);
    multMatrix(modelViewMatrix, viewMatrix, modelMatrix);
    multMatrix(modelViewProjMatrix, myProjectionMatrix, modelViewMatrix);
    cgSetMatrixParameterfr(myCgVertexParam_modelViewProj, modelViewProjMatrix);
    cgUpdateProgramParameters(myCgVertexProgram);
    
	cgGLBindProgram(myCgFragmentProgram);
    cgGLEnableProfile(myCgFragmentProfile); 
    cgGLSetTextureParameter(myCgFragmentParam_decal1, tex[0]);
    cgGLSetTextureParameter(myCgFragmentParam_decal2, tex[1]);
    cgGLSetTextureParameter(myCgFragmentParam_decal3, tex[2]);
    cgGLSetTextureParameter(myCgFragmentParam_decal4, tex[3]);
    cgGLSetTextureParameter(myCgFragmentParam_decal5, tex[4]);
    cgSetParameter3fv(myCgFragmentParam_normal1, mynormal1);
    cgSetParameter3fv(myCgFragmentParam_normal2, mynormal2);
    cgSetParameter3fv(myCgFragmentParam_normal3, mynormal3);
    cgSetParameter3fv(myCgFragmentParam_normal4, mynormal4);
    cgSetParameter3fv(myCgFragmentParam_normal5, mynormal5);
    cgUpdateProgramParameters(myCgFragmentProgram);
    cgGLEnableTextureParameter(myCgFragmentParam_decal1);
    cgGLEnableTextureParameter(myCgFragmentParam_decal2);
    cgGLEnableTextureParameter(myCgFragmentParam_decal3);
    cgGLEnableTextureParameter(myCgFragmentParam_decal4);
    cgGLEnableTextureParameter(myCgFragmentParam_decal5);

	//把图像的二值化信息拷贝到设备内存
	cudaMemcpy(dimage1,m_two1, sizeof(unsigned char) * 640*512, cudaMemcpyHostToDevice);
    cudaMemcpy(dimage2,m_two2, sizeof(unsigned char) * 640*512, cudaMemcpyHostToDevice);
    cudaMemcpy(dimage3,m_two3, sizeof(unsigned char) * 640*512, cudaMemcpyHostToDevice);
    cudaMemcpy(dimage4,m_two4, sizeof(unsigned char) * 640*512, cudaMemcpyHostToDevice);
    cudaMemcpy(dimage5,m_two5, sizeof(unsigned char) * 640*512, cudaMemcpyHostToDevice);

	//启动内核在GPU中并行三维重建
    modelReconstruction();
	
    glClearColor(0.0, 0.0, 1.0, 0.0); 
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);  
    glPolygonMode(GL_FRONT_AND_BACK, wireframe? GL_LINE : GL_FILL);

	//绘制重建的模型
    glPushMatrix();
    renderReconstructedModel();
    glPopMatrix();
	
	//关闭多重纹理映射功能
	glActiveTexture(GL_TEXTURE4);
	glDisable(GL_TEXTURE_2D);
	glActiveTexture(GL_TEXTURE3);
	glDisable(GL_TEXTURE_2D); 
	glActiveTexture(GL_TEXTURE2);
	glDisable(GL_TEXTURE_2D);
	glActiveTexture(GL_TEXTURE1);
	glDisable(GL_TEXTURE_2D); 
	glActiveTexture(GL_TEXTURE0);
	glDisable(GL_TEXTURE_2D); 
	
	CUT_SAFE_CALL(cutStopTimer(timer));
	printf("Time spent by Construction %.2f\n",( cutGetTimerValue(timer)-20.0) );
    glutSwapBuffers();
}

// Keyboard events callback

void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch(key) {
    case(27) :
        cleanup();
        cudaThreadExit();
        exit(0);
 
    case 'w':
        wireframe = !wireframe;
        break;
    case 'r':
        render = !render;
        break;
    case 'c':
        compute = !compute;
        break;
    }

    if (!compute) {
        modelReconstruction();        
    }

    glutPostRedisplay();
}

// Mouse event callback
void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN) {
        mouse_buttons |= 1<<button;
    } else if (state == GLUT_UP) {
        mouse_buttons = 0;
    }

    mouse_old_x = x;
    mouse_old_y = y;
    glutPostRedisplay();
}

void motion(int x, int y)
{
    float dx = x - mouse_old_x;
    float dy = y - mouse_old_y;

    if (mouse_buttons==1) {
        rotate.x += dy * 0.2;
        rotate.y += dx * 0.2;
    } else if (mouse_buttons==2) {
        translate.x += dx * 0.01;
        translate.y -= dy * 0.01;
    } else if (mouse_buttons==3) {
        translate.z += dy * 0.01;
    }

    mouse_old_x = x;
    mouse_old_y = y;
    glutPostRedisplay();
}

void reshape(int w, int h)
{
    glMatrixMode(GL_MODELVIEW);
    glViewport(0, 0, w, h);
}

void mainMenu(int i)
{
    keyboard((unsigned char) i, 0, 0);
}
