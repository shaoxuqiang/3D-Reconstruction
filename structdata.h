#ifndef STRUCDATA
#define STRUCDATA
struct voxelNode
{	
	double xMin,xMax;
	double yMin,yMax;
	double zMin,zMax;
};
struct projNode
{	
	short x[8];
	short y[8];
};

#endif