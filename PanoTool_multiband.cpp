#include <d3d11.h>
#include <XnaMath.h> 
#include <stdio.h> 
#include <Eigen/core> 
#include <Eigen/Geometry>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace Eigen;

const double PI = 3.141592653;
//弧度角度转换
const double RADIAN_TO_ANGLE = 180.0 / PI;
const double ANGLE_TO_RADIAN = PI / 180.0;

const int PANOWIDTH = 4096;//1600;//1920;
const int PANOHEIGHT = 2048;//800;//960;

const int MAXCAMERANUM = 12;
const int BLENDSIZE = 50;
const int BLENDSIZE_F = 250;

struct OLine
{
	int lensType;//0-rectiLinear,2-fisheye circular, 3- fisheye fullFrame
	double yaw, pitch, roll;
	double hFov;
	double a, b, c;//distortion
	double d,e;// initial lens offset in pixels, d - horizontal offset,e - vertical offset
	int crop,cl,cr,ct,cb;//crop
	char fileName[64];
};

//裁剪
void cropImage(cv::Mat image, int l, int r, int t, int b, cv::Mat &destImage, cv::Mat &destMask)
{
	int width = image.cols;
	int height = image.rows;
	
	int destWidth = r - l, destHeight = b - t;
	destImage = cv::Mat(destHeight, destWidth, CV_8UC3, cv::Scalar(0));
	cv::Mat destImageMask(destHeight, destWidth, CV_16UC1, cv::Scalar(0));
	destMask = destImageMask;

	int begX = l < 0 ? 0 : l, endX = r < width ? r : width, begY = t < 0 ? 0 : t, endY = b < height ? b : height;
	int copySize = (endX - begX) * 3;
	int destRowOffset = 0, destColOffset = 0;
	if(t < 0) destRowOffset = -t;
	if(l < 0) destColOffset = -l;

	unsigned char* src = image.data;//BGR
	unsigned char* dest = destImage.data;
	unsigned short* mask = (unsigned short*)destMask.data;
	for (int i = begY; i < endY; i++)
	{
		int destOffset = (i + destRowOffset) * destWidth * 3 + destColOffset * 3;
		int srcOffset = i * width * 3 + begX * 3;
		memcpy(dest + destOffset, src + srcOffset, copySize);
		memset(mask + (i + destRowOffset) * destWidth + destColOffset, 255, (endX - begX) * sizeof(short));
	}
}

//http://wiki.panotools.org/Lens_correction_model
void calibrateImage(cv::Mat &image, cv::Mat *imageMask, double a, double b, double c, cv::Mat &destImage, cv::Mat &destMask)
{
	int width = image.cols;
	int height = image.rows;
	cv::Mat _destImage(height, width, CV_8UC3);
	destImage = _destImage;
	cv::Mat _destImageMask(height, width, CV_8UC1);
	destMask = _destImageMask;

	unsigned char* dest = destImage.data;
	unsigned char* src = image.data;
	unsigned char* srcMaskData = imageMask ? imageMask->data : 0;
	unsigned char* maskData = destMask.data;

	double yLen = height / 2.0;
	double xLen = width / 2.0;
	double rLen = yLen < xLen ? yLen : xLen;
	double d = 1 - (a + b + c);
	for (int i = 0; i < height; i++)
	{
		double yn = (i - yLen) / rLen;
		for (int j = 0; j < width; j++)
		{
			double xn = (j - xLen) / rLen;
			double rDest = sqrt(xn * xn + yn * yn);
			double rSrc = (a * rDest * rDest * rDest + b  * rDest * rDest + c * rDest + d) * rDest;
			double times = rSrc / rDest;

			int x = (xn * times) * rLen + xLen;
			int y = (yn * times) * rLen + yLen;

			int maskIndex = i * width + j;
			if (x >= 0 && x < width && y >=0 && y < height)
			{
				int srcMaskIndex = y * width + x;
				if (!srcMaskData || srcMaskData[srcMaskIndex] == 255)
				{
					int srcIndex = (y * width + x) * 3;
					int destIndex = (i * width + j) * 3;
					dest[destIndex] = src[srcIndex];
					dest[destIndex + 1] = src[srcIndex + 1];
					dest[destIndex + 2] = src[srcIndex + 2];
					maskData[maskIndex] = 255;
				}
				else
					maskData[maskIndex] = 0;
			}
			else
				maskData[maskIndex] = 0;
		}
	}
}

bool UndistToDist(double x, double y, double xLen, double yLen, 
	double cx, double cy, double a, double b, double c, double &destX, double &destY)
{
	double rLen = yLen < xLen ? yLen : xLen;
	double yn = (y - yLen) / rLen;
	double xn = (x - xLen) / rLen;

	double rDest = sqrt(xn * xn + yn * yn);
	double d = (1.0 - a - b - c);
	double rSrc = (a * rDest * rDest * rDest + b  * rDest * rDest + c * rDest + d) * rDest;
	double times = rSrc / rDest;

	destX = (xn * times) * rLen + xLen;
	destY = (yn * times) * rLen + yLen;

	return true;
}


//坐标转换
void pano2FishEye(int x, int y, int srcWidth, int srcHeight, OLine &ol, XMMATRIX &mat, double &u, double &v)
{
	double hFov = ol.hFov;

	double hAngle = PI * 2.0 * x / PANOWIDTH;
	double vAngle = PI * y / PANOHEIGHT;

	double cz = cos(vAngle);
	double cxy = sin(vAngle);
	double cx = -cxy * sin(hAngle);
	double cy = cxy * cos(hAngle);

	//转成向量
	XMVECTOR vec = XMVectorSet(cx, cy, cz, 1);
	//向量乘以矩阵进行旋转
	XMVECTOR rvec = XMVector4Transform(vec, mat);
	XMFLOAT4 pos;
	//取出向量值
	XMStoreFloat4(&pos, rvec);

	if(pos.y < 0.001)
	{
		u = v = 0.0;
		return;
	}

	double len = sqrt(pos.x * pos.x + pos.z * pos.z);
	double times = (PI / 2 - acos(len)) / len;

	double vFov = hFov * srcHeight / srcWidth;
	hFov = hFov * ANGLE_TO_RADIAN;
	vFov = vFov * ANGLE_TO_RADIAN;
	u = 1.0 - (pos.x * times  + hFov / 2) / hFov;
	v = 1.0 - (pos.z * times  + vFov / 2) / vFov;

	double ux = srcWidth * u;
	double vy = srcHeight * v;
	double destX, destY;
	UndistToDist(ux + ol.d, vy + ol.e, srcWidth / 2, srcHeight / 2, 0, 0, ol.a, ol.b, ol.c, destX, destY);
	u = destX / srcWidth;
	v = destY / srcHeight;
}

void pano2FishEye2(int x, int y, int srcWidth, int srcHeight, OLine &ol, Matrix3d &mat, double &u, double &v)
{
	double hFov = ol.hFov;

	double hAngle = PI * 2.0 * x / PANOWIDTH;
	double vAngle = PI * y / PANOHEIGHT;

	double cz = cos(vAngle);
	double cxy = sin(vAngle);
	double cx = -cxy * sin(hAngle);
	double cy = cxy * cos(hAngle);

	//转成向量
	//XMVECTOR vec = XMVectorSet(cx, cy, cz, 1);
	Vector3d vec(cx, cy, cz);
	//向量乘以矩阵进行旋转
	//XMVECTOR rvec = XMVector4Transform(vec, mat);
	Vector3d rvec = mat * vec;// * mat;
	XMFLOAT4 pos(rvec.x(), rvec.y(), rvec.z(), 1);
	////取出向量值
	//XMStoreFloat4(&pos, rvec);

	if(pos.y < 0.001)
	{
		u = v = 0.0;
		return;
	}

	double len = sqrt(pos.x * pos.x + pos.z * pos.z);
	double times = (PI / 2 - acos(len)) / len;

	double vFov = hFov * srcHeight / srcWidth;
	hFov = hFov * ANGLE_TO_RADIAN;
	vFov = vFov * ANGLE_TO_RADIAN;
	u = 1.0 - (pos.x * times  + hFov / 2) / hFov;
	v = 1.0 - (pos.z * times  + vFov / 2) / vFov;

	double ux = srcWidth * u;
	double vy = srcHeight * v;
	double destX, destY;
	UndistToDist(ux + ol.d, vy + ol.e, srcWidth / 2, srcHeight / 2, 0, 0, ol.a, ol.b, ol.c, destX, destY);
	u = destX / srcWidth;
	v = destY / srcHeight;
}

//坐标转换
void pano2RectiLinear(int x, int y, int srcWidth, int srcHeight, OLine &ol, XMMATRIX &mat, double &u, double &v)
{
	double hFov = ol.hFov;

	double hAngle = PI * 2.0 * x / PANOWIDTH;
	double vAngle = PI * y / PANOHEIGHT;

	double cz = cos(vAngle);
	double cxy = sin(vAngle);
	double cx = -cxy * sin(hAngle);
	double cy = cxy * cos(hAngle);

	XMVECTOR vec = XMVectorSet(cx, cy, cz, 1);
	XMVECTOR rvec = XMVector4Transform(vec, mat);
	XMFLOAT4 pos;
	XMStoreFloat4(&pos, rvec);

	if(pos.y < 0.001)
	{
		u = v = 0.0;
		return;
	}

	double xLen = pos.x / pos.y;
	double zLen = pos.z / pos.y;
	hFov = hFov * ANGLE_TO_RADIAN;
	double uLen = tan(hFov / 2);
	double vLen = uLen * srcHeight / srcWidth;
	u = 1.0 - ((xLen / uLen) / 2 + 0.5);
	v = 1.0 - ((zLen / vLen) / 2 + 0.5);

	if (u < -0.1 || u > 1.1 || v < -0.1 || v > 1.1)
	{
		return ;
	}

	double ux = srcWidth * u;
	double vy = srcHeight * v;
	double destX, destY;
	UndistToDist(ux + ol.d, vy + ol.e, srcWidth / 2, srcHeight / 2, 0, 0, ol.a, ol.b, ol.c, destX, destY);
	u = destX / srcWidth;
	v = destY / srcHeight;
}

//取色
bool sampleColor(unsigned char* src, double u, double v, int width, int height, unsigned char* dest)
{
	if (u < 0.001 || u > 0.999 || v < 0.001 || v > 0.999)
	{
		return false;
	}
	
	int ux = width * u;
	int vy = height * v;
	int index = (vy * width + ux - width) * 3;
	if (index > 0)
	{
		dest[0] = src[index];
		dest[1] = src[index + 1];
		dest[2] = src[index + 2];
	}
	return true;
}

bool sampleColorBilinear(unsigned char* src, double u, double v, int width, int height, unsigned char* dest)
{
	if (u < 0.001 || u > 0.999 || v < 0.001 || v > 0.999)
	{
		return false;
	}

	double rx = width * u;
	double ry = height * v;

	int ux = width * u;
	int vy = height * v;
	int ux2 = ux + 1;
	int vy2 = vy + 1;
	double xWeight = 1.0 - (rx - ux);
	double yWeight = 1.0 - (ry - vy);


	int index = (vy * width + ux - width) * 3;
	if (index > 0)
	{
		if (ux2 < width && vy2 < height)
		{
			int indexRT = (vy * width + ux2 - width) * 3;
			int indexRB = (vy2 * width + ux2 - width) * 3;
			int indexLB = (vy2 * width + ux - width) * 3;

			dest[0] = src[index + 0] * xWeight * yWeight + src[indexRT + 0] * (1.0 - xWeight) * yWeight + src[indexRB + 0] * (1.0 - xWeight) * (1.0 - yWeight) + src[indexLB + 0] * xWeight * (1.0 - yWeight);
			dest[1] = src[index + 1] * xWeight * yWeight + src[indexRT + 1] * (1.0 - xWeight) * yWeight + src[indexRB + 1] * (1.0 - xWeight) * (1.0 - yWeight) + src[indexLB + 1] * xWeight * (1.0 - yWeight);
			dest[2] = src[index + 2] * xWeight * yWeight + src[indexRT + 2] * (1.0 - xWeight) * yWeight + src[indexRB + 2] * (1.0 - xWeight) * (1.0 - yWeight) + src[indexLB + 2] * xWeight * (1.0 - yWeight);
		}
		else
		{
			dest[0] = src[index];
			dest[1] = src[index + 1];
			dest[2] = src[index + 2];
		}
	}
	return true;
}

//取蒙版值，确定像素是否有效
unsigned short sampleMask(cv::Mat &mask, double u, double v, int width, int height)
{
	if (u < 0.001 || u > 0.999 || v < 0.001 || v > 0.999)
	{
		return 0;
	}

	int ux = width * u;
	int vy = height * v;
	int index = (vy * width + ux - width);
	if (index > 0)
	{
		unsigned short* src = (unsigned short*)mask.data;
		return src[index];
	}

	return 0;
}

//将旋转转成旋转矩阵
XMMATRIX transform(float yaw, float pitch, float roll)
{
	XMMATRIX pm = XMMatrixRotationX(-pitch * ANGLE_TO_RADIAN);//四元数转换
	XMMATRIX ym = XMMatrixRotationZ(-yaw * ANGLE_TO_RADIAN);
	XMMATRIX rm = XMMatrixRotationY(roll * ANGLE_TO_RADIAN);
	XMMATRIX rr = XMMatrixRotationZ(180 * ANGLE_TO_RADIAN);
	XMMATRIX mat = rr * ym * pm * rm;
	return mat;
}

Matrix3d transform2(float yaw, float pitch, float roll)
{
	Matrix3d pm = (Matrix3d)AngleAxisd(-pitch * ANGLE_TO_RADIAN, Vector3d::UnitX());//四元数转换
	Matrix3d ym = (Matrix3d)AngleAxisd(-yaw * ANGLE_TO_RADIAN, Vector3d::UnitZ());
	Matrix3d rm = (Matrix3d)AngleAxisd(roll * ANGLE_TO_RADIAN, Vector3d::UnitY());
	Matrix3d rr = (Matrix3d)AngleAxisd(180 * ANGLE_TO_RADIAN, Vector3d::UnitZ());
	Matrix3d mat = rm * pm * ym * rr;
	return mat;
}

//投影
void projectImage(cv::Mat& image, cv::Mat& imageMask, OLine &ol, cv::Mat &destImage, cv::Mat &destImageMask)
{
	double hFov = ol.hFov;
	XMMATRIX mat = transform(ol.yaw, ol.pitch, ol.roll);
	//Matrix3d mat2 = transform2(ol.yaw, ol.pitch, ol.roll);
	int width = image.cols;
	int height = image.rows;

	int panoWidth = destImage.cols;
	int panoHeight = destImage.rows;

	unsigned char* src = image.data;
	unsigned char* dest = destImage.data;
	unsigned short* maskData = (unsigned short*)destImageMask.data;
	unsigned char* destPtr;
	bool hasColor;
	unsigned short maskValue;
	double u,v;
	for (int i = 0; i < panoHeight; i++)
	{
		for (int j = 0; j < panoWidth; j++)
		{
			int index = i * panoWidth * 3 + j * 3;
			int maskIndex = i * panoWidth + j;

			if (ol.lensType == 0)
			{
				pano2RectiLinear(j, i, width, height, ol, mat, u, v);
			}
			else if (ol.lensType == 2 || ol.lensType == 3)
			{
				pano2FishEye(j, i, width, height, ol, mat, u, v);
				//pano2FishEye2(j, i, width, height, ol, mat2, u, v);
			}
				
			destPtr = dest + index;
			hasColor = sampleColorBilinear(src, u, v, width, height, destPtr);
			if (imageMask.data != 0)
			{
				maskValue = sampleMask(imageMask, u, v, width, height);
			}
			else
			{
				maskValue = 65535;
			}
			
			if (hasColor && maskValue)
			{
				maskData[maskIndex] = maskValue;
			}			
		}
	}
}

void projectImage2(cv::Mat& image, cv::Mat& imageF, cv::Mat& imageMask, OLine &ol, cv::Mat &destImage, cv::Mat &destImageF, cv::Mat &destImageMask)
{
	double hFov = ol.hFov;
	XMMATRIX mat = transform(ol.yaw, ol.pitch, ol.roll);
	//Matrix3d mat2 = transform2(ol.yaw, ol.pitch, ol.roll);
	int width = image.cols;
	int height = image.rows;

	int panoWidth = destImage.cols;
	int panoHeight = destImage.rows;

	unsigned char* src = image.data;
	unsigned char* srcF = imageF.data;
	unsigned char* dest = destImage.data;
	unsigned char* destF = destImageF.data;
	unsigned short* maskData = (unsigned short*)destImageMask.data;
	unsigned char* destPtr;
	bool hasColor;
	unsigned short maskValue;
	double u,v;
	for (int i = 0; i < panoHeight; i++)
	{
		for (int j = 0; j < panoWidth; j++)
		{
			int index = i * panoWidth * 3 + j * 3;
			int maskIndex = i * panoWidth + j;

			if (ol.lensType == 0)
			{
				pano2RectiLinear(j, i, width, height, ol, mat, u, v);
			}
			else if (ol.lensType == 2 || ol.lensType == 3)
			{
				pano2FishEye(j, i, width, height, ol, mat, u, v);
				//pano2FishEye2(j, i, width, height, ol, mat2, u, v);
			}

			destPtr = dest + index;
			hasColor = sampleColorBilinear(src, u, v, width, height, destPtr);
			destPtr = destF + index;
			hasColor = sampleColorBilinear(srcF, u, v, width, height, destPtr);
			if (imageMask.data != 0)
			{
				maskValue = sampleMask(imageMask, u, v, width, height);
			}
			else
			{
				maskValue = 65535;
			}

			if (hasColor && maskValue)
			{
				maskData[maskIndex] = maskValue;
			}			
		}
	}
}

void blendImages(cv::Mat image[], cv::Mat imageMask[], cv::Mat destAlphas[], int num, cv::Mat &destImage)
{
	unsigned char* dest[MAXCAMERANUM];
	unsigned short* maskData[MAXCAMERANUM];
	unsigned char* alphas[MAXCAMERANUM];
	for (int i = 0; i < num; i++)
	{
		dest[i] = image[i].data;
		maskData[i] = (unsigned short*)imageMask[i].data;
		alphas[i] = destAlphas[i].data;
	}
	unsigned char* destP = destImage.data;
	unsigned char* destPtr;

	int panoWidth = image[0].cols;
	int panoHeight = image[0].rows;
	for (int i = 0; i < panoHeight; i++)
	{
		for (int j = 0; j < panoWidth; j++)
		{
			int index = i * panoWidth * 3 + j * 3;
			int maskIndex = i * panoWidth + j;
			destPtr = destP + index;

			for (int k = 0; k < num; k++)
			{
				if (alphas[k][maskIndex] == 255)
				{
					destPtr[0] = dest[k][index    ];
					destPtr[1] = dest[k][index + 1];
					destPtr[2] = dest[k][index + 2];
					break;
				}
				else if (alphas[k][maskIndex] != 0)
				{
					destPtr[0] += dest[k][index    ] * alphas[k][maskIndex] / 255.0f;
					destPtr[1] += dest[k][index + 1] * alphas[k][maskIndex] / 255.0f;
					destPtr[2] += dest[k][index + 2] * alphas[k][maskIndex] / 255.0f;
				}
				
			}
		}
	}
}

void blendImages2(cv::Mat image[], cv::Mat imageF[], cv::Mat destAlphas[], cv::Mat destAlphasF[], int num, cv::Mat &destImage)
{
	unsigned char* dest[MAXCAMERANUM];
	unsigned char* destF[MAXCAMERANUM];
	unsigned char* alphas[MAXCAMERANUM];
	unsigned char* alphasF[MAXCAMERANUM];
	for (int i = 0; i < num; i++)
	{
		dest[i] = image[i].data;
		destF[i] = imageF[i].data;
		alphas[i] = destAlphas[i].data;
		alphasF[i] = destAlphasF[i].data;
	}
	unsigned char* destP = destImage.data;
	unsigned char* destPtr;

	int panoWidth = image[0].cols;
	int panoHeight = image[0].rows;
	float b,g,r,b1,g1,r1,b2,g2,r2,bF,gF,rF;
	bool blend;
	for (int i = 0; i < panoHeight; i++)
	{
		for (int j = 0; j < panoWidth; j++)
		{
			int index = i * panoWidth * 3 + j * 3;
			int maskIndex = i * panoWidth + j;
			destPtr = destP + index;

			b=g=r=b1=g1=r1=b2=g2=r2=bF=gF=rF =0;
			blend = true;
			for (int k = 0; k < num; k++)
			{
				if (alphasF[k][maskIndex] == 255)
				{
					blend = false;
					destPtr[0] = dest[k][index    ];
					destPtr[1] = dest[k][index + 1];
					destPtr[2] = dest[k][index + 2];
					break;
				}
				else
				{
					if (alphas[k][maskIndex] == 255 || (alphas[k][maskIndex] == 0 && alphasF[k][maskIndex] != 0))
					{
						if (alphas[k][maskIndex] == 255)
						{
							if(dest[k][index    ] > destF[k][index    ]) b1 += dest[k][index    ] - destF[k][index    ];
							else b2 += destF[k][index    ] - dest[k][index    ];
							if(dest[k][index + 1] > destF[k][index + 1]) g1 += dest[k][index + 1] - destF[k][index + 1];
							else g2 += destF[k][index + 1] - dest[k][index + 1];
							if(dest[k][index + 2] > destF[k][index + 2]) r1 += dest[k][index + 2] - destF[k][index + 2];
							else r2 += destF[k][index + 2] - dest[k][index + 2];
						}

						bF += destF[k][index    ] * alphasF[k][maskIndex] / 255.0f;
						gF += destF[k][index + 1] * alphasF[k][maskIndex] / 255.0f;
						rF += destF[k][index + 2] * alphasF[k][maskIndex] / 255.0f;
						//b=g=r=b1=g1=r1=b2=g2=r2=bF=gF=rF =0;
					}
					else if (alphas[k][maskIndex] != 0)
					{
						float a = alphas[k][maskIndex] / 255.0f;
						if(dest[k][index    ] > destF[k][index    ])
							b1 += (dest[k][index    ] - destF[k][index    ]) * a;
						else 
							b2 += (destF[k][index    ] - dest[k][index    ]) * a;
						if(dest[k][index + 1] > destF[k][index + 1])
							g1 += (dest[k][index + 1] - destF[k][index + 1]) * a;
						else
							g2 += (destF[k][index + 1] - dest[k][index + 1]) * a;
						if(dest[k][index + 2] > destF[k][index + 2])
							r1 += (dest[k][index + 2] - destF[k][index + 2]) * a;
						else 
							r2 += (destF[k][index + 2] - dest[k][index + 2]) * a;

						bF += destF[k][index    ] * alphasF[k][maskIndex] / 255.0f;
						gF += destF[k][index + 1] * alphasF[k][maskIndex] / 255.0f;
						rF += destF[k][index + 2] * alphasF[k][maskIndex] / 255.0f;
						//b=g=r=b1=g1=r1=b2=g2=r2=bF=gF=rF =0;
					}
				}
			}
			if (blend)
			{
				b = bF + b1 - b2;
				g = gF + g1 - g2;
				r = rF + r1 - r2;
				if(b < 0) b = 0;
				if(b > 255) b = 255;
				if(g < 0) g = 0;
				if(g > 255) g = 255;
				if(r < 0) r = 0;
				if(r > 255) r = 255;
				destPtr[0] = b;
				destPtr[1] = g;
				destPtr[2] = r;
			}

		}
	}
}

void maskToEdgeDistance(cv::Mat mask)
{
	unsigned short* maskData = (unsigned short*)mask.data;
	int width = mask.cols;
	int height = mask.rows;
	
	for (int i = 0; i < height; i++)
	{
		int dist = 0;
		int end = width;
		bool begin = false;
		for (int j = 0; j < end; j++)
		{
			int k = j % width;
			int maskIndex = i * width + k;
			if (maskData[maskIndex] == 0 && !begin)
			{
				begin = true;
				end = j + width;
			}

			if (maskData[maskIndex] != 0 && begin)
			{
				if(dist < maskData[maskIndex])
				{
					dist++;
					maskData[maskIndex] = dist;
				}

			}
			else
				dist = 0;

		}

		dist = 0;
		begin = false;
		end = -1;
		for (int j = width * 2 - 1; j > end; j--)
		{
			int k = j % width;
			int maskIndex = i * width + k;
			if (maskData[maskIndex] == 0 && !begin)
			{
				begin = true;
				end = j - width;
			}

			if (maskData[maskIndex] != 0 && begin)
			{
				if(dist < maskData[maskIndex])
				{
					dist++;
					maskData[maskIndex] = dist;
				}

			}
			else
				dist = 0;
		}
	}

	for (int i = 0; i < width; i++)
	{
		int dist = 0;
		bool begin = false;
		for (int j = 0; j < height; j++)
		{
			int maskIndex = j * width + i;
			if (maskData[maskIndex] == 0 && !begin)
			{
				begin = true;
				//end = j + height;
			}

			if (maskData[maskIndex] != 0 && begin)
			{
				if(dist < maskData[maskIndex])
				{
					dist++;
					maskData[maskIndex] = dist;
				}
				else
				{
					dist = maskData[maskIndex];
				}
			}
			else
				dist = 0;

		}

		dist = 0;
		begin = false;
		for (int j = height - 1; j > -1; j--)
		{
			int maskIndex = j * width + i;
			if (maskData[maskIndex] == 0 && !begin)
			{
				begin = true;
				//end = j - height;
			}

			if (maskData[maskIndex] != 0 && begin)
			{
				if(dist < maskData[maskIndex])
				{
					dist++;
					maskData[maskIndex] = dist;
				}
				else
				{
					dist = maskData[maskIndex];
				}
			}
			else
				dist = 0;
		}
	}
}

void maskToEdgeDistanceUchar(cv::Mat mask)
{
	unsigned char* maskData = mask.data;
	int width = mask.cols;
	int height = mask.rows;

	for (int i = 0; i < height; i++)
	{
		int dist = 0;
		int end = width;
		bool begin = false;
		for (int j = 0; j < end; j++)
		{
			int k = j % width;
			int maskIndex = i * width + k;
			if (maskData[maskIndex] == 0 && !begin)
			{
				begin = true;
				end = j + width;
			}

			if (maskData[maskIndex] != 0 && begin)
			{
				if(dist < maskData[maskIndex])
				{
					dist++;
					maskData[maskIndex] = dist;
				}

			}
			else
				dist = 0;

		}

		dist = 0;
		begin = false;
		end = -1;
		for (int j = width * 2 - 1; j > end; j--)
		{
			int k = j % width;
			int maskIndex = i * width + k;
			if (maskData[maskIndex] == 0 && !begin)
			{
				begin = true;
				end = j - width;
			}

			if (maskData[maskIndex] != 0 && begin)
			{
				if(dist < maskData[maskIndex])
				{
					dist++;
					maskData[maskIndex] = dist;
				}

			}
			else
				dist = 0;
		}
	}

	for (int i = 0; i < width; i++)
	{
		int dist = 0;
		bool begin = false;
		for (int j = 0; j < height; j++)
		{
			int maskIndex = j * width + i;
			if (!begin && maskData[maskIndex] == 0)
			{
				begin = true;
				//end = j + height;//todo: chuantou
			}
			
			if (maskData[maskIndex] != 0 && begin)
			{
				if(dist < maskData[maskIndex])
				{
					dist++;
					maskData[maskIndex] = dist;
				}
				else
				{
					dist = maskData[maskIndex];
				}
			}
			else
				dist = 0;

		}

		dist = 0;
		begin = false;
		for (int j = height - 1; j > -1; j--)
		{
			int maskIndex = j * width + i;
			if (maskData[maskIndex] == 0 && !begin)
			{
				begin = true;
				//end = j - height;
			}

			if (maskData[maskIndex] != 0 && begin)
			{
				if(dist < maskData[maskIndex])
				{
					dist++;
					maskData[maskIndex] = dist;
				}
				else
				{
					dist = maskData[maskIndex];
				}
			}
			else
				dist = 0;
		}
	}
}

void maskToEdgeDistance(cv::Mat mask, cv::Mat validMask)
{
	unsigned char* maskData = mask.data;
	unsigned short* validMaskData = (unsigned short*)validMask.data;
	int width = mask.cols;
	int height = mask.rows;

	for (int i = 0; i < height; i++)
	{
		int dist = 0;
		int end = width;
		bool begin = false;
		for (int j = 0; j < end; j++)
		{
			int k = j % width;
			int maskIndex = i * width + k;
			if (maskData[maskIndex] == 0 && !begin)
			{
				begin = true;
				end = j + width;
			}

			if (maskData[maskIndex] != 0 && begin)
			{
				if(validMaskData[maskIndex] > 0)
				{
					if(dist < maskData[maskIndex])
					{
						dist++;
						maskData[maskIndex] = dist;
					}
				}
				else
				{
					maskData[maskIndex] = 0;
				}

			}
			else
				dist = 0;

		}

		dist = 0;
		begin = false;
		end = -1;
		for (int j = width * 2 - 1; j > end; j--)
		{
			int k = j % width;
			int maskIndex = i * width + k;
			if (maskData[maskIndex] == 0 && !begin)
			{
				begin = true;
				end = j - width;
			}

			if (maskData[maskIndex] != 0 && begin)
			{
				if(validMaskData[maskIndex] > 0)
				{
					if(dist < maskData[maskIndex])
					{
						dist++;
						maskData[maskIndex] = dist;
					}
				}
				else
				{
					maskData[maskIndex] = 0;
				}

			}
			else
				dist = 0;
		}
	}

	for (int i = 0; i < width; i++)
	{
		int dist = 0;
		bool begin = false;
		for (int j = 0; j < height; j++)
		{
			int maskIndex = j * width + i;
			if (maskData[maskIndex] == 0 && !begin)
			{
				begin = true;
				//end = j + height;
			}

			if (maskData[maskIndex] != 0 && begin && validMaskData[maskIndex] > 0)
			{
				if(validMaskData[maskIndex] > 0)
				{
					if(dist < maskData[maskIndex])
					{
						dist++;
						maskData[maskIndex] = dist;
					}
					else
					{
						dist = maskData[maskIndex];
					}
				}
				else
				{
					maskData[maskIndex] = 0;
				}
			}
			else
				dist = 0;

		}

		dist = 0;
		begin = false;
		for (int j = height - 1; j > -1; j--)
		{
			int maskIndex = j * width + i;
			if (maskData[maskIndex] == 0 && !begin)
			{
				begin = true;
				//end = j - height;
			}

			if (maskData[maskIndex] != 0 && begin && validMaskData[maskIndex] > 0)
			{
				if(validMaskData[maskIndex] > 0)
				{
					if(dist < maskData[maskIndex])
					{
						dist++;
						maskData[maskIndex] = dist;
					}
					else
					{
						dist = maskData[maskIndex];
					}
				}
				else
				{
					maskData[maskIndex] = 0;
				}
			}
			else
				dist = 0;
		}
	}
}

void expandMaskUchar(cv::Mat mask, int expandSize, cv::Mat destMask)
{
	int width = mask.cols;
	int height = mask.rows;
	destMask = mask.clone();
	unsigned char* maskData = mask.data;
	unsigned char* destMaskData = destMask.data;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int maskIndex = i * width + j;
			int nj = (j + 1) % width;
			int nextIndex = i * width + nj;
			if (maskData[maskIndex] == 255 && maskData[nextIndex] == 0)
			{
				j++;
				for (int k = 0; k < expandSize; k++)
				{
					nj = (j + k) % width;
					nextIndex = i * width + nj;
					destMaskData[nextIndex] = 255;
				}
				j += expandSize;
			}

		}

		for (int j = width - 1; j > -1; j--)
		{
			int maskIndex = i * width + j;
			int nj = (j - 1 + width) % width;
			int nextIndex = i * width + nj;
			if (maskData[maskIndex] == 255 && maskData[nextIndex] == 0)
			{
				j--;
				for (int k = 0; k < expandSize; k++)
				{
					nj = (j - k + width) % width;
					nextIndex = i * width + nj;
					destMaskData[nextIndex] = 255;
				}
				j -= expandSize;
			}

		}
		
	}

	for (int j = 0; j < width; j++)
	{
		for (int i = 0; i < height; i++)
		{
			int maskIndex = i * width + j;
			int ni = (i + 1) % height;
			int nextIndex = ni * width + j;
			if (maskData[maskIndex] == 255 && maskData[nextIndex] == 0)
			{
				i++;
				for (int k = 0; k < expandSize; k++)
				{
					ni = i + k;
					if(ni > height - 1) break;

					nextIndex = ni * width + j;
					maskData[nextIndex] = 255;
				}
				i += expandSize;
			}

		}

		for (int i = height - 1; i > -1; i--)
		{
			int maskIndex = i * width + j;
			int ni = (i - 1 + height) % height;
			int nextIndex = ni * width + j;
			if (maskData[maskIndex] == 255 && maskData[nextIndex] == 0)
			{
				i--;
				for (int k = 0; k < expandSize; k++)
				{
					ni = i - k;
					if(ni < 0) break;

					nextIndex = ni * width + j;
					maskData[nextIndex] = 255;
				}
				i -= expandSize;
			}

		}

	}
}

void expandMaskUchar(cv::Mat mask, int expandSize)
{
	int width = mask.cols;
	int height = mask.rows;
	unsigned char* maskData = mask.data;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int maskIndex = i * width + j;
			int nj = (j + 1) % width;
			int nextIndex = i * width + nj;
			if (maskData[maskIndex] == 255 && maskData[nextIndex] == 0)
			{
				j++;
				for (int k = 0; k < expandSize; k++)
				{
					nj = (j + k) % width;
					nextIndex = i * width + nj;
					maskData[nextIndex] = 255;
				}
				j += expandSize;
			}

		}

		for (int j = width - 1; j > -1; j--)
		{
			int maskIndex = i * width + j;
			int nj = (j - 1 + width) % width;
			int nextIndex = i * width + nj;
			if (maskData[maskIndex] == 255 && maskData[nextIndex] == 0)
			{
				j--;
				for (int k = 0; k < expandSize; k++)
				{
					nj = (j - k + width) % width;
					nextIndex = i * width + nj;
					maskData[nextIndex] = 255;
				}
				j -= expandSize;
			}

		}

	}

	for (int j = 0; j < width; j++)
	{
		for (int i = 0; i < height - 1; i++)
		{
			int maskIndex = i * width + j;
			int ni = i + 1;
			int nextIndex = ni * width + j;
			if (maskData[maskIndex] == 255 && maskData[nextIndex] == 0)
			{
				i++;
				for (int k = 0; k < expandSize; k++)
				{
					ni = i + k;
					if(ni > height - 1) break;

					nextIndex = ni * width + j;
					maskData[nextIndex] = 255;
				}
				i += expandSize;
			}

		}

		for (int i = height - 1; i > 0; i--)
		{
			int maskIndex = i * width + j;
			int ni = i - 1;
			int nextIndex = ni * width + j;
			if (maskData[maskIndex] == 255 && maskData[nextIndex] == 0)
			{
				i--;
				for (int k = 0; k < expandSize; k++)
				{
					ni = i - k;
					if(ni < 0) break;

					nextIndex = ni * width + j;
					maskData[nextIndex] = 255;
				}
				i -= expandSize;
			}

		}

	}
}

void featherMasks2(cv::Mat image[], cv::Mat imageMask[], cv::Mat imageAlpha[], cv::Mat imageAlphaF[], int num)
{
	unsigned char* dest[MAXCAMERANUM];
	unsigned short* maskData[MAXCAMERANUM];
	unsigned char* alphas[MAXCAMERANUM];
	unsigned char* alphasF[MAXCAMERANUM];

	for (int i = 0; i < num; i++)
	{
		dest[i] = image[i].data;
		maskData[i] = (unsigned short*)imageMask[i].data;
		alphas[i] = imageAlpha[i].data;
		alphasF[i] = imageAlphaF[i].data;
	}

	cv::Mat imageIndexMask = imageAlpha[0].clone();
	unsigned char* imageIndexData = imageIndexMask.data;

	int imageIndex[MAXCAMERANUM];
	int maxIndex;
	int panoWidth = image[0].cols;
	int panoHeight = image[0].rows;
	for (int i = 0; i < panoHeight; i++)
	{
		for (int j = 0; j < panoWidth; j++)
		{
			int index = i * panoWidth * 3 + j * 3;
			int maskIndex = i * panoWidth + j;

			maxIndex = -1;
			for (int k = 0; k < num; k++)
			{
				if (maskData[k][maskIndex] > 0)
				{
					if (maxIndex == -1)
					{
						maxIndex = k;
					}else if (maskData[k][maskIndex] > maskData[maxIndex][maskIndex])
					{
						maxIndex = k;
					}
				}

			}
			if (maxIndex != -1)
			{
				alphas[maxIndex][maskIndex] = 255;
				alphasF[maxIndex][maskIndex] = 255;
				imageIndexData[maskIndex] = maskIndex + 1;
			}

		}
	}

	cv::Mat imageAlphasOutter[MAXCAMERANUM];
	cv::Mat imageAlphasInner[MAXCAMERANUM];
	unsigned char* alphasOutter[MAXCAMERANUM];
	unsigned char* alphasInner[MAXCAMERANUM];
	cv::Mat imageAlphasOutterF[MAXCAMERANUM];
	unsigned char* alphasOutterF[MAXCAMERANUM];
	for (int i = 0; i < num; i++)
	{
		imageAlphasInner[i] = imageAlpha[i].clone();
		imageAlphasOutter[i] = imageAlpha[i].clone();
		expandMaskUchar(imageAlphasOutter[i], BLENDSIZE);	
		maskToEdgeDistance(imageAlphasOutter[i], imageMask[i]);
		maskToEdgeDistanceUchar(imageAlphasInner[i]);
		alphasOutter[i] = imageAlphasOutter[i].data;
		alphasInner[i] = imageAlphasInner[i].data;

		imageAlphasOutterF[i] = imageAlpha[i].clone();
		expandMaskUchar(imageAlphasOutterF[i], BLENDSIZE_F);	
		maskToEdgeDistance(imageAlphasOutterF[i], imageMask[i]);
		alphasOutterF[i] = imageAlphasOutterF[i].data;

	}

	//begin calculate alpha
	int indexNum = 0;
	int alphasTmp[MAXCAMERANUM];
	for (int i = 0; i < panoHeight; i++)
	{
		for (int j = 0; j < panoWidth; j++)
		{
			int index = i * panoWidth * 3 + j * 3;
			int maskIndex = i * panoWidth + j;

			int maxIndex = -1;
			for (int k = 0; k < num; k++)
			{
				if (alphas[k][maskIndex] == 255)
				{
					maxIndex = k;
					break;
				}
				
			}

			if (maxIndex != -1 && alphasInner[maxIndex][maskIndex] < BLENDSIZE)
			{
				for (int k = 0; k < num; k++)
				{
					if (alphasOutter[k][maskIndex] != 0)
					{
						if (k != maxIndex)
						{
							float total = alphasOutter[k][maskIndex] + alphasInner[maxIndex][maskIndex];
							float alpha = 0;
							if (total > BLENDSIZE)
							{
								total = BLENDSIZE;
							}
							alpha = alphasInner[maxIndex][maskIndex] * 255 / total / 2 + 127;

							float dist = alpha * total / 255;
							float b = total;
							float c = total / 2.9;
							float ex = - (dist - b ) * (dist - b ) / 2 / c / c;
							alpha = 255 * expf(ex);

							for (int l = 0; l < k; l++)
							{
								if (alphasOutter[l][maskIndex] != 0)
								{
									alphas[l][maskIndex] = alphas[l][maskIndex] * alpha / 255.0f;
								}
							}

							alphas[k][maskIndex] = 255 - alpha;
						}
						else
						{
							float alpha = 255 - (alphasInner[maxIndex][maskIndex] * 255 / BLENDSIZE / 2 + 127);

							float total = BLENDSIZE;
							float dist = alpha * total / 255;
							float b = total;
							float c = total / 2.9;
							float ex = - (dist - b ) * (dist - b ) / 2 / c / c;
							alpha = 255 * expf(ex);

							for (int l = 0; l < k; l++)
							{
								if (alphasOutter[l][maskIndex] != 0)
								{
									alphas[l][maskIndex] = alphas[l][maskIndex] * alpha / 255.0f;
								}
							}

							alphas[k][maskIndex] = 255 - alpha;
						}
					}
					
				}

				float ceshi = 0;
				for (int k = 0; k < num; k++)
				{
					ceshi += alphas[k][maskIndex];
				}
				for (int k = 0; k < num; k++)
				{
					alphas[k][maskIndex] = alphas[k][maskIndex] * 255.0f / ceshi;
				}

			}
			
			if (maxIndex != -1 && alphasInner[maxIndex][maskIndex] < BLENDSIZE_F)
			{
				for (int k = 0; k < num; k++)
				{
					if (alphasOutterF[k][maskIndex] != 0)
					{
						if (k != maxIndex)
						{
							float total = alphasOutterF[k][maskIndex] + alphasInner[maxIndex][maskIndex];
							float alpha = 0;
							if (total > BLENDSIZE_F)
							{
								total = BLENDSIZE_F;
							}
							alpha = alphasInner[maxIndex][maskIndex] * 255 / total / 2 + 127;

							float dist = alpha * total / 255;
							float b = total;
							float c = total / 2.9;
							float ex = - (dist - b ) * (dist - b ) / 2 / c / c;
							alpha = 255 * expf(ex);

							for (int l = 0; l < k; l++)
							{
								if (alphasOutterF[l][maskIndex] != 0)
								{
									alphasF[l][maskIndex] = alphasF[l][maskIndex] * alpha / 255.0f;
								}
							}

							alphasF[k][maskIndex] = 255 - alpha;
						}
						else
						{
							float alpha = 255 - (alphasInner[maxIndex][maskIndex] * 255 / BLENDSIZE_F / 2 + 127);

							float total = BLENDSIZE_F;
							float dist = alpha * total / 255;
							float b = total;
							float c = total / 2.9;
							float ex = - (dist - b ) * (dist - b ) / 2 / c / c;
							alpha = 255 * expf(ex);

							for (int l = 0; l < k; l++)
							{
								if (alphasOutterF[l][maskIndex] != 0)
								{
									alphasF[l][maskIndex] = alphasF[l][maskIndex] * alpha / 255.0f;
								}
							}

							alphasF[k][maskIndex] = 255 - alpha;
						}
					}

				}

				float ceshi = 0;
				for (int k = 0; k < num; k++)
				{
					ceshi += alphasF[k][maskIndex];
				}
				for (int k = 0; k < num; k++)
				{
					alphasF[k][maskIndex] = alphasF[k][maskIndex] * 255.0f / ceshi;
				}

			}
		}
	}
}

void featherMasks(cv::Mat image[], cv::Mat imageMask[], cv::Mat imageAlpha[], int num)
{
	unsigned char* dest[MAXCAMERANUM];
	unsigned short* maskData[MAXCAMERANUM];
	unsigned char* alphas[MAXCAMERANUM];

	for (int i = 0; i < num; i++)
	{
		dest[i] = image[i].data;
		maskData[i] = (unsigned short*)imageMask[i].data;
		alphas[i] = imageAlpha[i].data;
	}

	cv::Mat imageIndexMask = imageAlpha[0].clone();
	unsigned char* imageIndexData = imageIndexMask.data;

	int imageIndex[MAXCAMERANUM];
	int maxIndex;
	int panoWidth = image[0].cols;
	int panoHeight = image[0].rows;
	for (int i = 0; i < panoHeight; i++)
	{
		for (int j = 0; j < panoWidth; j++)
		{
			int index = i * panoWidth * 3 + j * 3;
			int maskIndex = i * panoWidth + j;

			maxIndex = -1;
			for (int k = 0; k < num; k++)
			{
				if (maskData[k][maskIndex] > 0)
				{
					if (maxIndex == -1)
					{
						maxIndex = k;
					}else if (maskData[k][maskIndex] > maskData[maxIndex][maskIndex])
					{
						maxIndex = k;
					}
				}

			}
			if (maxIndex != -1)
			{
				alphas[maxIndex][maskIndex] = 255;
				imageIndexData[maskIndex] = maskIndex + 1;
			}

		}
	}

	cv::Mat imageAlphasOutter[MAXCAMERANUM];
	cv::Mat imageAlphasInner[MAXCAMERANUM];
	unsigned char* alphasOutter[MAXCAMERANUM];
	unsigned char* alphasInner[MAXCAMERANUM];
	for (int i = 0; i < num; i++)
	{
		imageAlphasInner[i] = imageAlpha[i].clone();
		imageAlphasOutter[i] = imageAlpha[i].clone();
		expandMaskUchar(imageAlphasOutter[i], BLENDSIZE);	
		maskToEdgeDistance(imageAlphasOutter[i], imageMask[i]);
		maskToEdgeDistanceUchar(imageAlphasInner[i]);
		alphasOutter[i] = imageAlphasOutter[i].data;
		alphasInner[i] = imageAlphasInner[i].data;
	}

	//begin calculate alpha
	int indexNum = 0;
	int alphasTmp[MAXCAMERANUM];
	for (int i = 0; i < panoHeight; i++)
	{
		for (int j = 0; j < panoWidth; j++)
		{
			int index = i * panoWidth * 3 + j * 3;
			int maskIndex = i * panoWidth + j;

			int maxIndex = -1;
			for (int k = 0; k < num; k++)
			{
				if (alphas[k][maskIndex] == 255)
				{
					maxIndex = k;
					break;
				}

			}

			if (maxIndex != -1 && alphasInner[maxIndex][maskIndex] < BLENDSIZE)
			{
				for (int k = 0; k < num; k++)
				{
					if (alphasOutter[k][maskIndex] != 0)
					{
						if (k != maxIndex)
						{
							float total = alphasOutter[k][maskIndex] + alphasInner[maxIndex][maskIndex];
							float alpha = 0;
							if (total > BLENDSIZE)
							{
								total = BLENDSIZE;
							}
							alpha = alphasInner[maxIndex][maskIndex] * 255 / total / 2 + 127;

							float dist = alpha * total / 255;
							float b = total;
							float c = total / 2.9;
							float ex = - (dist - b ) * (dist - b ) / 2 / c / c;
							alpha = 255 * expf(ex);

							for (int l = 0; l < k; l++)
							{
								if (alphasOutter[l][maskIndex] != 0)
								{
									alphas[l][maskIndex] = alphas[l][maskIndex] * alpha / 255.0f;
								}
							}

							alphas[k][maskIndex] = 255 - alpha;
						}
						else
						{
							float alpha = 255 - (alphasInner[maxIndex][maskIndex] * 255 / BLENDSIZE / 2 + 127);

							float total = BLENDSIZE;
							float dist = alpha * total / 255;
							float b = total;
							float c = total / 2.9;
							float ex = - (dist - b ) * (dist - b ) / 2 / c / c;
							alpha = 255 * expf(ex);

							for (int l = 0; l < k; l++)
							{
								if (alphasOutter[l][maskIndex] != 0)
								{
									alphas[l][maskIndex] = alphas[l][maskIndex] * alpha / 255.0f;
								}
							}

							alphas[k][maskIndex] = 255 - alpha;
						}
					}

				}

				float ceshi = 0;
				for (int k = 0; k < num; k++)
				{
					ceshi += alphas[k][maskIndex];
				}
				for (int k = 0; k < num; k++)
				{
					alphas[k][maskIndex] = alphas[k][maskIndex] * 255.0f / ceshi;
				}

			}

		}
	}

}

//解析配置文件
void parseOLine(const char* line, OLine &ol)
{
	ol.crop = 0;
	int index = 2;
	while(line[index] != '\0')
	{
		if(line[index] == ' ' || line[index] == '\t')
		{
			index++;
			continue;
		}
		
		if(line[index] == 'n')
		{
			index += 2;
			int endIndex = index + 1;
			while(line[endIndex] != '"') endIndex++;
			memset(ol.fileName, 0, 64);
			memcpy(ol.fileName, &line[index], endIndex - index);
		}
		else if(line[index] == 'f')
		{
			ol.lensType = atoi(&line[index + 1]);
		}
		else if(line[index] == 'y')
		{
			ol.yaw = atof(&line[index + 1]);
		}
		else if(line[index] == 'p')
		{
			ol.pitch = atof(&line[index + 1]);
		}
		else if(line[index] == 'r')
		{
			ol.roll = atof(&line[index + 1]);
		}
		else if(line[index] == 'v')
		{
			ol.hFov = atof(&line[index + 1]);
		}
		else if(line[index] == 'a')
		{
			ol.a = atof(&line[index + 1]);
		}
		else if(line[index] == 'b')
		{
			ol.b = atof(&line[index + 1]);
		}
		else if(line[index] == 'c')
		{
			ol.c = atof(&line[index + 1]);
		}
		else if(line[index] == 'd')
		{
			ol.d = atof(&line[index + 1]);
		}
		else if(line[index] == 'e')
		{
			ol.e = atof(&line[index + 1]);
		}
		else if(line[index] == 'C')
		{
			ol.crop = 1;
			ol.cl = atoi(&line[index + 1]);
			while(line[index] != ',') index++;
			ol.cr = atoi(&line[index + 1]);index++;
			while(line[index] != ',') index++;
			ol.ct = atoi(&line[index + 1]);index++;
			while(line[index] != ',') index++;
			ol.cb = atoi(&line[index + 1]);
		}
		
		while(line[index] != ' ' && line[index] != '\t' && line[index] != '\0') index++;
	}
}

//解析配置文件
bool parseStitchFile(const char* fileName, OLine ols[], int &num)
{
	FILE* fp = fopen(fileName, "r");
	if(fp == 0) return false;

	const int MaxLineSize = 300;
	char buf[MaxLineSize];
	int index = 0;
	while(fgets(buf, MaxLineSize, fp) != 0 && index < MAXCAMERANUM)
	{
		if (buf[0] == 'o')
		{
			parseOLine(buf, ols[index]);
			index++;
		}
		
	}

	num = index;
	return true;
}

void exposureTimes(cv::Mat image, double times)
{
	int width = image.cols;
	int height = image.rows;
	unsigned char* src = image.data;
	double b, g, r;
	double y, u, v;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int index = i * width * 3 + j * 3;
			b = *(src + index);
			g = *(src + index + 1);
			r = *(src + index + 2);

			y = 0.299 * r + 0.587 * g + 0.114 * b;
			u =  - 0.1687 * r - 0.3313 * g + 0.5 * b + 128;
			v = 0.5 * r - 0.4187 * g - 0.0813 * b + 128;

			y = y * times;
			if(y > 255) y = 255;

			r = y + 1.402 * (v-128);
			g = y - 0.34414 * (u-128) - 0.71414 * (v-128);
			b = y + 1.772 * (u-128);

			if(b < 0) b = 0;
			if(b > 255) b = 255;
			if(g < 0) g = 0;
			if(g > 255) g = 255;
			if(r < 0) r = 0;
			if(r > 255) r = 255;

			src[index + 0] = b;
			src[index + 1] = g;
			src[index + 2] = r;

		}
	}
}

double getYValue(unsigned char* pixel)
{
	double b, g, r;
	b = *(pixel);
	g = *(pixel + 1);
	r = *(pixel + 2);

	double y = 0.299 * r + 0.587 * g + 0.114 * b;
	return y;
}

void exposureImages(cv::Mat image[], cv::Mat imageMask[], int num)
{
	double exposures[MAXCAMERANUM][MAXCAMERANUM];
	memset(exposures, 0, sizeof(double) * MAXCAMERANUM * MAXCAMERANUM);

	unsigned char* dest[MAXCAMERANUM];
	unsigned short* maskData[MAXCAMERANUM];
	for (int i = 0; i < num; i++)
	{
		dest[i] = image[i].data;
		maskData[i] = (unsigned short*)imageMask[i].data;
	}

	int imageIndex[MAXCAMERANUM];
	int maskNum;
	int panoWidth = image[0].cols;
	int panoHeight = image[0].rows;
	for (int i = 0; i < panoHeight; i++)
	{
		for (int j = 0; j < panoWidth; j++)
		{
			int index = i * panoWidth * 3 + j * 3;
			int maskIndex = i * panoWidth + j;

			maskNum = 0;
			for (int k = 0; k < num; k++)
			{
				if (maskData[k][maskIndex] > 0)
				{
					imageIndex[maskNum] = k;
					maskNum++;
				}
			}
			
			if (maskNum > 1)
			{
				for (int m = 0; m < maskNum; m++)
				{
					double my = getYValue(&dest[imageIndex[m]][index]);
					for (int n = 0; n < maskNum; n++)
					{
						if (m != n)
						{
							double ny = getYValue(&dest[imageIndex[n]][index]);
							double ytimes = my > ny ? my / ny : ny / my;
							if (ytimes < 1.5)
							{
								exposures[imageIndex[m]][imageIndex[n]] += my;
							}
						}
						
					}
				}
			}
			
		}
	}
	
	for (int i = 0; i < num; i++)
	{
		exposures[i][i] = 1.0;
	}
	int overlapNum = 0;
	double totalExpo = 0.0;
	double expoTmp[MAXCAMERANUM] = {0};

	for (int it = 0; it < 5; it++)
	{
		for (int i = 0; i < num; i++)
		{
			overlapNum = 0;
			totalExpo = 0.0;
			for (int j = 0; j < num; j++)
			{
				if (i != j && exposures[i][j] > 1.0)
				{
					overlapNum++;
					totalExpo += exposures[j][i] * exposures[j][j] / (exposures[i][j] * exposures[i][i]);
				}

			}
			expoTmp[i] = totalExpo / overlapNum / 2 + 0.5;

		}
		for (int i = 0; i < num; i++)
		{
			exposures[i][i] = expoTmp[i];
		}
	}

	for (int i = 0; i < num; i++)
	{
		exposureTimes(image[i], exposures[i][i]);
	}
}

void test(cv::Mat image[], cv::Mat imageMask[], int num)
{
	unsigned char* dest[MAXCAMERANUM];
	unsigned char* maskData[MAXCAMERANUM];
	for (int i = 0; i < num; i++)
	{
		dest[i] = image[i].data;
		maskData[i] = (unsigned char*)imageMask[i].data;
	}

	int imageIndex[MAXCAMERANUM];
	int maskNum;
	int panoWidth = image[0].cols;
	int panoHeight = image[0].rows;
	int BLENDSIZE = 200;
	for (int i = 0; i < panoHeight; i++)
	{
		for (int j = 0; j < BLENDSIZE; j++)
		{
			int index = i * panoWidth * 3 + j * 3;
			int maskIndex = i * panoWidth + j;

			double val = 0;
			double pos = j * 1.0 / BLENDSIZE;
			//maskData[0][maskIndex] = 255 * pos;
			
			if (pos < 0.5)
			{
				val = 2 * pos * pos;
			}
			else
			{
				val = 1 - 2 * (1 - pos) * (1 - pos);
			}
			//maskData[0][maskIndex] = 255 * val;

			//float dist = j < i ? j : i;
			float dist = j;// < i ? j : i;
			float a = 255.0f;
			float b = BLENDSIZE;//.0f;
			float c = BLENDSIZE / 2.5;//(float)m_featherdelta;
			float ex = - (dist - b ) * (dist - b ) / 2 / c / c;
			//float ex = - (t - b ) * (t - b ) / 2 / c / c;
			float temp = a * expf(ex);
			maskData[0][maskIndex] = temp;
		}
	}

}

void blurImages(cv::Mat image, cv::Mat kernel, cv::Mat imageF)
{
	unsigned char* dest;
	unsigned char* destF;
	dest = image.data;
	destF = imageF.data;
	
	int imageIndex[MAXCAMERANUM];
	int maskNum;
	int panoWidth = image.cols;
	int panoHeight = image.rows;
	int panoWidthMax = panoWidth - 1;
	int panoHeightMax = panoHeight - 1;
	int kernelSize = kernel.rows;
	int halfKernelSize = (kernel.rows - 1) / 2;
	double r,g,b,a;
	for (int i = 0; i < panoHeight; i++)
	{
		for (int j = 0; j < panoWidth; j++)
		{
			r = g = b = a =0;
			for (int m = 0; m < kernelSize; m++)
			{
				int x = j - halfKernelSize + m;
				if(x < 0) x = 0;
				if(x > panoWidthMax) x = panoWidthMax;

				double val = kernel.ptr<float>(0)[m];

				int index = i * panoWidth * 3 + x * 3;
				int maskIndex = i * panoWidth + x;

				b += dest[index    ] * val;
				g += dest[index + 1] * val;
				r += dest[index + 2] * val;			
			}
			int index = i * panoWidth * 3 + j * 3;
			int maskIndex = i * panoWidth + j;
			destF[index    ] = b;
			destF[index + 1] = g;
			destF[index + 2] = r;
		}
	}
	for (int i = 0; i < panoHeight; i++)
	{
		for (int j = 0; j < panoWidth; j++)
		{
			r = g = b = a =0;
			for (int m = 0; m < kernelSize; m++)
			{
				int y = i - halfKernelSize + m;
				if(y < 0) y = 0;
				if(y > panoHeight - 1) y = panoHeight - 1;

				double val = kernel.ptr<float>(0)[m];

				int index = y * panoWidth * 3 + j * 3;
				int maskIndex = y * panoWidth + j;

				b += destF[index    ] * val;
				g += destF[index + 1] * val;
				r += destF[index + 2] * val;
			}
			int index = i * panoWidth * 3 + j * 3;
			int maskIndex = i * panoWidth + j;
			destF[index    ] = b;
			destF[index + 1] = g;
			destF[index + 2] = r;
		}
	}
}

void maskToEdgeDistanceSrc(cv::Mat mask)
{
	unsigned short* maskData = (unsigned short*)mask.data;
	int width = mask.cols;
	int height = mask.rows;

	for (int i = 0; i < height; i++)
	{
		int dist = 0;
		int end = width;
		bool begin = false;
		for (int j = 0; j < end; j++)
		{
			int k = j % width;
			int maskIndex = i * width + k;
			if (!begin)
			{
				begin = true;
				end = j + width;
			}

			if (maskData[maskIndex] != 0 && begin)
			{
				if(dist < maskData[maskIndex])
				{
					dist++;
					maskData[maskIndex] = dist;
				}

			}
			else
				dist = 0;

		}

		dist = 0;
		begin = false;
		end = -1;
		for (int j = width * 2 - 1; j > end; j--)
		{
			int k = j % width;
			int maskIndex = i * width + k;
			if (!begin)
			{
				begin = true;
				end = j - width;
			}

			if (maskData[maskIndex] != 0 && begin)
			{
				if(dist < maskData[maskIndex])
				{
					dist++;
					maskData[maskIndex] = dist;
				}

			}
			else
				dist = 0;
		}
	}

	for (int i = 0; i < width; i++)
	{
		int dist = 0;
		bool begin = false;
		for (int j = 0; j < height; j++)
		{
			int maskIndex = j * width + i;
			if (!begin)
			{
				begin = true;
				//end = j + height;
			}

			if (maskData[maskIndex] != 0 && begin)
			{
				if(dist < maskData[maskIndex])
				{
					dist++;
					maskData[maskIndex] = dist;
				}
				else
				{
					dist = maskData[maskIndex];
				}
			}
			else
				dist = 0;

		}

		dist = 0;
		begin = false;
		for (int j = height - 1; j > -1; j--)
		{
			int maskIndex = j * width + i;
			if (!begin)
			{
				begin = true;
				//end = j - height;
			}

			if (maskData[maskIndex] != 0 && begin)
			{
				if(dist < maskData[maskIndex])
				{
					dist++;
					maskData[maskIndex] = dist;
				}
				else
				{
					dist = maskData[maskIndex];
				}
			}
			else
				dist = 0;
		}
	}
}

//再来一个既含有高斯核直径kernelSize，又有单独的sigma的版本：  
//double sigma0 = (halfSize - 1)/ 2.0;  
cv::Mat gaussian_kernal(int kernelSize, double sigma0)  
{    
	int halfSize = (kernelSize-1)/ 2;  
	cv::Mat K(kernelSize, kernelSize, CV_64FC1);    

	//生成二维高斯核    
	double s2 = 2.0 * sigma0 * sigma0;    
	for(int i = (-halfSize); i <= halfSize; i++)    
	{    
		int m = i + halfSize;    
		for (int j = (-halfSize); j <= halfSize; j++)    
		{    
			int n = j + halfSize;    
			double v = exp(-(1.0*i*i + 1.0*j*j) / s2);    
			K.ptr<double>(m)[n] = v;    
		}    
	}    
	cv::Scalar all = sum(K);    
	cv::Mat gaussK;    
	K.convertTo(gaussK, CV_64FC1, (1/all[0]));    

	return gaussK;    
}

void filterImages(cv::Mat image[], cv::Mat imageMask[], int num, cv::Mat kernel, cv::Mat imageF[], cv::Mat imageMaskF[])
{
	unsigned char* dest[MAXCAMERANUM];
	unsigned char* maskData[MAXCAMERANUM];
	unsigned char* destF[MAXCAMERANUM];
	unsigned char* maskDataF[MAXCAMERANUM];
	for (int i = 0; i < num; i++)
	{
		dest[i] = image[i].data;
		maskData[i] = imageMask[i].data;
		destF[i] = imageF[i].data;
		maskDataF[i] = imageMaskF[i].data;
	}

	int imageIndex[MAXCAMERANUM];
	int maskNum;
	int panoWidth = image[0].cols;
	int panoHeight = image[0].rows;
	int panoWidthMax = panoWidth - 1;
	int panoHeightMax = panoHeight - 1;
	int kernelSize = kernel.cols;
	int halfKernelSize = (kernel.cols - 1) / 2;
	double r,g,b,a;
	for (int i = 0; i < panoHeight; i++)
	{
		for (int j = 0; j < panoWidth; j++)
		{
			r = g = b = a =0;
			for (int m = 0; m < kernelSize; m++)
			{
				for (int n = 0; n < kernelSize; n++)
				{
					int x = j - halfKernelSize + m;
					int y = i - halfKernelSize + n;

					if(x < 0) x = 0;
					if(x > panoWidthMax) x = panoWidthMax;
					if(y < 0) y = 0;
					if(y > panoHeightMax) y = panoHeightMax;

					double val = kernel.ptr<double>(m)[n];

					int index = y * panoWidth * 3 + x * 3;
					int maskIndex = y * panoWidth + x;

					b += dest[0][index    ] * val;
					g += dest[0][index + 1] * val;
					r += dest[0][index + 2] * val;
					a += maskData[0][maskIndex] * val;

				}
			}
			int index = i * panoWidth * 3 + j * 3;
			int maskIndex = i * panoWidth + j;
			destF[0][index    ] = b;
			destF[0][index + 1] = g;
			destF[0][index + 2] = r;
			maskDataF[0][maskIndex] = a;
		}
	}
}

void filterYImages(cv::Mat image[], int num, cv::Mat kernel, cv::Mat imageF[])
{
	unsigned char* dest[MAXCAMERANUM];
	unsigned char* destF[MAXCAMERANUM];
	for (int i = 0; i < num; i++)
	{
		dest[i] = image[i].data;
		destF[i] = imageF[i].data;
	}

	int imageIndex[MAXCAMERANUM];
	int maskNum;
	int panoWidth = image[0].cols;
	int panoHeight = image[0].rows;
	int panoWidthMax = panoWidth - 1;
	int panoHeightMax = panoHeight - 1;
	int kernelSize = kernel.cols;
	int halfKernelSize = (kernel.cols - 1) / 2;
	double a;
	for (int i = 0; i < panoHeight; i++)
	{
		for (int j = 0; j < panoWidth; j++)
		{
			a =0;
			for (int m = 0; m < kernelSize; m++)
			{
				for (int n = 0; n < kernelSize; n++)
				{
					int x = j - halfKernelSize + m;
					int y = i - halfKernelSize + n;

					if(x < 0) x = 0;
					if(x > panoWidthMax) x = panoWidthMax;
					if(y < 0) y = 0;
					if(y > panoHeightMax) y = panoHeightMax;

					double val = kernel.ptr<double>(m)[n];

					int index = y * panoWidth + x;
					a += dest[0][index] * val;

				}
			}
			int index = i * panoWidth + j;
			destF[0][index] = a;
		}
	}
}

void convertToYImage(cv::Mat image[], cv::Mat imageY[], int num)
{
	unsigned char* dest[MAXCAMERANUM];
	unsigned char* destY[MAXCAMERANUM];
	for (int i = 0; i < num; i++)
	{
		dest[i] = image[i].data;
		destY[i] = imageY[i].data;
	}
	int panoWidth = image[0].cols;
	int panoHeight = image[0].rows;

	for (int i = 0; i < panoHeight; i++)
	{
		for (int j = 0; j < panoWidth; j++)
		{
			int index = i * panoWidth * 3 + j * 3;
			int yIndex = i * panoWidth + j;
			
			destY[0][yIndex] = getYValue(&dest[0][index]);
		}
	}
}

void subtractYImages(cv::Mat imageY[], cv::Mat imageYF[], cv::Mat imageYL[], int num)
{
	unsigned char* dest[MAXCAMERANUM];
	unsigned char* dest2[MAXCAMERANUM];
	unsigned char* dest3[MAXCAMERANUM];
	for (int i = 0; i < num; i++)
	{
		dest[i] = imageY[i].data;
		dest2[i] = imageYF[i].data;
		dest3[i] = imageYL[i].data;
	}
	int panoWidth = imageY[0].cols;
	int panoHeight = imageY[0].rows;

	for (int i = 0; i < panoHeight; i++)
	{
		for (int j = 0; j < panoWidth; j++)
		{
			int yIndex = i * panoWidth + j;
			int delta = dest[0][yIndex] - dest2[0][yIndex];
			dest3[0][yIndex] = delta > 0 ? delta : -delta;
		}
	}
}


int main( void )
{
	int CameraNum = 0;
	OLine ols[MAXCAMERANUM];

	if(!parseStitchFile("./records/11.txt", ols, CameraNum)) return 0;
	std::string path = "./records/";
	//if(!parseStitchFile("./qq.txt", ols, CameraNum)) return 0;
	//std::string path = "./";
	//if(!parseStitchFile("./bug3/0000.txt", ols, CameraNum)) return 0;
	//std::string path = "./bug3/";

	cv::Mat destImage[MAXCAMERANUM],destMask[MAXCAMERANUM],destAlphas[MAXCAMERANUM],destAlphasF[MAXCAMERANUM],testMask[MAXCAMERANUM];
	cv::Mat destImageF[MAXCAMERANUM],destMaskF[MAXCAMERANUM];//,destImageFC[MAXCAMERANUM];
	cv::Mat destImageY[MAXCAMERANUM],destImageYF[MAXCAMERANUM],destImageYF2[MAXCAMERANUM],destImageYF3[MAXCAMERANUM];
	cv::Mat destImageL[MAXCAMERANUM],destImageL2[MAXCAMERANUM],destImageL3[MAXCAMERANUM];
	cv::Mat image[MAXCAMERANUM], croppedImage[MAXCAMERANUM], cropImageMask[MAXCAMERANUM], cImage[MAXCAMERANUM], cImageMask[MAXCAMERANUM];
	for (int i = 0; i < CameraNum; i++)
	{
		std::string fullPath = path + ols[i].fileName;
		image[i] = cv::imread(fullPath);//这里替换成你们代码

		destImage[i] = cv::Mat(PANOHEIGHT, PANOWIDTH, CV_8UC3, cv::Scalar(0));
		destMask[i] = cv::Mat(PANOHEIGHT, PANOWIDTH, CV_16UC1, cv::Scalar(0));
		destAlphas[i] = cv::Mat(PANOHEIGHT, PANOWIDTH, CV_8UC1, cv::Scalar(0));

		destImageF[i] = cv::Mat(PANOHEIGHT, PANOWIDTH, CV_8UC3, cv::Scalar(0));
		destAlphasF[i] = cv::Mat(PANOHEIGHT, PANOWIDTH, CV_8UC1, cv::Scalar(0));

		if (ols[i].crop == 1)
		{
			cropImage(image[i],ols[i].cl,ols[i].cr,ols[i].ct,ols[i].cb, croppedImage[i], cropImageMask[i]);
			//calibrateImage(croppedImage[i], &cropImageMask[i], ols[i].a, ols[i].b, ols[i].c, cImage[i], cImageMask[i]);
			projectImage(croppedImage[i], cropImageMask[i], ols[i], destImage[i], destMask[i]);
		}
		else
		{
			//calibrateImage(image[i], 0, ols[i].a, ols[i].b, ols[i].c, cImage[i], cImageMask[i]);
			croppedImage[i] = image[i];
			cropImageMask[i] = cv::Mat(image[i].rows, image[i].cols, CV_16UC1, cv::Scalar(65535));
			cv::Mat kernel = cv::getGaussianKernel(159, 159 / 4, CV_32F);
			cv::Mat bImage(croppedImage[i].rows, croppedImage[i].cols, CV_8UC3, cv::Scalar(0));
			blurImages(croppedImage[i], kernel, bImage);
			int dist = 159;
			int sigma = dist / 4;
			//cv::GaussianBlur(croppedImage[i], bImage, cv::Size(dist, dist), sigma, sigma);
			
			//projectImage(croppedImage[i], cropImageMask[i], ols[i], destImage[i], destMask[i]);
			projectImage2(croppedImage[i], bImage, cropImageMask[i], ols[i], destImage[i], destImageF[i], destMask[i]);

			//cv::imshow("croppedImage[0]", croppedImage[i]);
			//cv::imshow("destImage[0]", destImage[i]);
			//cv::imshow("destImageF[0]", destImageF[i]);
			//cv::waitKey(0);
		}

		maskToEdgeDistance(destMask[i]);

		//destMask[i] = cv::Mat(PANOHEIGHT, PANOWIDTH, CV_16UC1, cv::Scalar(66535));
		//destMask[i] = destMask[i] * 128;
		//cv::imshow("destMask[0]", destMask[i]);
		//cv::waitKey(0);
	}
	
	exposureImages(destImage, destMask, CameraNum);
	featherMasks2(destImage, destMask, destAlphas, destAlphasF, CameraNum);
//	featherMasks(destImage, destMask, destAlphas, CameraNum);

	cv::Mat destPano(PANOHEIGHT, PANOWIDTH, CV_8UC3, cv::Scalar(0));
	blendImages2(destImage, destImageF, destAlphas, destAlphasF, CameraNum, destPano);
	//blendImages(destImage, destMask, destAlphas, CameraNum, destPano);

	//cv::imshow("destAlphas[0]", destAlphas[0]);
	//cv::imshow("destAlphas[1]", destAlphas[1]);
	//cv::imshow("destAlphas[2]", destAlphas[2]);
	//cv::imshow("destAlphas[3]", destAlphas[3]);
	//cv::imshow("destAlphas[4]", destAlphas[4]);
	//cv::imshow("destAlphas[5]", destAlphas[5]);
	//cv::imshow("destImage[5]", destImage[5]);
	//cv::imshow("destImage[3]", destImage[2]);
	//cv::imshow("destMask[0]", destMask[0] * 255);
	//cv::imshow("destMask[1]", destMask[1] * 255);
	//cv::imshow("destMask[2]", destMask[2] * 255);
	//cv::imshow("destMask[3]", destMask[3] * 255);
	//cv::imshow("destMask[4]", destMask[4] * 255);
	//cv::imshow("destMask[5]", destMask[5] * 255);

	//cv::imwrite("000000.jpg", destImage[0]);
	//cv::namedWindow("pano", cv::WINDOW_NORMAL);
	cv::imshow("pano", destPano);
	cv::imwrite("haha.jpg", destPano);
	cv::waitKey(0);

	return 0;
}

