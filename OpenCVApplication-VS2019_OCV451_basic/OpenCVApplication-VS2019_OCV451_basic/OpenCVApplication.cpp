// OpenCVApplication.cpp : Defines the entry point for the console application.

#include "stdafx.h"
#include "common.h"
#include <fstream>
#include <cmath>
#include <vector>

bool isInside(Mat img, int row, int col)
{
	int height = img.rows;
	int width = img.cols;

	if (row < 0 || col < 0) return false;
	if (col >= width || row >= height) return false;

	return true;
}

// Lab1 - Regression
Mat closed_form(char* fname)
{
	int height = 500, width = 500;
	Mat img = Mat(height, width, CV_8UC3, cv::Scalar(255,255,255));
	uchar* imgData = img.data;

	std::ifstream input(fname);

	int n = 0;
	double sumX = 0, sumY = 0, sumXY = 0, sumX2 =0;

	// process data
	input >> n;
	for(int i = 0; i < n; i++)
	{
		double x, y;

		input >> x >> y;

		sumX += x;
		sumY += y;
		sumXY += x * y;
		sumX2 += x * x;

		// add point to image
		if (isInside(img, (int)x, (int)y))
		{
			int pixel = (int)y * width * 3 + (int)x * 3;

			imgData[pixel] = 0;
			imgData[pixel + 1] = 0;
			imgData[pixel + 2] = 0;
		}
	}

	double o1, o0;

	o1 = ((double)n * sumXY - sumX * sumY) / ((double)n * sumX2 - sumX * sumX);
	o0 = (1.0 / n) * (sumY - o1 * sumX);

	cv::Point2f point0(0, o0 + o1 * 0);
	cv::Point2f pointW(width, o0 + o1 * width);

	cv::line(img, point0, pointW, cv::Scalar(0, 0, 255));

	input.close();
	
	return img;
}

Mat polar_coordinates(char* fname)
{
	int height = 500, width = 500;
	Mat img = Mat(height, width, CV_8UC3, cv::Scalar(255, 255, 255));
	uchar* imgData = img.data;

	std::ifstream input(fname);

	int n = 0;
	double sumX = 0, sumY = 0, sumXY = 0, sumY2X2 = 0;

	// process data
	input >> n;
	for (int i = 0; i < n; i++)
	{
		double x, y;

		input >> x >> y;

		sumX += x;
		sumY += y;
		sumXY += x * y;
		sumY2X2 += (y * y - x * x);

		// add point to image
		if (isInside(img, (int)x, (int)y))
		{
			int pixel = (int)y * width * 3 + (int)x * 3;

			imgData[pixel] = 0;
			imgData[pixel + 1] = 0;
			imgData[pixel + 2] = 0;
		}
	}

	double b, p;

	double n1 = 1.0 / (double)n;
	b = -0.5 * atan2(2 * sumXY - 2 * n1 * sumX * sumY,
				sumY2X2 + n1 * sumX * sumX - n1 * sumY * sumY);
	p = n1 * (cos(b) * sumX + sin(b) * sumY);

	double y0 = (p - cos(b) * 0) / sin(b);
	double yw = (p - cos(b) * width) / sin(b);

	cv::Point2f point0(0, (float)y0);
	cv::Point2f pointW(width, (float)yw);

	cv::line(img, point0, pointW, cv::Scalar(0, 0, 255));

	input.close();

	return img;
}

void regression()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat cf = closed_form(fname);
		Mat pc = polar_coordinates(fname);

		cv::imshow("Closed Form", cf);
		cv::imshow("Polar Coordinates", pc);

		cv::waitKey();
		cv::destroyAllWindows();
	}
}

int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");

		printf(" Lab01--------------------------------\n");
		printf("\t 1 - Regression\n");

		printf("\n0 - Exit\n\n");

		printf("Option: ");
		scanf("%d", &op);
		switch (op)
		{
			case 1:
				regression();
		}
	}
	while (op!=0);
	return 0;
}