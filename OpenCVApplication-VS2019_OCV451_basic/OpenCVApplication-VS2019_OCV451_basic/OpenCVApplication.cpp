// OpenCVApplication.cpp : Defines the entry point for the console application.

#include "stdafx.h"
#include "common.h"
#include <fstream>
#include <cmath>
#include <vector>
#include <random>

using namespace std;

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
	Mat img = Mat(height, width, CV_8UC3, Scalar(255,255,255));
	uchar* imgData = img.data;

	ifstream input(fname);

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

	Point2f point0(0, o0 + o1 * 0);
	Point2f pointW(width, o0 + o1 * width);

	line(img, point0, pointW, Scalar(0, 0, 255));

	input.close();
	
	return img;
}

Mat polar_coordinates(char* fname)
{
	int height = 500, width = 500;
	Mat img = Mat(height, width, CV_8UC3, Scalar(255, 255, 255));
	uchar* imgData = img.data;

	ifstream input(fname);

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

	Point2f point0(0, (float)y0);
	Point2f pointW(width, (float)yw);

	line(img, point0, pointW, Scalar(0, 0, 255));

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

		imshow("Closed Form", cf);
		imshow("Polar Coordinates", pc);

		waitKey();
		destroyAllWindows();
	}
}

// Lab 2 - RANSAC
vector<Point> getPoints(Mat img)
{
	vector<Point> points;

	uchar* imgData = img.data;

	int height = img.rows;
	int width = img.cols;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int pixel = i * width + j;

			if (imgData[pixel] == 0)
			{
				points.push_back(Point(j, i));
			}
		}
	}

	return points;
}

void ransac()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat img = imread(fname, CV_8UC1);
		vector<Point> points = getPoints(img);

		int n = points.size();	// total number of points
		double t = 10;			// inlier threshold
		double p = 0.99;		// that at least one of the random samples of s points is free from outliers
		double q = 0.8;			// probability that the selected point is an inlier
		int s = 2;				// selected points in each iteration
		int N;					// number of iterations;
		double T;					// number of inliers

		N = log(1 - p) / log(1 - pow(q, s));
		T = q * n;

		random_device rd;
		default_random_engine gen(rd());
		uniform_int_distribution<int> distribution(0, n - 1);

		double best_a = 0, best_b = 0, best_c = 0;
		int best_inliers = 0;

		for (int i = 1; i <= N && best_inliers < T; i++)
		{
			// chose 2 random points
			int point1, point2;
			point1 = distribution(gen);
			do
			{
				point2 = distribution(gen);
			}while (point2 == point1);

			// line equation aX + bY + c = 0
			Point p1 = points[point1];
			Point p2 = points[point2];
			double a = p1.y - p2.y;
			double b = p2.x - p1.x;
			double c = p1.x * p2.y - p2.x * p1.y;

			// distance of each point to the line (ax + by + c) / sqrt(a^2 + b^2)
			vector<double> distances;
			for (Point p : points)
			{
				double d = abs(a * p.x + b * p.y + c) / sqrt(a * a + b * b);
				distances.push_back(d);
			}

			// valid points
			int inliers = 0;
			for (double d : distances)
			{
				if (d < t)
				{
					inliers++;
				}
			}

			if (inliers > best_inliers)
			{
				best_inliers = inliers;
				best_a = a;
				best_b = b;
				best_c = c;
			}
		}

		// aX + bY + c = 0
		Point2f point0(0, (-best_a * 0 - best_c) / best_b);
		Point2f pointW(img.cols, (-best_a * img.cols - best_c) / best_b);

		line(img, point0, pointW, Scalar(0, 0, 0));

		imshow("img", img);

		waitKey();
		destroyAllWindows();
	}
}

// Lab 3 - Hough
struct peak {
	int theta, ro, hval;
	bool operator < (const peak& o) const {
		return hval > o.hval;
	}
};

vector<peak> findPeaks(Mat h, int n, int th)
{
	int n2 = n / 2;

	int height = h.rows;
	int width = h.cols;

	int* hData = (int*)h.data;

	vector <peak> peaks;

	for (int i = n2; i < height - n2; i++)
	{
		for (int j = n2; j < width - n2; j++)
		{
			int pixel = i * width + j;

			if (hData[pixel] > th)
			{
				bool isMax = true;
				for (int u = -n2; u < n2; u++)
				{
					for (int v = -n2; v < n2; v++)
					{
						int pixel2 = (i + u) * width + (j + v);
						if (hData[pixel2] > hData[pixel])
							isMax = false;
					}
				}

				if (isMax)
				{
					peak p;
					p.hval = hData[pixel];
					p.ro = i;
					p.theta = j;
					peaks.push_back(p);
				}
			}
		}
	}

	return peaks;
}

void hough()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat imgSrc = imread(fname, IMREAD_GRAYSCALE);
		Mat img = imgSrc.clone();

		int k, n, th = 20;
		cout << "k = ";
		cin >> k;
		cout << "n = ";
		cin >> n;

		double height = img.rows, width = img.cols;

		int omax = 360;
		int pmax = sqrt(height * height + width * width);
		int o, p;
		int ostep = 1, pstep = 1;

		Mat h(pmax + 1, 360, CV_32SC1);
		h.setTo(0);

		uchar* imgData = img.data;
		int* hData = (int*)h.data;

		// hough accumulator
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				int pixel = i * width + j;

				if (imgData[pixel] == 255)
				{
					o = 0;
					while (o < omax)
					{
						double orad = CV_PI / 180 * o;
						p = j * cos(orad) + i * sin(orad);
						if (p <= pmax && p >= 0)
							hData[p * 360 + o] ++;
						o += ostep;
					}
				}
			}
		}

		// find maximum
		int max = 0;
		for (int i = 0; i < h.rows; i++)
		{
			for (int j = 0; j < h.cols; j++)
			{
				int pixel = i * h.cols + j;

				if (hData[pixel] > max)
				{
					max = hData[pixel];
				}
			}
		}

		// scale accumulator to diplay
		Mat houghImg;
		h.convertTo(houghImg, CV_8UC1, 255.f / max);

		// get and sort peaks
		vector<peak> peaks = findPeaks(h, n, th);
		sort(peaks.begin(), peaks.end());

		// draw lines
		cvtColor(img, img, COLOR_GRAY2RGB);
		for (int i = 0; i < k; i++)
		{
			double x, y;
			peak p = peaks[i];

			double rad = CV_PI / 180 * p.theta;

			x = 0;
			y = (p.ro - x * cos(rad)) / sin(rad);
			Point2f point0(x, y);

			x = img.cols;
			y = (p.ro - x * cos(rad)) / sin(rad);
			Point2f pointW(x, y);

			line(img, point0, pointW, Scalar(0, 255, 0));
		}
		
		// show images
		imshow("src", imgSrc);
		imshow("hough", houghImg);
		imshow("lines", img);

		waitKey();
		destroyAllWindows();
	}
}

// Lab 4 - DT
pair<int, int> center_of_mass(Mat img)
{
	pair<int, int> com{ 0, 0 }; // center of mass - {mx, my}
	int pixels = 0;

	uchar* imgData = img.data;
	int height = img.rows;
	int width = img.cols;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int pixel = i * width + j;

			if (imgData[pixel] == 0)
			{
				com.first += i;
				com.second += j;
				pixels++;
			}
		}
	}

	com.first = round(((double)com.first) / pixels);
	com.second = round(((double)com.second) / pixels);

	return com;
}

Mat distace_transform(Mat img)
{
	Mat dt = img.clone();
	uchar* dtData = dt.data;

	vector<pair<int, int>> topdownKernel{ {-1, -1}, {-1, 0}, {-1, 1}, {0, -1} };
	vector<pair<int, int>> bottomupKernel{ {0, 1}, {1, -1}, {1, 0}, {1, 1} };
	vector<vector<int>> weights{ {3, 2, 3}, {2, 0, 2}, {3, 2, 3} };

	int kernel_size = 3;
	int k2 = kernel_size / 2;
	int height = dt.rows;
	int width = dt.cols;

	// top-down scan
	for (int i = k2; i < height - k2; i++)
	{
		for (int j = k2; j < width - k2; j++)
		{
			int pixel = i * width + j;

			int dt_min = dtData[pixel]; // + 0

			for (pair<int, int> pii : topdownKernel)
			{
				int pixelK = (i + pii.first) * width + (j + pii.second);
				int value = dtData[pixelK] + weights[pii.first + 1][pii.second + 1];
				value = value > 255 ? 255 : value;
				if (value < dt_min)
					dt_min = value;
			}

			dtData[pixel] = dt_min;
		}
	}

	// bottom-up scan
	for (int i = height - 1 - k2; i > k2; i--)
	{
		for (int j = width - 1 - k2; j > k2; j--)
		{
			int pixel = i * width + j;

			int dt_min = dtData[pixel]; // + 0

			for (pair<int, int> pii : bottomupKernel)
			{
				int pixelK = (i + pii.first) * width + (j + pii.second);
				int value = dtData[pixelK] + weights[pii.first + 1][pii.second + 1];
				value = value > 255 ? 255 : value;
				if (value < dt_min)
					dt_min = value;
			}

			dtData[pixel] = dt_min;
		}
	}

	return dt;
}

double pattern_matching(Mat tmp_dt, Mat ukn, pair<int, int> com_dif)
{
	int sum = 0;
	int pixels = 0;
	int height = ukn.rows;
	int width = ukn.cols;

	uchar* tmpData = tmp_dt.data;
	uchar* uknData = ukn.data;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int pixel = i * width + j;

			if (uknData[pixel] == 0 && isInside(tmp_dt, i + com_dif.first, j + com_dif.second))
			{
				pixels++;

				int tmpPixel = (i + com_dif.first) * width + (j + com_dif.second);
				sum += tmpData[tmpPixel];
			}
		}
	}

	return ((double)sum) / pixels;
}

void dt()
{
	char fname_template[MAX_PATH];
	while (openFileDlg(fname_template))
	{
		Mat tmp = imread(fname_template, IMREAD_GRAYSCALE);
		Mat tmp_dt = distace_transform(tmp);
		pair<int, int> tmp_com = center_of_mass(tmp);

		Mat tmp_stack;
		hconcat(tmp, tmp_dt, tmp_stack);

		char fname_unknown[MAX_PATH];
		while (openFileDlg(fname_unknown))
		{
			Mat ukn = imread(fname_unknown, IMREAD_GRAYSCALE);
			pair<int, int> ukn_com = center_of_mass(ukn);
			
			pair<int, int> com_dif = { tmp_com.first - ukn_com.first, tmp_com.second - ukn_com.second };
			pair<int, int> com_dif0 = { 0, 0 };

			cout << "With translation: " << pattern_matching(tmp_dt, ukn, com_dif) << '\n';
			cout << "No translation:   " << pattern_matching(tmp_dt, ukn, com_dif0) << '\n';

			imshow("Template", tmp_stack);
			imshow("Unknown", ukn);

			waitKey();
			destroyWindow("Unknown");
		}

		destroyAllWindows();
	}
}

int main()
{
	int op = 3;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");

		printf(" Lab01--------------------------------\n");
		printf("\t 1 - Regression\n");

		printf(" Lab02--------------------------------\n");
		printf("\t 2 - RANSAC\n");

		printf(" Lab03--------------------------------\n");
		printf("\t 3 - Hough\n");

		printf(" Lab04--------------------------------\n");
		printf("\t 4 - DT\n");

		printf("\n0 - Exit\n\n");

		printf("Option: ");
		cin >> op;
		switch (op)
		{
			case 1:
				regression();
			case 2:
				ransac();
			case 3:
				hough();
			case 4:
				dt();
		}
	}
	while (op!=0);
	return 0;
}