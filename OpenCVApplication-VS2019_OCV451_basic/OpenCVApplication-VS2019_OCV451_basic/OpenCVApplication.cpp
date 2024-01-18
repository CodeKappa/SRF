// OpenCVApplication.cpp : Defines the entry point for the console application.

#include "stdafx.h"
#include "common.h"
#include <fstream>
#include <cmath>
#include <vector>
#include <random>
#include <float.h>
#include <limits>

using namespace std;

bool isInside(Mat img, int row, int col)
{
	int height = img.rows;
	int width = img.cols;

	if (row < 0 || col < 0) return false;
	if (col >= width || row >= height) return false;

	return true;
}

/* Histogram display function - display a histogram using bars
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const string& name, vector<int> hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i < hist_cols; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

// Lab1 - Regression
Mat closed_form(char* fname)
{
	int height = 500, width = 500;
	Mat img = Mat(height, width, CV_8UC3, Scalar(255, 255, 255));
	uchar* imgData = img.data;

	ifstream input(fname);

	int n = 0;
	double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;

	// process data
	input >> n;
	for (int i = 0; i < n; i++)
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
			} while (point2 == point1);

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

// Lab 4 - Distance Transform
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

// Lab 5 - Statistical Data Analysis
void vectorToCsv(vector<double> v)
{
	ofstream fout("vector.csv");
	for (double value : v)
	{
		fout << value << ',';
	}
	fout.close();
}

void matrixToCsv(vector<vector<double>> m, string name)
{
	ofstream fout(name);
	for (vector<double> row : m)
	{
		for (double value : row)
		{
			fout << value << ',';
		}
		fout << '\n';
	}
	fout.close();
}

void statistical_analysis()
{
	char folderName[MAX_PATH];
	//while (openFolderDlg(folderName))
	if (1 == 1)
	{
		strcpy(folderName, "./Data/prs_res_Statistics");
		Mat featureMatrix = Mat(400, 19 * 19, CV_8UC1);

		uchar* fmData = featureMatrix.data;
		int width = 19 * 19;

		// feature matrix
		char fname[256];
		for (int k = 0; k < 400; k++)
		{
			sprintf(fname, "%s/face%05d.bmp", folderName, k + 1);
			Mat img = imread(fname, IMREAD_GRAYSCALE);

			uchar* imgData = img.data;
			for (int i = 0; i < 19; i++)
			{
				for (int j = 0; j < 19; j++)
				{
					fmData[(k * width) + i * 19 + j] = imgData[(i * 19 + j)];
				}
			}
		}

		// vector mean
		vector<double> mean_vector;
		for (int col = 0; col < 19 * 19; col++)
		{
			double mean = 0;
			for (int row = 0; row < 400; row++)
			{
				mean += fmData[row * 361 + col];
			}
			mean /= 400.0;
			mean_vector.push_back(mean);
		}

		vectorToCsv(mean_vector);

		// cov matrix
		vector<vector<double>> cov_matrix(361, vector<double>(361, 0));
		vector<double> aux;

		for (int col1 = 0; col1 < 361; col1++)
		{
			double mean = 0;
			for (int col2 = 0; col2 < 361; col2++)
			{
				for (int row = 0; row < 400; row++)
				{
					mean += (fmData[row * 361 + col1] - mean_vector[col1])
						* (fmData[row * 361 + col2] - mean_vector[col2]);
				}
				mean /= 400.0;
				cov_matrix[col1][col2] = mean;
			}
		}

		matrixToCsv(cov_matrix, "cov.csv");

		// deviation
		vector<double> deviation;
		for (int col = 0; col < 19 * 19; col++)
		{
			double dev = 0;
			for (int row = 0; row < 400; row++)
			{
				double value = fmData[row * 361 + col] - mean_vector[col];
				dev += value * value;
			}
			dev /= 400;
			dev = sqrt(dev);
			deviation.push_back(dev);
		}

		// corelation
		vector<vector<double>> cor_matrix(361, vector<double>(361));
		for (int row = 0; row < 361; row++)
		{
			for (int col = 0; col < 361; col++)
			{
				double value = cov_matrix[row][col];
				value = value / (deviation[row] * deviation[col]);
				cor_matrix[row][col] = value;
			}
		}

		matrixToCsv(cor_matrix, "cor.csv");

		// render
		int a1, b1, a2, b2;
		//cin >> a1 >> b1 >> a2 >> b2;
		a1 = 5;
		b1 = 4;
		a2 = 5;
		b2 = 14;

		//Mat see = Mat(256, 256, CV_8UC1);
		Mat see(256, 256, CV_8UC1, Scalar(255, 255, 255));
		uchar* seeData = see.data;

		int p1 = a1 * 19 + b1;
		int p2 = a2 * 19 + b2;

		for (int k = 0; k < 400; k++)
		{
			int a = fmData[k * 361 + p1];
			int b = fmData[k * 361 + p2];

			seeData[a * 256 + b] = 0;
		}

		cout << "coef: " << cor_matrix[p1][p2] << '/n';

		imshow("Feature matrix", featureMatrix);
		imshow("Image", see);

		waitKey();
		destroyAllWindows();
	}
}

// Lab 6 - Principal Component Analysis
Mat pca_coef(Mat X, Mat Q, int n, int d, int k)
{
	Mat Qk = Q(Range(0, d), Range(0, k));
	Mat Xcoeff = X * Qk;
	//Mat Xk = X * Qk * (Qk.t());
	return Xcoeff;
}

void pca()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		// read data
		ifstream fin(fname);

		int n, d;
		double x;

		fin >> n >> d;

		Mat X(n, d, CV_64FC1);
		double* xData = (double*)X.data;

		for (int i = 0; i < n; i++)
		{
			vector<double> aux;
			for (int j = 0; j < d; j++)
			{
				int pixel = i * d + j;

				fin >> x;
				xData[pixel] = x;
			}
		}
		fin.close();

		// data normalization
		vector<double> mean_vector;
		double sum;
		for (int col = 0; col < d; col++)
		{
			sum = 0;
			for (int row = 0; row < n; row++)
			{
				int pixel = row * d + col;
				sum += xData[pixel];
			}
			mean_vector.push_back(sum / n);
		}

		cout << "mean_vector: ";
		for (auto v : mean_vector)
		{
			cout << v << ' ';
		}
		cout << '\n';

		for (int col = 0; col < d; col++)
		{
			sum = 0;
			for (int row = 0; row < n; row++)
			{
				int pixel = row * d + col;
				xData[pixel] -= mean_vector[col];
			}
		}

		// covariance matrix
		Mat cov = X.t() * X / (n - 1);

		// eigen decomposition
		Mat lmbda, Q;
		eigen(cov, lmbda, Q);
		Q = Q.t();

		// print eigen
		cout << "Eigen values:";
		for (int i = 0; i < d; i++)
		{
			cout << lmbda.at<double>(i, 0) << ' ';
		}
		cout << '\n';

		// pca coefficients
		int k = 1;
		Mat Qk = Q(Range(0, d), Range(0, k));
		Mat Xcoeff = X * Qk;
		Mat Xk = X * Qk * (Qk.t());
		double* xkData = (double*)Xk.data;

		// mean absolute difference

		double mad = 0;

		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < d; j++)
			{
				int pixel = i * d + j;

				mad += abs(xData[pixel] - xkData[pixel]);
			}
		}

		mad /= ((double)n) * d;

		cout << "Mean absolute difference: " << mad << '\n';
		vector<double> mins;
		vector<double> maxs;

		double m, M;

		// max and min coeff
		Xcoeff = pca_coef(X, Q, n, d, d);
		double* xcData = (double*)Xcoeff.data;

		for (int col = 0; col < d; col++)
		{
			m = DBL_MAX;
			M = DBL_MIN;
			for (int row = 0; row < n; row++)
			{
				int pixel = row * d + col;

				if (xcData[pixel] > M)
				{
					M = xcData[pixel];
				}
				if (xcData[pixel] < m)
				{
					m = xcData[pixel];
				}
			}
			mins.push_back(m);
			maxs.push_back(M);
		}



		cin >> k;
		M = maxs[0] - mins[0] > maxs[1] - mins[1] ? maxs[0] - mins[0] : maxs[1] - mins[1];
		Mat Xc = Xcoeff(Range(0, n), Range(0, k));
		double* data = (double*)Xc.data;
		if (k == 2)
		{
			for (int col = 0; col < k; col++)
			{
				for (int row = 0; row < n; row++)
				{
					int pixel = row * d + col;

					data[pixel] -= mins[col];
				}
			}

			Mat img(M + 1, M + 1, CV_8UC1);
			for (int row = 0; row < n; row++)
			{
				int pixel = row * d;

				img.at<uchar>(data[pixel], data[pixel + 1]) = 0;

			}

			imshow("2", img);
		}
		else if (k == 3)
		{
			for (int col = 0; col < 2; col++)
			{
				for (int row = 0; row < n; row++)
				{
					int pixel = row * d + col;

					data[pixel] -= mins[col];
				}
			}

			for (int row = 0; row < n; row++)
			{
				int pixel = row * d + 2;

				data[pixel] -= mins[2];
				data[pixel] /= (maxs[2] - mins[2]);
				data[pixel] *= 255;
				data[pixel] = data[pixel] > 255 ? 255 : data[pixel];
			}

			Mat img(M + 1, M + 1, CV_8UC1);
			for (int row = 0; row < n; row++)
			{
				int pixel = row * d;

				img.at<uchar>(data[pixel], data[pixel + 1]) = data[pixel + 2];

			}
			imshow("3", img);
		}

		// find kth

		double b = 0;
		for (int row = 0; row < d; row++)
		{
			b += lmbda.at<double>(row, 0);
		}

		for (k = 1; k <= d; k++)
		{
			double a = 0;
			for (int row = 0; row < k; row++)
			{
				a += lmbda.at<double>(row, 0);
			}



			if ((a / b) > 0.99)
			{
				cout << "min k: " << k;
				break;
			}
		}

		waitKey();
		destroyAllWindows();
	}
}

// Lab 7 - K-means Clustering
vector<int> kmeans(vector<vector<int>> points, int d, int k, vector<vector<double>>& centers, int max_iterations = 20)
{
	vector<int> clusters(points.size());

	random_device rd;
	default_random_engine gen(rd());
	uniform_int_distribution<int> distribution(0, points.size());

	// initialize centers
	set<int> unique_centers;
	while (unique_centers.size() != k)
	{
		int number = distribution(gen);
		unique_centers.insert(number);
	}

	for (int it : unique_centers)
	{
		vector<double>center(points[it].begin(), points[it].end());
		centers.push_back(center);
	}

	int iteration = 0;
	do
	{
		// assign points to clusters
		for (int i = 0; i < points.size(); i++)
		{
			double min_dist = DBL_MAX;

			for (int j = 0; j < centers.size(); j++)
			{
				int dist = 0;
				for (int l = 0; l < d; l++)
				{
					dist += (centers[j][l] - points[i][l]) * (centers[j][l] - points[i][l]);
				}
				dist = sqrt(dist);
				if (dist < min_dist)
				{
					min_dist = dist;
					clusters[i] = j;
				}
			}
		}

		// update centers
		vector<vector<double>> new_centers(centers.size(), vector<double>(d));
		vector<int> points_per_center(centers.size());
		for (int i = 0; i < points.size(); i++)
		{
			points_per_center[clusters[i]]++;
			for (int l = 0; l < d; l++)
			{
				new_centers[clusters[i]][l] += points[i][l];
			}
		}
		for (int i = 0; i < centers.size(); i++)
		{
			for (int l = 0; l < d; l++)
			{
				new_centers[i][l] /= points_per_center[i];
			}
		}

		// if there is no change -> break
		bool change = false;
		for (int i = 0; i < centers.size() && change == false; i++)
		{
			for (int l = 0; l < d; l++)
			{
				if (new_centers[i][l] != centers[i][l])
				{
					change = true;
					break;
				}
			}
		}

		if (change == false)
			break;

		centers = new_centers;

		iteration++;
		cout << iteration << '\n';
	} while (iteration < max_iterations);

	return clusters;
}

Mat voronoi(Mat source, vector<vector<double>> centers, vector<Vec3b> colors, int d)
{
	int rows = source.rows;
	int cols = source.cols;
	Mat Voronoi(rows, cols, CV_8UC3);

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			double min_dist = DBL_MAX;
			int center = 0;

			for (int l = 0; l < centers.size(); l++)
			{
				int dist = 0;
				dist += (centers[l][0] - i) * (centers[l][0] - i);
				dist += (centers[l][1] - j) * (centers[l][1] - j);
				dist = sqrt(dist);
				if (dist < min_dist)
				{
					min_dist = dist;
					center = l;
				}
			}

			Voronoi.at<Vec3b>(i, j) = colors[center];
		}
	}

	return Voronoi;
}

void points_kmeans(Mat source, vector<Vec3b> colors)
{
	int d = 2, k;
	cout << "k= ";
	cin >> k;
	//k = 3;

	int rows = source.rows;
	int cols = source.cols;
	uchar* data = source.data;

	vector<vector<int>> points;

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			int pixel = (i * cols + j) * 3;

			if (data[pixel] == 0 && data[pixel + 1] == 0 && data[pixel + 2] == 0)
			{
				vector<int> dimensions;
				dimensions.push_back(i);
				dimensions.push_back(j);
				points.push_back(dimensions);
			}
		}
	}

	vector<vector<double>> centers;
	vector<int> clusters = kmeans(points, d, k, centers);

	Mat img(rows, cols, CV_8UC3, Scalar(255, 255, 255));

	for (int i = 0; i < points.size(); i++)
	{
		img.at<Vec3b>(points[i][0], points[i][1]) = colors[clusters[i]];
	}

	for (int i = 0; i < centers.size(); i++)
	{
		Point p = Point(centers[i][1], centers[i][0]);
		circle(img, p, 1, colors[i], 5, 5, 0);
	}

	Mat Voronoi = voronoi(source, centers, colors, d);


	imshow("kmeans", img);
	imshow("voronoi", Voronoi);

	waitKey();
}

void gray_kmeans(Mat source)
{
	int d = 1, k;
	cout << "k= ";
	cin >> k;
	//k = 10;

	Mat greyMat;
	cvtColor(source, greyMat, cv::COLOR_RGB2GRAY);

	int rows = greyMat.rows;
	int cols = greyMat.cols;
	uchar* data = greyMat.data;

	vector<vector<int>> points;

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			int pixel = i * cols + j;
			vector<int> dimensions;
			dimensions.push_back(data[pixel]);
			points.push_back(dimensions);
		}
	}

	vector<vector<double>> centers;
	vector<int> clusters = kmeans(points, d, k, centers, 10);


	vector<double> intensities(k);
	vector<int> count(k);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			int pixel = i * cols + j;
			intensities[clusters[pixel]] += data[pixel];
			count[clusters[pixel]]++;
		}
	}

	for (int i = 0; i < k; i++)
	{
		intensities[i] /= count[i];
	}

	Mat img(rows, cols, CV_8UC1);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			img.at<uchar>(i, j) = intensities[clusters[i * cols + j]];
		}
	}

	imshow("gray", img);
}

void color_kmeans(Mat source)
{
	int d = 3, k;
	cout << "k= ";
	cin >> k;
	//k = 3;

	int rows = source.rows;
	int cols = source.cols;
	uchar* data = source.data;

	vector<vector<int>> points;

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			int pixel = (i * cols + j) * 3;
			vector<int> dimensions;
			dimensions.push_back(data[pixel]);
			dimensions.push_back(data[pixel + 1]);
			dimensions.push_back(data[pixel + 2]);
			points.push_back(dimensions);
		}
	}

	vector<vector<double>> centers;
	vector<int> clusters = kmeans(points, d, k, centers);

	vector<vector<double>> intensities(k, vector<double>(3));
	vector<int> count(k);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			int pixel = i * cols + j;
			intensities[clusters[pixel]][0] += source.at<Vec3b>(i, j)[0];
			intensities[clusters[pixel]][1] += source.at<Vec3b>(i, j)[1];
			intensities[clusters[pixel]][2] += source.at<Vec3b>(i, j)[2];
			count[clusters[pixel]]++;
		}
	}

	for (int i = 0; i < k; i++)
	{
		intensities[i][0] /= count[i];
		intensities[i][1] /= count[i];
		intensities[i][2] /= count[i];
	}

	Mat img(rows, cols, CV_8UC3);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			int pixel = i * cols + j;
			img.at<Vec3b>(i, j)[0] = intensities[clusters[pixel]][0];
			img.at<Vec3b>(i, j)[1] = intensities[clusters[pixel]][1];
			img.at<Vec3b>(i, j)[2] = intensities[clusters[pixel]][2];
		}
	}

	imshow("Color", img);
}

vector<Vec3b> random_colors(int k = 10)
{
	default_random_engine gen;
	uniform_int_distribution<int> distribution(0, 200);
	int rand_val = distribution(gen);

	vector<Vec3b> colors;
	for (int i = 0; i < k; i++)
		colors.push_back({ (uchar)distribution(gen),
		(uchar)distribution(gen),
		(uchar)distribution(gen) });

	return colors;
}

void clustering()
{
	vector<Vec3b> colors = random_colors();

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		int c;
		cout << "case = ";
		cin >> c;
		//c = 3;

		Mat source = imread(fname, IMREAD_COLOR);

		if (c == 1)
		{
			points_kmeans(source, colors);
		}
		else if (c == 2)
		{
			gray_kmeans(source);
		}
		else if (c == 3)
		{
			color_kmeans(source);
		}

		waitKey();
		destroyAllWindows();
	}

}

// Lab 8 - K-nearest Neighbors
vector<int> calcHist(Mat img, int nr_bins)
{
	vector<int> hist(nr_bins * 3);

	uchar* imgData = img.data;

	int d = 256 / nr_bins;
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			int pixel = (i * img.cols + j) * 3;

			int blue = imgData[pixel];
			int green = imgData[pixel + 1];
			int red = imgData[pixel + 2];

			hist[blue / d]++;
			hist[green / d + nr_bins]++;
			hist[red / d + nr_bins * 2]++;
		}
	}

	return hist;
}

double euclDist(vector<int> q, vector<int> p)
{
	double sum = 0;
	for (int i = 0; i < q.size(); i++)
	{
		sum += (q[i] - p[i]) * (q[i] - p[i]);
	}

	sum = sqrt(sum);

	return sum;
}

void kkn_classifier(vector<pair<vector<int>, int>> test, vector<pair<vector<int>, int>> train, int k, int nr_classes)
{
	vector<vector<double>> C(nr_classes, vector<double>(nr_classes));


	for (pair<vector<int>, int> te : test)
	{
		vector<pair<double, int>> distances;
		for (pair<vector<int>, int> tr : train)
		{
			double distance = euclDist(te.first, tr.first);
			distances.push_back({ distance, tr.second });
		}

		sort(distances.begin(), distances.end(), [](auto& left, auto& right) {
			return left.first < right.first;
			});

		vector<int>predictions(6, 0);

		for (int i = 0; i < k; i++)
		{
			predictions[distances[i].second]++;
			//C[distances[i].second][te.second]++;
		}

		int prediction = predictions[0];
		int predicted_class = 0;
		for (int i = 0; i < nr_classes; i++)
		{
			if (predictions[i] > prediction)
			{
				prediction = predictions[i];
				predicted_class = i;
			}
		}

		C[predicted_class][te.second]++;

	}

	double acc = 0, diagonal = 0, sum = 0;

	for (int i = 0; i < nr_classes; i++)
	{
		diagonal += C[i][i];
	}

	for (int i = 0; i < nr_classes; i++)
	{
		for (int j = 0; j < nr_classes; j++)
		{
			sum += C[i][j];
		}
	}

	acc = diagonal / sum;

	cout << "Accuracy = " << acc << "\n";
}

void neighbors()
{
	const int nrclasses = 6;
	char classes[nrclasses][10] =
	{ "beach", "city", "desert", "forest", "landscape", "snow" };

	char train_fldname[MAX_PATH], test_fldname[MAX_PATH];
	char fname[1000];
	strcpy(train_fldname, "D:/GitHub/SRF/OpenCVApplication-VS2019_OCV451_basic/OpenCVApplication-VS2019_OCV451_basic/Data/prs_res_KNN/train");
	strcpy(test_fldname, "D:/GitHub/SRF/OpenCVApplication-VS2019_OCV451_basic/OpenCVApplication-VS2019_OCV451_basic/Data/prs_res_KNN/test");
	//while (openFolderDlg(train_fldname))

	int nr_bins;
	cout << "\nBins: ";
	//cin >> nr_bins;
	nr_bins = 8;

	// train features 
	vector<pair<vector<int>, int>> train(0);

	int c = 0, fileNr = 0;
	while (true)
	{
		sprintf(fname, "%s/%s/%06d.jpeg", train_fldname, classes[c], fileNr++);
		Mat img = imread(fname);
		if (img.empty())
		{
			if (c > nrclasses)
				break;
			c++;
			fileNr = 0;
			continue;
		}

		vector<int> hist = calcHist(img, nr_bins);

		//showHistogram("hist", hist, 3 * nr_bins, 100);
		//waitKey();
		//destroyAllWindows();

		train.push_back({ hist, c });
	}

	// test features
	vector<pair<vector<int>, int>> test(0);

	c = 0, fileNr = 0;
	while (true)
	{
		sprintf(fname, "%s/%s/%06d.jpeg", test_fldname, classes[c], fileNr++);
		Mat img = imread(fname);
		if (img.empty())
		{
			if (c >= nrclasses)
				break;
			c++;
			fileNr = 0;
			continue;
		}

		vector<int> hist = calcHist(img, nr_bins);

		test.push_back({ hist, c });
	}

	// classify
	for (int i = 1; i < 3 * nr_bins; i++)
	{
		cout << i << ": ";
		kkn_classifier(test, train, i, nrclasses);
	}
}

// Lab 9 - Naive Bayesian Classifier
vector<Mat> load_mnist(string path, int classes = 8)
{
	int num_samples = 200;
	int img_size = 28;

	vector<Mat> features;

	for (int c = 0; c < classes; c++)
	{
		Mat feature(num_samples, img_size * img_size, CV_8UC1);
		for (int s = 0; s < num_samples; s++)
		{
			char img_path[500];
			sprintf(img_path, "%s/%d/%06d.png", path.c_str(), c, s);
			Mat img = imread(img_path, IMREAD_GRAYSCALE);
			if (img.cols == 0) break;

			for (int i = 0; i < img_size; i++)
			{
				for (int j = 0; j < img_size; j++)
				{
					int pixel = i * img_size + j;

					int feature_row = s * img_size * img_size;
					feature.data[feature_row + pixel] = img.data[pixel];
				}
			}
		}
		features.push_back(feature);
	}

	return features;
}

pair<Mat, vector<double>> naive_bayes_classifier(vector<Mat> features)
{
	// thresholding
	for (Mat feature : features)
	{
		threshold(feature, feature, 128, 255, THRESH_BINARY);
	}

	// priors
	int total_samples = 0;
	for (Mat feature : features)
		total_samples += feature.rows;

	vector<double> priors(features.size());
	for (int i = 0; i < features.size(); i++)
		priors[i] = ((double)features[i].rows) / total_samples;

	cout << "Priors: ";
	for (double value : priors)
		cout << value << " ";
	cout << '\n';

	// likelihood
	Mat likelihood(features.size(), features[0].cols, CV_64FC1, Scalar(1));

	for (int f = 0; f < features.size(); f++)
	{
		Mat feature = features[f];
		for (int col = 0; col < feature.cols; col++)
		{
			int hit = 0;
			for (int row = 0; row < feature.rows; row++)
			{
				int position = row * feature.cols + col;
				if (feature.data[position] == 255)
					hit++;
			}
			likelihood.at<double>(f, col) = ((double)hit) / feature.rows;
		}
	}

	return { likelihood, priors };
}

int classifyBayes(Mat img, vector<double> priors, Mat likelihood)
{
	threshold(img, img, 128, 255, THRESH_BINARY);
	Mat img_flat(1, 28 * 28, CV_8UC1);
	for (int i = 0; i < 28; i++)
	{
		for (int j = 0; j < 28; j++)
		{
			int pixel = i * 28 + j;

			img_flat.data[pixel] = img.data[pixel];
		}
	}

	Mat prob(likelihood.rows, 1, CV_64FC1);
	for (int c = 0; c < likelihood.rows; c++)
	{
		prob.at<double>(c, 0) = log(priors[c]);
		for (int col = 0; col < likelihood.cols; col++)
		{
			double lkh = likelihood.at<double>(c, col);
			prob.at<double>(c, 0) += img_flat.at<uchar>(0, col) == 255 ? log(lkh) : log(1 - lkh);
		}
	}

	int classP = 0;
	double cprob = cprob = prob.at<double>(0, 0);;
	for (int c = 0; c < prob.rows; c++)
	{
		if (prob.at<double>(c, 0) > cprob)
		{
			cprob = prob.at<double>(c, 0);
			classP = c;
		}
	}

	cout << classP << " with prob: " << cprob << '\n';

	return classP;
}

void mnist()
{
	string train_path, test_path;
	train_path = "D:/GitHub/SRF/OpenCVApplication-VS2019_OCV451_basic/OpenCVApplication-VS2019_OCV451_basic/Data/prs_res_Bayes/train";
	//test_path = "D:/GitHub/SRF/OpenCVApplication-VS2019_OCV451_basic/OpenCVApplication-VS2019_OCV451_basic/Data/prs_res_Bayes/test";
	//openFolderDlg(train_path)
	//openFolderDlg(test_path)

	vector<Mat>train_features = load_mnist(train_path);
	//vector<Mat>test_features = load_mnist(test_path);

	pair<Mat, vector<double>> classifier = naive_bayes_classifier(train_features);

	char fname[MAX_PATH];
	openFileDlg(fname);

	Mat test_img = imread(fname, IMREAD_GRAYSCALE);

	classifyBayes(test_img, classifier.second, classifier.first);

}

// Lab 10 - Linear Classifiers
Mat draw_line(Mat src, double a, double b, double c)
{
	Mat img = src.clone();
	// aX + bY + c = 0
	Point2f point0(0, (-a * 0 - c) / b);
	Point2f pointW(img.cols, (-a * img.cols - c) / b);

	line(img, point0, pointW, Scalar(0, 255, 0));
	return img;
}

void online_perceptron(vector<pair<Point, double>> X, vector<double>& w, double lr = 0.0001, double elimit = 0.00001, int max_iter = 10000)
{
	int n = X.size();
	for (int iter = 0; iter < max_iter; iter++)
	{
		double errors = 0;
		double z;

		for (int i = 0; i < n; i++)
		{
			z = w[0] * 1 +
				w[1] * X[i].first.x +
				w[2] * X[i].first.y;
			if (z * X[i].second < 0)
			{
				w[0] += lr * 1 * X[i].second;
				w[1] += lr * X[i].first.x * X[i].second;
				w[2] += lr * X[i].first.y * X[i].second;

				errors++;
			}
		}

		errors /= n;
		if (errors < elimit)
			break;
	}
}

void batch_perceptron(vector<pair<Point, double>> X, vector<double>& w, Mat img, double lr = 0.0001, double elimit = 0.000001, int max_iter = 10000)
{
	int n = X.size();
	for (int iter = 0; iter < max_iter; iter++)
	{
		double errors = 0;
		double loss = 0;
		double z;
		vector<double> gradient{ 0, 0, 0 };

		for (int i = 0; i < n; i++)
		{
			z = w[0] * 1 +
				w[1] * X[i].first.x +
				w[2] * X[i].first.y;
			if (z * X[i].second < 0)
			{
				gradient[0] -= 1 * X[i].second;
				gradient[1] -= X[i].first.x * X[i].second;
				gradient[2] -= X[i].first.y * X[i].second;

				loss -= z * X[i].second;
				errors++;
			}
		}

		errors /= n;
		loss /= n;

		gradient[0] /= n;
		gradient[1] /= n;
		gradient[2] /= n;

		if (errors < elimit)
			break;

		w[0] -= lr * 10 * gradient[0];
		w[1] -= lr * gradient[1];
		w[2] -= lr * gradient[2];

		cout << "Iter " << iter << " loss " << loss << '\n';
		if (iter % 100 == 0)
		{
			Mat preview = draw_line(img, w[2], w[1], w[0]);
			imshow("preview", preview);
			waitKey();
		}
	}
}

void linear_classifier()
{
	char fname[MAX_PATH];
	openFileDlg(fname);

	Mat img = imread(fname);

	vector<pair<Point, double>> data;

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			int pixel = (i * img.cols + j) * 3;
			if (img.data[pixel + 0] == 255 && img.data[pixel + 1] == 0 && img.data[pixel + 2] == 0) // blue
				data.push_back({ Point(i, j),  -1 });
			if (img.data[pixel + 0] == 0 && img.data[pixel + 1] == 0 && img.data[pixel + 2] == 255) // red
				data.push_back({ Point(i, j),  1 });
		}
	}

	vector<double> w1{ 0.1, 0.1, 0.1 };
	online_perceptron(data, w1);
	Mat online = draw_line(img, w1[2], w1[1], w1[0]);


	vector<double> w2{ 0.1, 0.1, 0.1 };
	batch_perceptron(data, w2, img);
	Mat batch = draw_line(img, w2[2], w2[1], w2[0]);

	imshow("img", img);
	imshow("online", online);
	imshow("batch", batch);

	waitKey();
}

//Lab 11
template <typename T>
using matrix = std::vector<std::vector<T>>;

struct WeakLearner
{
	int feature_idx;
	int threshold;
	int class_label;
	double error;

	WeakLearner() : feature_idx(0), threshold(0), class_label(0), error(0) {}

	WeakLearner(int feature_idx, int threshold, int class_label, double error)
		: feature_idx(feature_idx), threshold(threshold), class_label(class_label), error(error) {}


	int classify(const vector<int> feature) const
	{
		return feature[feature_idx] < threshold ? class_label : -class_label;
	}

	static WeakLearner findWeakLearner(
		const matrix<int> features,
		const vector<double> labels,
		const vector<double> weights,
		const Mat image)
	{
		WeakLearner bestWL;
		double bestError = DBL_MAX;

		for (int j = 0; j < 2; j++)
		{
			int img_size = (j == 0) ? image.rows : image.cols;

			for (int threshold = 0; threshold < img_size; threshold++)
			{
				for (int class_label : {-1, 1})
				{
					double error = 0;

					for (int i = 0; i < features.size(); i++)
					{
						int z = features[i][j] < threshold ? class_label : -class_label;
						if (z * labels[i] < 0)
							error += weights[i];
					}

					if (error < bestError)
					{
						bestError = error;
						bestWL = { j, threshold, class_label, error };
					}
				}
			}
		}

		return bestWL;
	}
};

class AdaBoostClassifier
{
	int T;
	vector<float> alphas;
	vector<WeakLearner> hs;
	matrix<int> features;
	vector<double> labels;
	Mat image;

	void AdaBoost()
	{
		int n = features.size();

		vector<double> weights(n, 1.0 / n);

		for (int t = 0; t < T; t++)
		{
			hs[t] = WeakLearner::findWeakLearner(features, labels, weights, image);
			alphas[t] = 0.5 * log((1.0 - hs[t].error) / hs[t].error);

			double sum = 0;
			for (int i = 0; i < n; i++)
			{
				weights[i] *= exp(-alphas[t] * labels[i] * hs[t].classify(features[i]));
				sum += weights[i];
			}

			for (int i = 0; i < n; i++)
				weights[i] /= sum;

		}
	}

public:

	AdaBoostClassifier(const int T, const matrix<int> features, const vector<double> labels, Mat image)
		: T(T), features(features), labels(labels), image(image)
	{
		alphas = vector<float>(T);
		hs = vector<WeakLearner>(T);
		AdaBoost();
	}

	int classify(const vector<int> feature) const
	{
		double sum = 0;

		for (int t = 0; t < T; t++)
		{
			sum += alphas[t] * hs[t].classify(feature);
		}

		return sum < 0 ? -1 : 1;
	}

	vector<float> getAlphas()
	{
		return alphas;
	}

	vector<WeakLearner> getHs()
	{
		return getHs();
	}
};


auto isColorMatch = [](const uchar* imageData, const Vec3b& color, const int pixelIndex) -> bool {
	return imageData[pixelIndex] == color[0]
		&& imageData[pixelIndex + 1] == color[1]
		&& imageData[pixelIndex + 2] == color[2];
};

auto assignColor = [](uchar* imageData, const Vec3b& color, const int pixelIndex) -> void {
	imageData[pixelIndex] = color[0];
	imageData[pixelIndex + 1] = color[1];
	imageData[pixelIndex + 2] = color[2];
};

Mat drawBoundary(const Mat source, const AdaBoostClassifier classifier)
{
	Mat image = source.clone();

	Vec3b yellow{ 0, 255, 255 };
	Vec3b cyan{ 255, 255, 0 };
	Vec3b white{ 255, 255, 255 };


	for (int row = 0; row < image.rows; row++)
	{
		for (int col = 0; col < image.cols; col++)
		{
			int pixelIndex = (row * image.cols + col) * 3;

			if (isColorMatch(image.data, white, pixelIndex))
			{
				vector<int> point{ row, col };
				int cls = classifier.classify(point);

				if (cls < 0)
					assignColor(image.data, yellow, pixelIndex);
				else
					assignColor(image.data, cyan, pixelIndex);
			}
		}

	}

	return image;
}

void adaBoost()
{
	char filename[MAX_PATH];
	openFileDlg(filename);

	Mat image = imread(filename);

	matrix<int> features;
	vector<double> labels;

	Vec3b red{ 0, 0, 255 };
	Vec3b blue{ 255, 0, 0 };

	for (int row = 0; row < image.rows; row++)
	{
		for (int col = 0; col < image.cols; col++)
		{
			int pixelIndex = (row * image.cols + col) * 3;

			if (isColorMatch(image.data, blue, pixelIndex))
			{
				features.push_back(vector<int>{row, col});
				labels.push_back({ -1 });
			}

			if (isColorMatch(image.data, red, pixelIndex))
			{
				features.push_back(vector<int>{row, col});
				labels.push_back({ 1 });
			}
		}
	}

	AdaBoostClassifier adaBoostClassifier1(1, features, labels, image);
	AdaBoostClassifier adaBoostClassifier13(13, features, labels, image);

	Mat adaBoost1 = drawBoundary(image, adaBoostClassifier1);
	Mat adaBoost13 = drawBoundary(image, adaBoostClassifier13);

	imshow("image", image);
	imshow("adaBoost1", adaBoost1);
	imshow("adaBoost13", adaBoost13);
	waitKey();
	destroyAllWindows();
}

int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		cout << "Menu:\n";

		cout << " Lab01--------------------------------\n";
		cout << "\t 1 - Regression\n";

		cout << " Lab02--------------------------------\n";
		cout << "\t 2 - RANSAC\n";

		cout << " Lab03--------------------------------\n";
		cout << "\t 3 - Hough\n";

		cout << " Lab04--------------------------------\n";
		cout << "\t 4 - Distance transform\n";

		cout << " Lab05--------------------------------\n";
		cout << "\t 5 - Statistical Data Analysis\n";

		cout << " Lab06--------------------------------\n";
		cout << "\t 6 - Principal Component Analysis\n";

		cout << " Lab07--------------------------------\n";
		cout << "\t 7 - K-means Clustering\n";

		cout << " Lab08--------------------------------\n";
		cout << "\t 8 - K-nearest Neighbors\n";

		cout << " Lab09--------------------------------\n";
		cout << "\t 9 - Naive Bayesian Classifier\n";

		cout << " Lab10--------------------------------\n";
		cout << "\t 10 - Linear Classifiers\n";

		cout << " Lab11--------------------------------\n";
		cout << "\t 11 - AdaBoost\n";

		cout << "\n0 - Exit\n\n";

		cout << "Option: ";
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
		case 5:
			statistical_analysis();
		case 6:
			pca();
		case 7:
			clustering();
		case 8:
			neighbors();
		case 9:
			mnist();
		case 10:
			linear_classifier();
		case 11:
			adaBoost();
		}
		//op = 0;
	} while (op != 0);
	return 0;
}