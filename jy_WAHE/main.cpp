#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

Mat f_WAHE(Mat img, float g, int threshold);
//parameters of function f_WAHE:
//    img      : Input image
//    dst      : Output image
//    g        : Level of enhancement
//    threshold: 



void main()
{
	Mat in_img = imread("palermo.jpg",0);
	Mat gray;
	Mat out_img;
	int threshold = 2;

	//gray = cvtColor(in_img, gray, COLOR_RGB2GRAY)
	out_img=f_WAHE(in_img, 1.0, threshold);

	namedWindow("input image", CV_WINDOW_AUTOSIZE);
	imshow("input image", in_img);
	namedWindow("output image", CV_WINDOW_AUTOSIZE);
	imshow("output image", out_img);


	waitKey(0);

}

Mat f_WAHE(Mat img, float g, int threshold)
{
	int row, col;
	row = img.rows;
	col = img.cols;

	int kappa = 0;  //initialize kappa
	double count = 0;  //initialize count
	double gamma = 100;  //initialize smoothing parameter gamma

	int h[256] = { 0 };  //initialize histogram

	for (int m = 0; m < row; m++)
	{
		for (int n = 2; n < col; n++)
		{
			int temp = 0;

			temp = img.at<uchar>(m, n) - img.at<uchar>(m, n - 2);
			kappa = kappa + abs(temp);
			if (abs(temp) > threshold)
			{ 


				h[img.at<uchar>(m, n)]++;
				count++;
			}
		}
	}//get histogram

	kappa /= row*col;
	float kappa_star;
	//kappa_star = 1 / (1 + g*kappa);
	kappa_star = 1 / (1 + g);
	float u;
	u = count / 256;    //uniform


	////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////  B&W stretch  /////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////





	////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////  WAHE h_tilda  /////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////
   
	//WAHE
	int h_tilda1[256] = { 0 };
	int h_sum1 = 0;

	////get WAHE histogram
	for (int n = 0; n < 256; n++)
	{
		h_tilda1[n]=round( (1 - kappa_star)*u + kappa_star * h[n]);
		h_sum1 = h_sum1 + h_tilda1[n];
	
	}


	////////////////////////////////////////////////////////////////////////////////
	////////////////////////////  get smoothed histogram  //////////////////////////
	////////////////////////////////////////////////////////////////////////////////
	
	//smoothing
	int h_tilda2[256] = { 0 };
	int h_sum2 = 0;

	//create difference matrix D
	Mat D(255,256, CV_32FC1);
	for (int m = 0; m < 255; m++)
	{
		for (int n = 0; n < 256; n++)
		{
			if( m == n )
				D.at<float>(m, n) = -1;
			else
			{
				if (n == (m + 1))
					D.at<float>(m, n) = 1;
				else
					D.at<float>(m, n) = 0;
			}
		}
	}
	//cout << D << endl;


	//get transpose matrix of D
	Mat Dt(256, 255, CV_32FC1);
	transpose(D, Dt);
	//cout << Dt<<endl;

    //cout << Dt.at<float>(255, 254) << endl;
	Mat smooth(256, 256, CV_32FC1);
	smooth =gamma*Dt*D;
	//cout << smooth << endl;
	for (int m = 0; m < 256; m++)
	{
		smooth.at<float>(m, m) = smooth.at<float>(m, m) + 1 + g;
	}
	//cout << smooth << endl;

	//smoothed h_tilda
	Mat smooth_inv(256, 256, CV_32FC1);
	smooth_inv = smooth.inv();
	//cout << smooth_inv << endl;
	for (int m = 0; m < 256; m++)
	{
		for (int n = 0; n < 256; n++)
		{
			h_tilda2[m] += smooth_inv.at<float>(m, n)*(h[n] + g*u);
		}
		h_sum2 = h_sum2 + h_tilda2[m];
	}


	////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////  draw histograms  /////////////////////////////
	////////////////////////////////////////////////////////////////////////////////

	const int h_max = 550;
	const int domain = 5;
	Mat histImage(h_max, 256, CV_8UC3, Scalar(0, 0, 0));
	for (int n = 1; n < 256; n++)
	{
		//original histogram
		line(histImage, Point(n - 1, h_max - h[n-1]/ domain), Point(n, h_max - h[n]/ domain), Scalar(255, 0, 0), 2, 8, 0) ;

		//WAHE histogram
		line(histImage, Point(n - 1, h_max - h_tilda1[n - 1]/ domain), Point(n, h_max - h_tilda1[n]/ domain), Scalar(0, 255, 0), 2, 8, 0);

		//smoothed histogram
		line(histImage, Point(n - 1, h_max - h_tilda2[n - 1]/ domain), Point(n, h_max - h_tilda2[n]/ domain), Scalar(0, 0, 255), 2, 8, 0);
	}
	namedWindow("histogram", CV_WINDOW_AUTOSIZE);
	imshow("histogram", histImage);


	////////////////////////////////////////////////////////////////////////////////
	///////////////////////  reconstruct transformation function  //////////////////
	////////////////////////////////////////////////////////////////////////////////

	////get pdf by WAHE
	//float pdf[256];
	//for (int n = 0; n < 256; n++)
	//{
	//	pdf[n] = float(h_tilda1[n]) / float(h_sum1);
	//}

	//get pdf by smoothing method
	float pdf[256];
	for (int n = 0; n < 256; n++)
	{
		pdf[n] = float(h_tilda2[n]) / float(h_sum2);
	}

	//get cdf
	float cdf[256];
	cdf[0] = pdf[0];
	for (int n = 1; n < 256; n++)
	{
		cdf[n] = cdf[n - 1] + pdf[n];
	}

	//get transformation function
	int transform[256];
	for (int n = 0; n < 256; n++)
	{
		transform[n] = round(255 * cdf[n]);
	}
	//for (int m = 0; m < 256; m++)
	//{
	//	cout << transform[m] << endl;
	//}

	////////////////////////////////////////////////////////////////////////////////
	/////////////////////////  reconstruct image  //////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////

	Mat dst=img.clone();
	int cunt = 0;
	for (int m = 0; m < row; m++)
	{
		for (int n = 0; n < col; n++)
		{
			int temp;
			temp = img.at<uchar>(m, n);


			dst.at<uchar>(m, n) = transform[temp];
			cunt = cunt + 1;
		}
	}

	//uchar* pxvec = dst.ptr<uchar>(0);
	//for (int m = 0; m < row; m++)
	//{
	//	pxvec = dst.ptr<uchar>(m);
	//	for (int n = 0; n < col; n++)
	//	{
	//		int temp;
	//		temp = img.at<uchar>(m, n);
	//		pxvec[n] = transform[temp];
	//	}
	//}

	return dst;

}