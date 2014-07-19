package com.ait.msr;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.features2d.DMatch;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

import android.util.Log;

public class MatchShape {
	private static final String TAG = "AIT::MSR::MatchShape";

	public double match(String refImgPath, String drawImgPath) {
		Mat refImg = Highgui.imread(refImgPath, CvType.CV_8UC1);
		Mat drawImg = Highgui.imread(drawImgPath, CvType.CV_8UC1);

		return match(refImg, drawImg, 3);
	}

	public double match(Mat refImg, Mat drawImg, int method) {

		Imgproc.resize(drawImg, drawImg,
				new Size(refImg.width(), refImg.height()));
		// thresholding the gray scale image to get better results
		Imgproc.threshold(refImg, refImg, 127, 255, Imgproc.THRESH_BINARY_INV
				+ Imgproc.THRESH_OTSU);
		Imgproc.threshold(drawImg, drawImg, 127, 255, Imgproc.THRESH_BINARY_INV
				+ Imgproc.THRESH_OTSU);
		switch (method) {
		case 0:
			return this.imageSubtraction(refImg, drawImg);
		case 1:
			return this.NormL2(drawImg, refImg);
		case 2:
			return this.getPSNR(refImg, drawImg);
		case 3:
			return this.getMSSIM(refImg, drawImg);
		case 4:
			return this.surfMatch(refImg, drawImg);
		case 5:
			return this.countourMatch(refImg, drawImg);
		case 6:
			return this.Hausdorff_distance(refImg, drawImg);
		default:
			return this.getPSNR(refImg, drawImg);

		}
	}

	// Crop image to fit object size
	public static Mat cropImage(Mat image) {
		Mat imgThresholded = new Mat();
		Rect boundBox, maxBox;
		List<MatOfPoint> contours = new ArrayList<MatOfPoint>();

		Imgproc.threshold(image, imgThresholded, 127, 255,
				Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU);
		Imgproc.findContours(imgThresholded, contours, new Mat(),
				Imgproc.RETR_CCOMP, Imgproc.CHAIN_APPROX_SIMPLE);
		maxBox = new Rect(0, 0, 0, 0);
		for (MatOfPoint cnt : contours) {
			boundBox = Imgproc.boundingRect(cnt);
			if (boundBox.area() > maxBox.area())
				maxBox = boundBox;
		}
		return image.submat(maxBox);
	}

	// Compare two images by getting the L2 error (square-root of sum of squared
	// error).
	private double NormL2(Mat A, Mat B) {
		if (A.rows() > 0 && A.rows() == B.rows() && A.cols() > 0
				&& A.cols() == B.cols()) {
			// Calculate the L2 relative error between images.
			double errorL2 = Core.norm(A, B, Core.NORM_L2);
			// Convert to a reasonable scale, since L2 error is summed across
			// all pixels of the image.
			double similarity = errorL2 / (double) (A.rows() * A.cols());
			return similarity;
		} else {
			// Images have a different size
			return 1000.0; // Return a bad value
		}
	}

	private double getPSNR(Mat I1, Mat I2) {
		Mat s1 = new Mat();
		Core.absdiff(I1, I2, s1); // |I1 - I2|
		s1.convertTo(s1, CvType.CV_32F); // cannot make a square on 8 bits
		s1 = s1.mul(s1); // |I1 - I2|^2

		Scalar s = Core.sumElems(s1); // sum elements per channel

		double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels

		if (sse <= 1e-10) // for small values return zero
			return 0;
		else {
			double mse = sse / (double) (I1.channels() * I1.total());
			double psnr = 10.0 * Math.log10((255 * 255) / mse);
			return 1 / psnr;
		}
	}

	private double getMSSIM(Mat I1, Mat I2) {
		double C1 = 6.5025, C2 = 58.5225;
		/***************************** INITS **********************************/
		int d = CvType.CV_32F;

		// Mat I1 = new Mat(), I2 = new Mat();
		I1.convertTo(I1, d); // cannot calculate on one byte large values
		I2.convertTo(I2, d);

		Mat I2_2 = I2.mul(I2); // I2^2
		Mat I1_2 = I1.mul(I1); // I1^2
		Mat I1_I2 = I1.mul(I2); // I1 * I2

		/*************************** END INITS **********************************/

		Mat mu1 = new Mat(), mu2 = new Mat(); // PRELIMINARY COMPUTING
		Imgproc.GaussianBlur(I1, mu1, new Size(11, 11), 1.5);
		Imgproc.GaussianBlur(I2, mu2, new Size(11, 11), 1.5);

		Mat mu1_2 = mu1.mul(mu1);
		Mat mu2_2 = mu2.mul(mu2);
		Mat mu1_mu2 = mu1.mul(mu2);

		Mat sigma1_2 = new Mat(), sigma2_2 = new Mat(), sigma12 = new Mat();

		Imgproc.GaussianBlur(I1_2, sigma1_2, new Size(11, 11), 1.5);
		Core.subtract(sigma1_2, mu1_2, sigma1_2);

		Imgproc.GaussianBlur(I2_2, sigma2_2, new Size(11, 11), 1.5);
		Core.subtract(sigma2_2, mu2_2, sigma2_2);

		Imgproc.GaussianBlur(I1_I2, sigma12, new Size(11, 11), 1.5);
		Core.subtract(sigma12, mu1_mu2, sigma12);

		// /////////////////////////////// FORMULA
		// ////////////////////////////////
		Mat t1 = new Mat(), t2 = new Mat(), t3 = new Mat();

		Core.multiply(mu1_mu2, new Scalar(2), t1);
		Core.add(t1, new Scalar(C1), t1);

		Core.multiply(sigma12, new Scalar(2), t2);
		Core.add(t2, new Scalar(C2), t2);

		t3 = t1.mul(t2); // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

		Core.add(mu1_2, mu2_2, t1);
		Core.add(t1, new Scalar(C1), t1);

		Core.add(sigma1_2, sigma2_2, t2);
		Core.add(t2, new Scalar(C2), t2);

		t1 = t1.mul(t2); // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 +
							// C2))

		Mat ssim_map = new Mat();
		Core.divide(t3, t1, ssim_map); // ssim_map = t3./t1;

		Scalar mssim = Core.mean(ssim_map); // mssim = average of ssim map
		return 1 / mssim.val[0];
	}

	private double surfMatch(Mat I1, Mat I2) {
		MatOfKeyPoint I1Keypoints = new MatOfKeyPoint();
		MatOfKeyPoint I2Keypoints = new MatOfKeyPoint();
		FeatureDetector surf = FeatureDetector.create(FeatureDetector.SURF);
		surf.detect(I1, I1Keypoints);
		surf.detect(I2, I2Keypoints);

		DescriptorExtractor extractor = DescriptorExtractor
				.create(DescriptorExtractor.SURF);
		Mat I1Descriptors = new Mat(), I2Descriptors = new Mat();
		extractor.compute(I1, I1Keypoints, I1Descriptors);
		extractor.compute(I2, I2Keypoints, I2Descriptors);

		DescriptorMatcher matcher = DescriptorMatcher
				.create(DescriptorMatcher.BRUTEFORCE);
		MatOfDMatch matches = new MatOfDMatch();

		matcher.match(I1Descriptors, I2Descriptors, matches);
		double max_dist = 0;
		double min_dist = 10;

		// -- Quick calculation of max and min distances between keypoints
		List<DMatch> matches2 = matches.toList();
		for (int i = 0; i < matches2.size(); i++) {
			double dist = matches2.get(i).distance;
			if (dist < min_dist)
				min_dist = dist;
			if (dist > max_dist)
				max_dist = dist;
		}
		
		List<DMatch> good_matches = new ArrayList<DMatch>();

		for (int i = 0; i < I1Descriptors.rows(); i++) {
			// 50% Threshold
			if (matches2.get(i).distance <= (max_dist - min_dist) / 2) {
				good_matches.add(matches2.get(i));
			}
		}
		
		return 1 / ((double) good_matches.size() / (double) matches2.size());
	}


	private double imageSubtraction(Mat I1, Mat I2) {
		Mat endResult = new Mat();
		Mat endResult2 = new Mat();

		Core.subtract(I1, I2, endResult);
		Core.subtract(I2, I1, endResult2);
		// make negative to zero
		Imgproc.threshold(endResult, endResult, 0, 255, Imgproc.THRESH_BINARY);
		Imgproc.threshold(endResult2, endResult2, 0, 255, Imgproc.THRESH_BINARY);

		return (Core.sumElems(endResult).val[0] / Core.sumElems(I1).val[0] + Core
				.sumElems(endResult2).val[0] / Core.sumElems(I2).val[0]) / 2;
	}

	private double countourMatch(Mat I1, Mat I2) {
//		Canny edge detection not help much + slow process
//		 int thresh = 100;
//		 Imgproc.Canny(I1, I1, thresh, thresh * 2);
//		 Imgproc.Canny(I2, I2, thresh, thresh * 2);
		List<MatOfPoint> I1Contours = new ArrayList<MatOfPoint>();
		List<MatOfPoint> I2Contours = new ArrayList<MatOfPoint>();

		//find only parent contours and only corner point of image
		Imgproc.findContours(I1, I1Contours, new Mat(), Imgproc.RETR_EXTERNAL,
				Imgproc.CHAIN_APPROX_SIMPLE);
		Imgproc.findContours(I2, I2Contours, new Mat(), Imgproc.RETR_EXTERNAL,
				Imgproc.CHAIN_APPROX_SIMPLE);

		// Compare only largest
		return Imgproc.matchShapes(
				I2Contours.get(findLargestContourIndex(I2Contours)),
				I1Contours.get(findLargestContourIndex(I1Contours)),
				Imgproc.CV_CONTOURS_MATCH_I1, 0.0) * 100;
	}

	private int findLargestContourIndex(List<MatOfPoint> contours) {
		double largest_area = 0;
		int largest_contour_index = 0;
		for (int i = 0; i < contours.size(); i++) {// iterate through each
													// contour.
			double a;
			a = Imgproc.contourArea(contours.get(i), false); // Find the area of
																// contour
			if (a > largest_area) {
				largest_area = a;
				largest_contour_index = i; // Store the index of largest contour
			}
		}

		return largest_contour_index;
	}
	
	private double Hausdorff_distance(Mat I1, Mat I2) {
		List<MatOfPoint> I1Contours = new ArrayList<MatOfPoint>();
		List<MatOfPoint> I2Contours = new ArrayList<MatOfPoint>();

//		Imgproc.Canny(I1, I1, 100, 200);
//		Imgproc.Canny(I2, I2, 100, 200);
		
		Imgproc.findContours(I1, I1Contours, new Mat(), Imgproc.RETR_TREE,
				Imgproc.CHAIN_APPROX_NONE);
		Imgproc.findContours(I2, I2Contours, new Mat(), Imgproc.RETR_TREE,
				Imgproc.CHAIN_APPROX_NONE);

//		double maxI1_2 = 0, maxI2_1 = 0;
		double maxI1_2 = Double.NEGATIVE_INFINITY, maxI2_1 = Double.NEGATIVE_INFINITY;
		Point[] Largest_I1Contours = I1Contours
				.get(findLargestContourIndex(I1Contours)).toArray();
		Point[] Largest_I2Contours = I2Contours
				.get(findLargestContourIndex(I2Contours)).toArray();
		
		//it too large for smart phone application so use sampling data should be better
		Largest_I1Contours = pickSample(Largest_I1Contours,(int)(Largest_I1Contours.length*0.1),new Random());
		Largest_I2Contours = pickSample(Largest_I2Contours,(int)(Largest_I2Contours.length*0.1),new Random());
		
		double temp;
		// I1 supremum infimum I1 to I2
		for (Point x : Largest_I1Contours) {
			temp = pointDist(x, Largest_I2Contours);
//				maxI1_2 += temp;
			if (maxI1_2 < temp)
				maxI1_2 = temp;
		}
		// I1 supremum infimum I2 to I1
		for (Point x : Largest_I2Contours) {
			temp = pointDist(x, Largest_I1Contours);
//			maxI2_1 += temp;
			if (maxI2_1 < temp)
				maxI2_1 = temp;
		}
		return Math.max(maxI1_2, maxI2_1);
	}

	/*
	 * Finds minimum distance between contour and pt
	 */
	private double pointDist(Point pt, Point[] contour) {
		double distance = Imgproc.pointPolygonTest(new MatOfPoint2f(contour),
				pt, true);
		return Math.abs(distance);
	}
	
	public static <T> T[] pickSample(T[] population, int nSamplesNeeded, Random r) {
		  T[] ret = (T[]) Array.newInstance(population.getClass().getComponentType(),
		                                    nSamplesNeeded);
		  int nPicked = 0, i = 0, nLeft = population.length;
		  while (nSamplesNeeded > 0) {
		    int rand = r.nextInt(nLeft);
		    if (rand < nSamplesNeeded) {
		      ret[nPicked++] = population[i];
		      nSamplesNeeded--;
		    }
		    nLeft--;
		    i++;
		  }
		  return ret;
		}
}
