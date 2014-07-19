package com.ait.msr;

import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

public class ObjectItem implements Comparable<ObjectItem>{ 
	public int itemId;
	public String itemName;
	public Mat img;
	public Mat img_thumb;
	public double value;
	private static final String TAG = "AIT::MSR::ObjectItem";
	// constructor
	public ObjectItem(int itemId, String itemName, Mat img) {
		this.itemId = itemId;
		this.itemName = itemName;
		this.img = img;
		this.img_thumb = new Mat();
		Imgproc.resize(img, this.img_thumb, new Size(img.cols()*0.2,img.rows()*0.2));
		this.img = MatchShape.cropImage(this.img);
		this.value = 0.0;
//		Log.d(TAG,"Set Object filename : " + this.itemName);
	}
	
	public int compareTo(ObjectItem compareItem) {
 
		//ascending order
		double result = (this.value - compareItem.value);
    	if(result > -1 && result < 1) 
    		result = 1/result;
        return  (int) result;
        
		//descending order
		//return compareQuantity - this.quantity;
 
	}

}
