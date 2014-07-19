package com.ait.msr;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.highgui.Highgui;

import android.content.Context;
import android.content.res.AssetManager;
import android.util.Log;


public class Utils extends org.opencv.android.Utils{
	
	private static final String TAG = "AIT::MSR::Utils";
	public static Mat loadResourceFromFilePath(Context context, String filepath, int flags) throws IOException
    {
		InputStream bitmap=null;
		AssetManager assetManager = context.getAssets();
		bitmap=assetManager.open(filepath);
        ByteArrayOutputStream os = new ByteArrayOutputStream(bitmap.available());

        byte[] buffer = new byte[4096];
        int bytesRead;
        while ((bytesRead = bitmap.read(buffer)) != -1) {
            os.write(buffer, 0, bytesRead);
        }
        bitmap.close();

        Mat encoded = new Mat(1, os.size(), CvType.CV_8U);
        encoded.put(0, 0, os.toByteArray());
        os.close();

        Mat decoded = Highgui.imdecode(encoded, flags);
        encoded.release();

        return decoded;
    }
}
