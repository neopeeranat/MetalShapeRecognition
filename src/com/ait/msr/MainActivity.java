package com.ait.msr;

import java.util.Arrays;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.app.AlertDialog;
import android.app.ProgressDialog;
import android.content.DialogInterface;
import android.content.DialogInterface.OnCancelListener;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.AsyncTask;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.AdapterView;
import android.widget.AdapterView.OnItemSelectedListener;
import android.widget.ListView;
import android.widget.Spinner;
import android.widget.Toast;

public class MainActivity extends Activity {
	private static final String TAG = "AIT::MSR::Activity";

	private DrawingView drawView;
	private float smallBrush, mediumBrush, largeBrush;
	private Spinner method_selector;
	AlertDialog alertDialogStores;

	AssetManager assetManager;
	ObjectItem[] refImageData;

	MatchShape matcher = new MatchShape();

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		Log.i(TAG, "called onCreate");
		super.onCreate(savedInstanceState);
		getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
		setContentView(R.layout.activity_main);

		// / Drawing Part ///
		drawView = (DrawingView) findViewById(R.id.drawing);

		smallBrush = getResources().getInteger(R.integer.small_size);
		mediumBrush = getResources().getInteger(R.integer.medium_size);
		largeBrush = getResources().getInteger(R.integer.large_size);
		
		method_selector = (Spinner) findViewById(R.id.spinner1);
		method_selector.setSelection(2);
		method_selector.setOnItemSelectedListener(new OnItemSelectedListener()
				{ public void onItemSelected(AdapterView<?> adapter, View view, 
						int position, long id) {
			        // An item was selected. You can retrieve the selected item using
					String item = adapter.getItemAtPosition(position).toString();
					Toast.makeText(getApplicationContext(),
	                        "Selected Method : " + item, Toast.LENGTH_LONG).show();
			    }

				@Override
				public void onNothingSelected(AdapterView<?> parent) {
					// TODO Auto-generated method stub
					
				}} );

	}

	private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
		@Override
		public void onManagerConnected(int status) {
			switch (status) {
			case LoaderCallbackInterface.SUCCESS: {
				Log.i(TAG, "OpenCV loaded successfully");
				setUpRefData();
			}
				break;
			default: {
				super.onManagerConnected(status);
			}
				break;
			}
		}
	};

	@Override
	public void onResume() {
		super.onResume();
		Log.i(TAG, "called onResume");
		OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_9, this,
				mLoaderCallback);
	}

	public void setUpRefData() {
		assetManager = this.getAssets();
		final ProgressDialog ringProgressDialog = ProgressDialog.show(
				MainActivity.this, "Please wait ...",
				"Loading Reference Image ...", true);
		ringProgressDialog.setCancelable(false);
		new Thread(new Runnable() {
			@Override
			public void run() {
				try {
					String[] files = assetManager.list("ref_data");
					refImageData = new ObjectItem[files.length];
					for (int i = 0; i < files.length; i++) {
						refImageData[i] = new ObjectItem(i,
								files[i].toString(),
								Utils.loadResourceFromFilePath(
										MainActivity.this, "ref_data/"
												+ files[i], CvType.CV_8UC1));
					}
				} catch (Exception e) {

				}
				ringProgressDialog.dismiss();
			}
		}).start();

	}
	

	public void showPopUp(ObjectItem[] itemList) {
		// our adapter instance
		ArrayAdapterItem adapter = new ArrayAdapterItem(this,
				R.layout.list_view_row_item, itemList);

		// create a new ListView, set the adapter and item click listener
		ListView listViewItems = new ListView(this);
		listViewItems.setAdapter(adapter);
		listViewItems
				.setOnItemClickListener(new OnItemClickListenerListViewItem());

		// put the ListView in the pop up
		
		alertDialogStores = new AlertDialog.Builder(MainActivity.this)
				.setView(listViewItems).setTitle("Results : " + this.getResources().getStringArray(R.array.method)[ method_selector.getSelectedItemPosition()]).show();
	}

	public void onNewClicked(View view) {
		// new button
		AlertDialog.Builder newDialog = new AlertDialog.Builder(this);
		newDialog.setTitle("New drawing");
		newDialog
				.setMessage("Start new drawing (you will lose the current drawing)?");
		newDialog.setPositiveButton("Yes",
				new DialogInterface.OnClickListener() {
					public void onClick(DialogInterface dialog, int which) {
						drawView.startNew();
						drawView.setColor("#FF000000");
						dialog.dismiss();
					}
				});
		newDialog.setNegativeButton("Cancel",
				new DialogInterface.OnClickListener() {
					public void onClick(DialogInterface dialog, int which) {
						dialog.cancel();
					}
				});
		newDialog.show();
	}

	public void onDrawClicked(View view) {
		drawView.setColor("#FF000000");
	}

	public void onEraseClicked(View view) {
		drawView.setColor("#FFFFFFFF");
	}


	String time;

	public void onSearchClicked(View view) {
		AsyncTask<Void, Integer, ObjectItem[]> task;
		
		task = new AsyncTask<Void, Integer, ObjectItem[]>() {
			ProgressDialog pd;

			@Override
			protected void onPreExecute() {
				super.onPreExecute();
				pd = new ProgressDialog(MainActivity.this);
				pd.setTitle("Please wait ...");
				pd.setMessage("Processing...");
				pd.setCancelable(true);
				pd.setIndeterminate(true);
				pd.setOnCancelListener(new OnCancelListener() {
					@Override
					public void onCancel(DialogInterface dialog) {
						Log.d(TAG, "Progress Dialog cancelled.");
						cancel(true);
						if (pd != null) {
							pd.dismiss();
						}
					}
				});
				pd.show();
			}

			@SuppressLint("DefaultLocale")
			@Override
			protected ObjectItem[] doInBackground(Void... params) {
				float elapsedTimeSec = 0;
				ObjectItem[] fortestObjects = Arrays.copyOf(refImageData,
						refImageData.length);
				try {
					// Matching
					drawView.setDrawingCacheEnabled(true);
					drawView.buildDrawingCache(true);
					Bitmap drawImg_bitmap = Bitmap.createBitmap(drawView
							.getDrawingCache());

					Mat dragImg = new Mat();
					Utils.bitmapToMat(drawImg_bitmap, dragImg);
					Imgproc.cvtColor(dragImg, dragImg, Imgproc.COLOR_RGB2GRAY);
					dragImg = MatchShape.cropImage(dragImg);
					// drawView.destroyDrawingCache();
					drawView.setDrawingCacheEnabled(false);
					// Count function taking time
					long start = System.currentTimeMillis();
					for (int i = 0; i < fortestObjects.length; i++) {
						
						this.publishProgress(i + 1);
						//Log.d(TAG,"test");
						fortestObjects[i].value = matcher.match(
								fortestObjects[i].img, dragImg, method_selector.getSelectedItemPosition());
//						Log.d(TAG,"Loop : " + i 
//								+ ", value = " + String.format("%.2f", fortestObjects[i].value) 
//								+ ", image name = " + fortestObjects[i].itemName);
						if (isCancelled())
							break;
					}
					long elapsedTimeMillis = System.currentTimeMillis() - start;
					elapsedTimeSec = elapsedTimeMillis / 1000F;
					time = String.format("%.2f", elapsedTimeSec);
					Log.i(TAG,
							"Matching total time : "
									+ Float.toString(elapsedTimeSec) + " Sec");
					Arrays.sort(fortestObjects);
				} catch (Exception e) {
					e.printStackTrace();
				}
				return fortestObjects;
			}

			@Override
			protected void onPostExecute(ObjectItem[] result) {
				super.onPreExecute();
				if (pd != null) {
					pd.dismiss();
					showPopUp(result);
					Toast countTImeToast = Toast.makeText(
							getApplicationContext(), "Matching total time : "
									+ time + " Sec", Toast.LENGTH_LONG);
					countTImeToast.show();
				}
			}

			protected void onProgressUpdate(Integer... progress) {
				super.onProgressUpdate(progress);
				pd.setMessage("Processing...    " + (progress[0]) + " of "
						+ refImageData.length);
			}

		};
		task.execute();
	}
}
