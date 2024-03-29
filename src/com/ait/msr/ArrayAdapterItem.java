package com.ait.msr;

import org.opencv.android.Utils;

import android.app.Activity;
import android.content.Context;
import android.graphics.Bitmap;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.ImageView;
import android.widget.TextView;

// here's our beautiful adapter
public class ArrayAdapterItem extends ArrayAdapter<ObjectItem> {

    Context mContext;
    int layoutResourceId;
    ObjectItem data[] = null;

    public ArrayAdapterItem(Context mContext, int layoutResourceId, ObjectItem[] data) {

        super(mContext, layoutResourceId, data);

        this.layoutResourceId = layoutResourceId;
        this.mContext = mContext;
        this.data = data;
    }

    @Override
    public View getView(int position, View convertView, ViewGroup parent) {

        /*
         * The convertView argument is essentially a "ScrapView" as described is Lucas post 
         * http://lucasr.org/2012/04/05/performance-tips-for-androids-listview/
         * It will have a non-null value when ListView is asking you recycle the row layout. 
         * So, when convertView is not null, you should simply update its contents instead of inflating a new row layout.
         */
        if(convertView==null){
            // inflate the layout
            LayoutInflater inflater = ((Activity) mContext).getLayoutInflater();
            convertView = inflater.inflate(layoutResourceId, parent, false);
        }

        // object item based on the position
        ObjectItem objectItem = data[position];

        // get the TextView and then set the text (item name) and tag (item ID) values
        TextView textViewItem = (TextView) convertView.findViewById(R.id.textViewItem);
        textViewItem.setText(objectItem.itemName);
        
        ImageView imgViewItem = (ImageView) convertView.findViewById(R.id.ImageView);
        Bitmap imgBMP = Bitmap.createBitmap(objectItem.img_thumb.cols(), objectItem.img_thumb.rows(),Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(objectItem.img_thumb, imgBMP);
        imgViewItem.setImageBitmap(imgBMP);
        
        TextView textViewValue = (TextView) convertView.findViewById(R.id.ValueItem);
        textViewValue.setText(String.format( "%.2f", objectItem.value ));
        
        textViewItem.setTag(objectItem.itemId);

        return convertView;

    }

}