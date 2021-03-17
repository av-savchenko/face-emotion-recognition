package com.asav.facialprocessing.mtcnn;

import android.graphics.Point;
import android.graphics.Rect;

import static java.lang.Math.max;
import static java.lang.Math.min;

public class Box {
    public  int[] box;       //left:box[0],top:box[1],right:box[2],bottom:box[3]
    public  float score;    //probability
    public  float[] bbr;    //bounding box regression
    public  boolean deleted;
    public  Point[] landmark; //facial landmark.只有ONet输出Landmark
    Box(){
        box=new int[4];
        bbr=new float[4];
        deleted=false;
        landmark=new Point[5];
    }
    public int left(){return box[0];}
    public int right(){return box[2];}
    public int top(){return box[1];}
    public int bottom(){return box[3];}
    public int width(){return box[2]-box[0]+1;}
    public int height(){return box[3]-box[1]+1;}

    public Rect transform2Rect(){
        Rect rect=new Rect();
        rect.left=Math.round(box[0]);
        rect.top=Math.round(box[1]);
        rect.right=Math.round(box[2]);
        rect.bottom=Math.round(box[3]);
        return  rect;
    }

    public  int area(){
        return width()*height();
    }

    public void calibrate(){
        int w=box[2]-box[0]+1;
        int h=box[3]-box[1]+1;
        box[0]=(int)(box[0]+w*bbr[0]);
        box[1]=(int)(box[1]+h*bbr[1]);
        box[2]=(int)(box[2]+w*bbr[2]);
        box[3]=(int)(box[3]+h*bbr[3]);
        for (int i=0;i<4;i++) bbr[i]=0.0f;
    }

    public void toSquareShape(){
        int w=width();
        int h=height();
        if (w>h){
            box[1]-=(w-h)/2;
            box[3]+=(w-h+1)/2;
        }else{
            box[0]-=(h-w)/2;
            box[2]+=(h-w+1)/2;
        }
    }

    public void limit_square(int w,int h){
        if (box[0]<0 || box[1]<0){
            int len=max(-box[0],-box[1]);
            box[0]+=len;
            box[1]+=len;
        }
        if (box[2]>=w || box[3]>=h){
            int len=max(box[2]-w+1,box[3]-h+1);
            box[2]-=len;
            box[3]-=len;
        }
    }
}
