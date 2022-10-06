package com.asav.facialprocessing;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.view.MenuCompat;

import android.content.*;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.graphics.*;
import android.media.ExifInterface;
import android.media.Image;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.widget.*;

import com.asav.facialprocessing.mtcnn.Box;
import com.asav.facialprocessing.mtcnn.MTCNN;

import java.io.*;
import java.nio.ByteBuffer;
import java.util.*;
import androidx.camera.core.*;
import androidx.camera.core.Preview.OnPreviewOutputUpdateListener;
import androidx.camera.core.Preview.PreviewOutput;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";
    private final int REQUEST_CODE_ASK_MULTIPLE_PERMISSIONS = 124;
    private ImageView imageView;
    private Bitmap sampledImage=null;
    private MenuItem checkBoxTorchTf=null;

    private HandlerThread mBackgroundThread=null;
    private Handler mBackgroundHandler=null;

    private static int minFaceSize=32;
    private MTCNN mtcnnFaceDetector=null;
    private AgeGenderEthnicityTfLiteClassifier facialAttributeClassifier=null;
    private EmotionTfLiteClassifier emotionClassifierTfLite =null;
    private EmotionPyTorchClassifier emotionClassifierPyTorch = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        androidx.appcompat.widget.Toolbar toolbar = (androidx.appcompat.widget.Toolbar) findViewById(R.id.main_toolbar);
        setSupportActionBar(toolbar);
        imageView=(ImageView)findViewById(R.id.inputImageView);

        if (!allPermissionsGranted()) {
            ActivityCompat.requestPermissions(this, getRequiredPermissions(), REQUEST_CODE_ASK_MULTIPLE_PERMISSIONS);
        }
        else
            init();
    }
    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.toolbar_menu, menu);
        checkBoxTorchTf=menu.findItem(R.id.action_emotion_tf_or_torch);
        MenuCompat.setGroupDividerEnabled(menu, true);
        return true;
    }

    private void init(){
        try {
            emotionClassifierPyTorch =new EmotionPyTorchClassifier(getApplicationContext());
        } catch (final Exception e) {
            Log.e(TAG, "Exception initializing EmotionPyTorchClassifier!", e);
        }
        try {
            mtcnnFaceDetector =new MTCNN(getApplicationContext());
        } catch (final Exception e) {
            Log.e(TAG, "Exception initializing MTCNNModel!"+e);
        }
        try {
            facialAttributeClassifier=new AgeGenderEthnicityTfLiteClassifier(getApplicationContext());
        } catch (final Exception e) {
            Log.e(TAG, "Exception initializing AgeGenderEthnicityTfLiteClassifier!", e);
        }

        try {
            emotionClassifierTfLite =new EmotionTfLiteClassifier(getApplicationContext());
        } catch (final Exception e) {
            Log.e(TAG, "Exception initializing EmotionTfLiteClassifier!", e);
        }

    }
    private String[] getRequiredPermissions() {
        try {
            PackageInfo info =
                    getPackageManager()
                            .getPackageInfo(getPackageName(), PackageManager.GET_PERMISSIONS);
            String[] ps = info.requestedPermissions;
            if (ps != null && ps.length > 0) {
                return ps;
            } else {
                return new String[0];
            }
        } catch (Exception e) {
            return new String[0];
        }
    }
    private boolean allPermissionsGranted() {
        for (String permission : getRequiredPermissions()) {
            int status= ContextCompat.checkSelfPermission(this,permission);
            if (ContextCompat.checkSelfPermission(this,permission)
                    != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }



    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        switch (requestCode) {
            case REQUEST_CODE_ASK_MULTIPLE_PERMISSIONS:
                Map<String, Integer> perms = new HashMap<String, Integer>();
                boolean allGranted = true;
                for (int i = 0; i < permissions.length; i++) {
                    perms.put(permissions[i], grantResults[i]);
                    if (grantResults[i] != PackageManager.PERMISSION_GRANTED)
                        allGranted = false;
                }
                // Check for ACCESS_FINE_LOCATION
                if (allGranted) {
                    // All Permissions Granted
                    init();
                } else {
                    // Permission Denied
                    Toast.makeText(MainActivity.this, "Some Permission is Denied", Toast.LENGTH_SHORT)
                            .show();
                    finish();
                }
                break;
            default:
                super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        }
    }

    private static final int SELECT_PICTURE = 1;
    private void openImageFile(int requestCode){
        Intent intent = new Intent();
        intent.setType("image/*");
        intent.setAction(Intent.ACTION_GET_CONTENT);
        startActivityForResult(Intent.createChooser(intent,"Select Picture"),requestCode);
    }
    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            case R.id.action_openGallery:
                if(!isCameraRunning()) {
                    openImageFile(SELECT_PICTURE);
                }
                return true;
            /*case R.id.action_detectface_mtcnn:
                if(isImageLoaded() && !isCameraRunning()) {
                    mtcnnDetectionAndAttributesRecognition(null);
                }
                return true;*/
            case R.id.action_agegender:
                if(isImageLoaded() && !isCameraRunning()) {
                    mtcnnDetectionAndAttributesRecognition(facialAttributeClassifier);
                }
                return true;
            case R.id.action_emotion:
                if(!isCameraRunning()) {
                    recognizeEmotions();
                }
                return true;
            case R.id.action_emotion_tf_or_torch:
                if(!isCameraRunning()) {
                    if (item.isChecked()) {
                        item.setChecked(false);
                    } else {
                        item.setChecked(true);
                    }
                    recognizeEmotions();
                }
                return true;
            case R.id.action_capturecamera:
                if(mBackgroundThread==null){
                    item.setTitle(R.string.action_StopCamera);
                    setupCameraX();
                }
                else{
                    item.setTitle(R.string.action_CaptureCamera);
                    stopCamera();
                }
                return true;
            default:
                // If we got here, the user's action was not recognized.
                // Invoke the superclass to handle it.
                return super.onOptionsItemSelected(item);
        }
    }
    private void recognizeEmotions(){
        if(isImageLoaded()) {
            if (checkBoxTorchTf.isChecked()) {
                mtcnnDetectionAndEmotionPyTorchRecognition();
            } else {
                mtcnnDetectionAndAttributesRecognition(emotionClassifierTfLite);
            }
        }

    }
    private Bitmap getImage(Uri selectedImageUri)
    {
        Bitmap bmp=null;
        try {
            InputStream ims = getContentResolver().openInputStream(selectedImageUri);
            bmp= BitmapFactory.decodeStream(ims);
            ims.close();
            ims = getContentResolver().openInputStream(selectedImageUri);
            ExifInterface exif = new ExifInterface(ims);//selectedImageUri.getPath());
            int orientation = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION,1);
            int degreesForRotation=0;
            switch (orientation)
            {
                case ExifInterface.ORIENTATION_ROTATE_90:
                    degreesForRotation=90;
                    break;
                case ExifInterface.ORIENTATION_ROTATE_270:
                    degreesForRotation=270;
                    break;
                case ExifInterface.ORIENTATION_ROTATE_180:
                    degreesForRotation=180;
                    break;
            }
            if(degreesForRotation!=0) {
                Matrix matrix = new Matrix();
                matrix.setRotate(degreesForRotation);
                bmp=Bitmap.createBitmap(bmp, 0, 0, bmp.getWidth(),
                        bmp.getHeight(), matrix, true);
            }

        } catch (Exception e) {
            Log.e(TAG, "Exception thrown: " + e+" "+Log.getStackTraceString(e));
        }
        return bmp;
    }
    private boolean isCameraRunning(){
        if(mBackgroundThread!=null)
            Toast.makeText(getApplicationContext(),
                    "Stop camera firstly",
                    Toast.LENGTH_SHORT).show();
        return mBackgroundThread!=null;
    }
    private boolean isImageLoaded(){
        if(sampledImage==null)
            Toast.makeText(getApplicationContext(),
                    "It is necessary to open image firstly",
                    Toast.LENGTH_SHORT).show();
        return sampledImage!=null;
    }
    private void setImage(Bitmap bitmap){
        runOnUiThread(new Runnable() {

            @Override
            public void run() {
                imageView.setImageBitmap(bitmap);
            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if(resultCode==RESULT_OK) {
            if (requestCode == SELECT_PICTURE) {
                Uri selectedImageUri = data.getData(); //The uri with the location of the file
                Log.d(TAG, "uri" + selectedImageUri);
                sampledImage=getImage(selectedImageUri);
                if(sampledImage!=null)
                    setImage(sampledImage);
            }
        }
    }
    private void mtcnnDetectionAndAttributesRecognition(TfLiteClassifier classifier){
        Bitmap bmp = sampledImage;
        Bitmap resizedBitmap=bmp;
        double minSize=600.0;
        double scale=Math.min(bmp.getWidth(),bmp.getHeight())/minSize;
        if(scale>1.0) {
            resizedBitmap = Bitmap.createScaledBitmap(bmp, (int)(bmp.getWidth()/scale), (int)(bmp.getHeight()/scale), false);
            //bmp=resizedBitmap;
        }
        long startTime = SystemClock.uptimeMillis();
        Vector<Box> bboxes = mtcnnFaceDetector.detectFaces(resizedBitmap, minFaceSize);//(int)(bmp.getWidth()*MIN_FACE_SIZE));
        Log.i(TAG, "Timecost to run mtcnn: " + Long.toString(SystemClock.uptimeMillis() - startTime));

        Bitmap tempBmp = Bitmap.createBitmap(resizedBitmap.getWidth(), resizedBitmap.getHeight(), Bitmap.Config.ARGB_8888);
        Canvas c = new Canvas(tempBmp);
        Paint p = new Paint();
        p.setStyle(Paint.Style.STROKE);
        p.setAntiAlias(true);
        p.setFilterBitmap(true);
        p.setDither(true);
        p.setColor(Color.BLUE);
        p.setStrokeWidth(5);

        Paint p_text = new Paint();
        p_text.setColor(Color.WHITE);
        p_text.setStyle(Paint.Style.FILL);
        p_text.setColor(Color.BLUE);
        p_text.setTextSize(24);

        c.drawBitmap(resizedBitmap, 0, 0, null);

        for (Box box : bboxes) {

            p.setColor(Color.RED);
            android.graphics.Rect bbox = box.transform2Rect();//new android.graphics.Rect(Math.max(0,box.left()),Math.max(0,box.top()),box.right(),box.bottom());
            c.drawRect(bbox, p);
            if(classifier!=null && bbox.width()>0 && bbox.height()>0) {
                int w=bmp.getWidth();
                int h=bmp.getHeight();
                android.graphics.Rect bboxOrig = new android.graphics.Rect(
                        Math.max(0,w*bbox.left / resizedBitmap.getWidth()),
                        Math.max(0,h*bbox.top / resizedBitmap.getHeight()),
                        Math.min(w,w * bbox.right / resizedBitmap.getWidth()),
                        Math.min(h,h * bbox.bottom / resizedBitmap.getHeight())
                );
                Bitmap faceBitmap = Bitmap.createBitmap(bmp, bboxOrig.left, bboxOrig.top, bboxOrig.width(), bboxOrig.height());
                Bitmap resultBitmap = Bitmap.createScaledBitmap(faceBitmap, classifier.getImageSizeX(), classifier.getImageSizeY(), false);
                ClassifierResult res = classifier.classifyFrame(resultBitmap);
                c.drawText(res.toString(), Math.max(0,bbox.left), Math.max(0, bbox.top - 20), p_text);
                Log.i(TAG, res.toString());
            }
        }
        setImage(tempBmp);
    }
    private void mtcnnDetectionAndEmotionPyTorchRecognition(){
        Bitmap bmp = sampledImage;
        Bitmap resizedBitmap=bmp;
        double minSize=600.0;
        double scale=Math.min(bmp.getWidth(),bmp.getHeight())/minSize;
        if(scale>1.0) {
            resizedBitmap = Bitmap.createScaledBitmap(bmp, (int)(bmp.getWidth()/scale), (int)(bmp.getHeight()/scale), false);
            bmp=resizedBitmap;
        }
        long startTime = SystemClock.uptimeMillis();
        Vector<Box> bboxes = mtcnnFaceDetector.detectFaces(resizedBitmap, minFaceSize);//(int)(bmp.getWidth()*MIN_FACE_SIZE));
        Log.i(TAG, "Timecost to run mtcnn: " + Long.toString(SystemClock.uptimeMillis() - startTime));

        Bitmap tempBmp = Bitmap.createBitmap(bmp.getWidth(), bmp.getHeight(), Bitmap.Config.ARGB_8888);
        Canvas c = new Canvas(tempBmp);
        Paint p = new Paint();
        p.setStyle(Paint.Style.STROKE);
        p.setAntiAlias(true);
        p.setFilterBitmap(true);
        p.setDither(true);
        p.setColor(Color.BLUE);
        p.setStrokeWidth(5);

        Paint p_text = new Paint();
        p_text.setColor(Color.WHITE);
        p_text.setStyle(Paint.Style.FILL);
        p_text.setColor(Color.BLUE);
        p_text.setTextSize(24);

        c.drawBitmap(bmp, 0, 0, null);

        for (Box box : bboxes) {
            android.graphics.Rect bbox = box.transform2Rect();//new android.graphics.Rect(Math.max(0,box.left()),Math.max(0,box.top()),box.right(),box.bottom());
            p.setColor(Color.RED);
            c.drawRect(bbox, p);
            if(emotionClassifierPyTorch!=null && bbox.width()>0 && bbox.height()>0) {
                int w=bmp.getWidth();
                int h=bmp.getHeight();
                android.graphics.Rect bboxOrig = new android.graphics.Rect(
                        Math.max(0,w*bbox.left / resizedBitmap.getWidth()),
                        Math.max(0,h*bbox.top / resizedBitmap.getHeight()),
                        Math.min(w,w * bbox.right / resizedBitmap.getWidth()),
                        Math.min(h,h * bbox.bottom / resizedBitmap.getHeight())
                );
                Bitmap faceBitmap = Bitmap.createBitmap(bmp, bboxOrig.left, bboxOrig.top, bboxOrig.width(), bboxOrig.height());
                String res=emotionClassifierPyTorch.recognize(faceBitmap);
                c.drawText(res, Math.max(0,bbox.left), Math.max(0, bbox.top - 20), p_text);
                Log.i(TAG, res);
            }
        }
        setImage(tempBmp);
    }


    private void setupCameraX() {
        PreviewConfig previewConfig = new PreviewConfig.Builder()
                .setLensFacing(CameraX.LensFacing.FRONT)
                .build();
        Preview preview = new Preview(previewConfig);
        mBackgroundThread = new HandlerThread("AnalysisThread");
        mBackgroundThread.start();
        mBackgroundHandler = new Handler(mBackgroundThread.getLooper());

        ImageAnalysis imageAnalysis = new ImageAnalysis(new ImageAnalysisConfig.Builder()
                .setLensFacing(CameraX.LensFacing.FRONT)
                .setCallbackHandler(mBackgroundHandler)
                .setImageReaderMode(ImageAnalysis.ImageReaderMode.ACQUIRE_LATEST_IMAGE)
                .build());
        imageAnalysis.setAnalyzer(
                new ImageAnalysis.Analyzer() {
                    public void analyze(ImageProxy image, int rotationDegrees) {
                        sampledImage=imgToBitmap(image.getImage(), rotationDegrees);
                        recognizeEmotions();
                    }
                }
        );

        CameraX.unbindAll();
        CameraX.bindToLifecycle(this, preview, imageAnalysis);
    }

    private Bitmap imgToBitmap(Image image, int rotationDegrees) {
        // NV21 is a plane of 8 bit Y values followed by interleaved  Cb Cr
        ByteBuffer ib = ByteBuffer.allocate(image.getHeight() * image.getWidth() * 2);

        ByteBuffer y = image.getPlanes()[0].getBuffer();
        ByteBuffer cr = image.getPlanes()[1].getBuffer();
        ByteBuffer cb = image.getPlanes()[2].getBuffer();
        ib.put(y);
        ib.put(cb);
        ib.put(cr);

        YuvImage yuvImage = new YuvImage(ib.array(),
                ImageFormat.NV21, image.getWidth(), image.getHeight(), null);

        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0,
                image.getWidth(), image.getHeight()), 50, out);
        byte[] imageBytes = out.toByteArray();
        Bitmap bm = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
        Bitmap bitmap = bm;

        // On android the camera rotation and the screen rotation
        // are off by 90 degrees, so if you are capturing an image
        // in "portrait" orientation, you'll need to rotate the image.
        if (rotationDegrees != 0) {
            Matrix matrix = new Matrix();
            matrix.postRotate(rotationDegrees);
            Bitmap scaledBitmap = Bitmap.createScaledBitmap(bm,
                    bm.getWidth(), bm.getHeight(), true);
            bitmap = Bitmap.createBitmap(scaledBitmap, 0, 0,
                    scaledBitmap.getWidth(), scaledBitmap.getHeight(), matrix, true);
        }
        return bitmap;
    }
    private void stopCamera() {
        CameraX.unbindAll();
        mBackgroundThread.quitSafely();
        try {
            mBackgroundThread.join();
        } catch (InterruptedException e) {
            Log.e(TAG, "Exception stoppingCamera!", e);
        }
        mBackgroundThread = null;
        mBackgroundHandler = null;
    }
}