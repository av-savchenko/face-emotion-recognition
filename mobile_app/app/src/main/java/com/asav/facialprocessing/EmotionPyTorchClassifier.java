package com.asav.facialprocessing;

import android.content.Context;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.util.Log;
import android.util.Pair;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

/**
 * Created by avsavchenko.
 */
public class EmotionPyTorchClassifier{

    /** Tag for the {@link Log}. */
    private static final String TAG = "EmotionPyTorch";

    private static final String MODEL_FILE = "enet_b0_8_va_mtl.ptl";//"enet_b0_8_best_vgaf.ptl";//"enet_b2_8.ptl";
    private List<String> labels;
    private Module module=null;
    //private int width=260, height=260;
    private int width=224, height=224;

    public EmotionPyTorchClassifier(final Context context) throws IOException {
        module= LiteModuleLoader.load(assetFilePath(context, MODEL_FILE));
        loadLabels(context);
    }
    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }

    private void loadLabels(final Context context){
        BufferedReader br = null;
        labels=new ArrayList<>();
        try {
            br = new BufferedReader(new InputStreamReader(context.getAssets().open("affectnet_labels.txt")));
            String line;
            int line_ind=0;
            while ((line = br.readLine()) != null) {
                ++line_ind;
                //line=line.toLowerCase();
                String[] categoryInfo=line.trim().split(":");
                String category=categoryInfo[1];
                labels.add(category);
            }
            br.close();
        } catch (IOException e) {
            throw new RuntimeException("Problem reading emotion label file!" , e);
        }
    }
    public String recognize(Bitmap bitmap){
        Pair<Long,float[]> res =  classifyImage(bitmap);
        final float[] scores=res.second;
        int numEmotions=Math.min(labels.size(),scores.length);
        Integer index[] = new Integer[numEmotions];
        for (int i = 0; i < numEmotions; i++) {
            index[i]=i;
        }
        Arrays.sort(index, new Comparator<Integer>() {
            @Override
            public int compare(Integer idx1, Integer idx2) {
                return Float.compare(scores[idx2],scores[idx1]);
            }
        });
        int K=3;
        StringBuilder str=new StringBuilder();
        str.append("Timecost (ms):").append(Long.toString(res.first)).append("\nResult:\n");
        for(int i=0;i<K;++i){
            str.append(labels.get(index[i])+" "+String.valueOf(index[i])+" "+String.valueOf(scores[index[i]])+"\n");
        }
        Log.i(TAG, "PyTorch result: " + str.toString());
        return labels.get(index[0]);
    }

    private Pair<Long,float[]> classifyImage(Bitmap bitmap) {
        bitmap=Bitmap.createScaledBitmap(bitmap, width, height, false);
        final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);
        long startTime = SystemClock.uptimeMillis();
        final Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();
        long timecostMs=SystemClock.uptimeMillis() - startTime;
        Log.i(TAG, "Timecost to run PyTorch model inference: " + timecostMs);
        final float[] scores = outputTensor.getDataAsFloatArray();
        return new Pair<Long,float[]>(timecostMs,scores);
    }
}
