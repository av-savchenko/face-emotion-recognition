package com.asav.facialprocessing.mtcnn;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.graphics.Point;
import android.graphics.Rect;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Vector;

/**
 * MTCNN for Android.
 */
public class MTCNN {
    // 参数
    private float factor = 0.709f;
    private float pNetThreshold = 0.6f;
    private float rNetThreshold = 0.7f;
    private float oNetThreshold = 0.7f;

    private static final String MODEL_FILE_PNET = "pnet.tflite";
    private static final String MODEL_FILE_RNET = "rnet.tflite";
    private static final String MODEL_FILE_ONET = "onet.tflite";

    private Interpreter pInterpreter;
    private Interpreter rInterpreter;
    private Interpreter oInterpreter;

    public MTCNN(Context context) throws IOException {
        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(4);
        pInterpreter = new Interpreter(FileUtil.loadMappedFile(context, MODEL_FILE_PNET), options);
        rInterpreter = new Interpreter(FileUtil.loadMappedFile(context, MODEL_FILE_RNET), options);
        oInterpreter = new Interpreter(FileUtil.loadMappedFile(context, MODEL_FILE_ONET), options);
    }

    /**
     * 人脸检测
     * @param bitmap 要处理的图片
     * @param minFaceSize 最小的人脸像素值. (此值越大，检测越快)
     */
    public Vector<Box> detectFaces(Bitmap bitmap, int minFaceSize) {
        Vector<Box> boxes;
        try {
            //【1】PNet generate candidate boxes
            boxes = pNet(bitmap, minFaceSize);
            square_limit(boxes, bitmap.getWidth(), bitmap.getHeight());

            //【2】RNet
            boxes = rNet(bitmap, boxes);
            square_limit(boxes, bitmap.getWidth(), bitmap.getHeight());

            //【3】ONet
            boxes = oNet(bitmap, boxes);
        } catch (IllegalArgumentException e) {
            e.printStackTrace();
            boxes = new Vector<>();
        }
        return boxes;
    }

    private void square_limit(Vector<Box> boxes, int w, int h) {
        // square
        for (int i = 0; i < boxes.size(); i++) {
            boxes.get(i).toSquareShape();
            boxes.get(i).limitSquare(w, h);
        }
    }

    /**
     * NMS执行完后，才执行Regression
     * (1) For each scale , use NMS with threshold=0.5
     * (2) For all candidates , use NMS with threshold=0.7
     * (3) Calibrate Bounding Box
     * 注意：CNN输入图片最上面一行，坐标为[0..width,0]。所以Bitmap需要对折后再跑网络;网络输出同理.
     *
     * @param bitmap
     * @return
     */
    private Vector<Box> pNet(Bitmap bitmap, int minSize) {
        int whMin = Math.min(bitmap.getWidth(), bitmap.getHeight());
        float currentFaceSize = minSize; // currentFaceSize=minSize/(factor^k) k=0,1,2... until excced whMin
        Vector<Box> totalBoxes = new Vector<>();
        //【1】Image Paramid and Feed to Pnet
        while (currentFaceSize <= whMin) {
            float scale = 12.0f / currentFaceSize;

            // (1)Image Resize
            Bitmap bm = bitmapResize(bitmap, scale);
            int w = bm.getWidth();
            int h = bm.getHeight();

            // (2)RUN CNN
            int outW = (int) (Math.ceil(w * 0.5 - 5) + 0.5);
            int outH = (int) (Math.ceil(h * 0.5 - 5) + 0.5);
            float[][][][] prob1 = new float[1][outW][outH][2];
            float[][][][] conv4_2_BiasAdd = new float[1][outW][outH][4];
            pNetForward(bm, prob1, conv4_2_BiasAdd);
            prob1 = transposeBatch(prob1);
            conv4_2_BiasAdd = transposeBatch(conv4_2_BiasAdd);

            // (3)数据解析
            Vector<Box> curBoxes = new Vector<>();
            generateBoxes(prob1, conv4_2_BiasAdd, scale, curBoxes);

            // (4)nms 0.5
            nms(curBoxes, 0.5f, "Union");

            // (5)add to totalBoxes
            for (int i = 0; i < curBoxes.size(); i++)
                if (!curBoxes.get(i).deleted)
                    totalBoxes.addElement(curBoxes.get(i));

            // Face Size等比递增
            currentFaceSize /= factor;
        }

        // NMS 0.7
        nms(totalBoxes, 0.7f, "Union");

        // BBR
        BoundingBoxReggression(totalBoxes);

        return updateBoxes(totalBoxes);
    }

    /**
     * pnet前向传播
     *
     * @param bitmap
     * @param prob1
     * @param conv4_2_BiasAdd
     * @return
     */
    private void pNetForward(Bitmap bitmap, float[][][][] prob1, float[][][][] conv4_2_BiasAdd) {
        float[][][] img = normalizeImage(bitmap);
        float[][][][] pNetIn = new float[1][][][];
        pNetIn[0] = img;
        pNetIn = transposeBatch(pNetIn);

        Map<Integer, Object> outputs = new HashMap<>();
        outputs.put(pInterpreter.getOutputIndex("pnet/prob1"), prob1);
        outputs.put(pInterpreter.getOutputIndex("pnet/conv4-2/BiasAdd"), conv4_2_BiasAdd);

        pInterpreter.runForMultipleInputsOutputs(new Object[]{pNetIn}, outputs);
    }

    private int generateBoxes(float[][][][] prob1, float[][][][] conv4_2_BiasAdd, float scale, Vector<Box> boxes) {
        int h = prob1[0].length;
        int w = prob1[0][0].length;

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                float score = prob1[0][y][x][1];
                // only accept prob >threadshold(0.6 here)
                if (score > pNetThreshold) {
                    Box box = new Box();
                    // core
                    box.score = score;
                    // box
                    box.box[0] = Math.round(x * 2 / scale);
                    box.box[1] = Math.round(y * 2 / scale);
                    box.box[2] = Math.round((x * 2 + 11) / scale);
                    box.box[3] = Math.round((y * 2 + 11) / scale);
                    // bbr
                    for (int i = 0; i < 4; i++) {
                        box.bbr[i] = conv4_2_BiasAdd[0][y][x][i];
                    }
                    // add
                    boxes.addElement(box);
                }
            }
        }
        return 0;
    }

    /**
     * nms，不符合条件的deleted设置为true
     *
     * @param boxes
     * @param threshold
     * @param method
     */
    private void nms(Vector<Box> boxes, float threshold, String method) {
        // NMS.两两比对
        // int delete_cnt = 0;
        for (int i = 0; i < boxes.size(); i++) {
            Box box = boxes.get(i);
            if (!box.deleted) {
                // score<0表示当前矩形框被删除
                for (int j = i + 1; j < boxes.size(); j++) {
                    Box box2 = boxes.get(j);
                    if (!box2.deleted) {
                        int x1 = Math.max(box.box[0], box2.box[0]);
                        int y1 = Math.max(box.box[1], box2.box[1]);
                        int x2 = Math.min(box.box[2], box2.box[2]);
                        int y2 = Math.min(box.box[3], box2.box[3]);
                        if (x2 < x1 || y2 < y1) continue;
                        int areaIoU = (x2 - x1 + 1) * (y2 - y1 + 1);
                        float iou = 0f;
                        if (method.equals("Union"))
                            iou = 1.0f * areaIoU / (box.area() + box2.area() - areaIoU);
                        else if (method.equals("Min"))
                            iou = 1.0f * areaIoU / (Math.min(box.area(), box2.area()));
                        if (iou >= threshold) { // 删除prob小的那个框
                            if (box.score > box2.score)
                                box2.deleted = true;
                            else
                                box.deleted = true;
                        }
                    }
                }
            }
        }
    }

    private void BoundingBoxReggression(Vector<Box> boxes) {
        for (int i = 0; i < boxes.size(); i++)
            boxes.get(i).calibrate();
    }

    /**
     * Refine Net
     * @param bitmap
     * @param boxes
     * @return
     */
    private Vector<Box> rNet(Bitmap bitmap, Vector<Box> boxes) {
        // RNet Input Init
        int num = boxes.size();
        float[][][][] rNetIn = new float[num][24][24][3];
        for (int i = 0; i < num; i++) {
            float[][][] curCrop = cropAndResize(bitmap, boxes.get(i), 24);
            curCrop = transposeImage(curCrop);
            rNetIn[i] = curCrop;
        }

        // Run RNet
        rNetForward(rNetIn, boxes);

        // RNetThreshold
        for (int i = 0; i < num; i++) {
            if (boxes.get(i).score < rNetThreshold) {
                boxes.get(i).deleted = true;
            }
        }

        // Nms
        nms(boxes, 0.7f, "Union");
        BoundingBoxReggression(boxes);
        return updateBoxes(boxes);
    }

    /**
     * RNET跑神经网络，将score和bias写入boxes
     * @param rNetIn
     * @param boxes
     */
    private void rNetForward(float[][][][] rNetIn, Vector<Box> boxes) {
        int num = rNetIn.length;
        float[][] prob1 = new float[num][2];
        float[][] conv5_2_conv5_2 = new float[num][4];

        Map<Integer, Object> outputs = new HashMap<>();
        outputs.put(rInterpreter.getOutputIndex("rnet/prob1"), prob1);
        outputs.put(rInterpreter.getOutputIndex("rnet/conv5-2/conv5-2"), conv5_2_conv5_2);
        rInterpreter.runForMultipleInputsOutputs(new Object[]{rNetIn}, outputs);

        // 转换
        for (int i = 0; i < num; i++) {
            boxes.get(i).score = prob1[i][1];
            for (int j = 0; j < 4; j++) {
                boxes.get(i).bbr[j] = conv5_2_conv5_2[i][j];
            }
        }
    }

    /**
     * ONet
     * @param bitmap
     * @param boxes
     * @return
     */
    private Vector<Box> oNet(Bitmap bitmap, Vector<Box> boxes) {
        // ONet Input Init
        int num = boxes.size();
        float[][][][] oNetIn = new float[num][48][48][3];
        for (int i = 0; i < num; i++) {
            float[][][] curCrop = cropAndResize(bitmap, boxes.get(i), 48);
            curCrop = transposeImage(curCrop);
            oNetIn[i] = curCrop;
        }

        // Run ONet
        oNetForward(oNetIn, boxes);
        // ONetThreshold
        for (int i = 0; i < num; i++) {
            if (boxes.get(i).score < oNetThreshold) {
                boxes.get(i).deleted = true;
            }
        }
        BoundingBoxReggression(boxes);
        // Nms
        nms(boxes, 0.7f, "Min");
        return updateBoxes(boxes);
    }

    private void oNetForward(float[][][][] oNetIn, Vector<Box> boxes) {
        int num = oNetIn.length;
        float[][] prob1 = new float[num][2];
        float[][] conv6_2_conv6_2 = new float[num][4];
        float[][] conv6_3_conv6_3 = new float[num][10];

        Map<Integer, Object> outputs = new HashMap<>();
        outputs.put(oInterpreter.getOutputIndex("onet/prob1"), prob1);
        outputs.put(oInterpreter.getOutputIndex("onet/conv6-2/conv6-2"), conv6_2_conv6_2);
        outputs.put(oInterpreter.getOutputIndex("onet/conv6-3/conv6-3"), conv6_3_conv6_3);
        oInterpreter.runForMultipleInputsOutputs(new Object[]{oNetIn}, outputs);

        // 转换
        for (int i = 0; i < num; i++) {
            // prob
            boxes.get(i).score = prob1[i][1];
            // bias
            for (int j = 0; j < 4; j++) {
                boxes.get(i).bbr[j] = conv6_2_conv6_2[i][j];
            }
            // landmark
            for (int j = 0; j < 5; j++) {
                int x = Math.round(boxes.get(i).left() + (conv6_3_conv6_3[i][j] * boxes.get(i).width()));
                int y = Math.round(boxes.get(i).top() + (conv6_3_conv6_3[i][j + 5] * boxes.get(i).height()));
                boxes.get(i).landmark[j] = new Point(x, y);
            }
        }
    }

    /**
     * 删除做了delete标记的box
     * @param boxes
     * @return
     */
    public static Vector<Box> updateBoxes(Vector<Box> boxes) {
        Vector<Box> b = new Vector<>();
        for (int i = 0; i < boxes.size(); i++) {
            if (!boxes.get(i).deleted) {
                b.addElement(boxes.get(i));
            }
        }
        return b;
    }

    public static float[][][] normalizeImage(Bitmap bitmap) {
        int h = bitmap.getHeight();
        int w = bitmap.getWidth();
        float[][][] floatValues = new float[h][w][3];

        float imageMean = 127.5f;
        float imageStd = 128;

        int[] pixels = new int[h * w];
        bitmap.getPixels(pixels, 0, bitmap.getWidth(), 0, 0, w, h);
        for (int i = 0; i < h; i++) { // жіЁж„ЏжЇе…€й«еђЋе®Ѕ
            for (int j = 0; j < w; j++) {
                final int val = pixels[i * w + j];
                float r = (((val >> 16) & 0xFF) - imageMean) / imageStd;
                float g = (((val >> 8) & 0xFF) - imageMean) / imageStd;
                float b = ((val & 0xFF) - imageMean) / imageStd;
                float[] arr = {r, g, b};
                floatValues[i][j] = arr;
            }
        }
        return floatValues;
    }

    public static Bitmap bitmapResize(Bitmap bitmap, float scale) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        Matrix matrix = new Matrix();
        matrix.postScale(scale, scale);
        return Bitmap.createBitmap(
                bitmap, 0, 0, width, height, matrix, true);
    }

    public static float[][][] transposeImage(float[][][] in) {
        int h = in.length;
        int w = in[0].length;
        int channel = in[0][0].length;
        float[][][] out = new float[w][h][channel];
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                out[j][i] = in[i][j] ;
            }
        }
        return out;
    }

    /**
     * 4з»ґе›ѕз‰‡batchзџ©йµе®Ѕй«иЅ¬зЅ®
     * @param in
     * @return
     */
    public static float[][][][] transposeBatch(float[][][][] in) {
        int batch = in.length;
        int h = in[0].length;
        int w = in[0][0].length;
        int channel = in[0][0][0].length;
        float[][][][] out = new float[batch][w][h][channel];
        for (int i = 0; i < batch; i++) {
            for (int j = 0; j < h; j++) {
                for (int k = 0; k < w; k++) {
                    out[i][k][j] = in[i][j][k] ;
                }
            }
        }
        return out;
    }
    public static float[][][] cropAndResize(Bitmap bitmap, Box box, int size) {
        // crop and resize
        Matrix matrix = new Matrix();
        float scaleW = 1.0f * size / box.width();
        float scaleH = 1.0f * size / box.height();
        matrix.postScale(scaleW, scaleH);
        Rect rect = box.transform2Rect();
        Bitmap croped = Bitmap.createBitmap(
                bitmap, rect.left, rect.top, box.width(), box.height(), matrix, true);

        return normalizeImage(croped);
    }

}
