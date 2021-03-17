package com.asav.facialprocessing.mtcnn

import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.Point
import org.tensorflow.contrib.android.TensorFlowInferenceInterface
import java.util.*
import android.content.res.AssetManager
import kotlin.math.max
import kotlin.math.min


class MTCNNModel(val tensorFlowInferenceInterface: TensorFlowInferenceInterface)  {
    companion object {
        fun create (assetManager : AssetManager) : MTCNNModel {
            return MTCNNModel(TensorFlowInferenceInterface(assetManager, "file:///android_asset/mtcnn_model.pb"))
        }
    }

    private val factor = 0.709f
    private val PNetThreshold = 0.6f
    private val RNetThreshold = 0.7f
    private val ONetThreshold = 0.7f
    private val pNetInputName = "pnet/input:0"
    private val pNetOutputName = arrayOf("pnet/prob1:0", "pnet/conv4-2/BiasAdd:0")
    private val rNetInputName = "rnet/input:0"
    private val rNetOutputName = arrayOf("rnet/prob1:0", "rnet/conv5-2/conv5-2:0")
    private val oNetInputName = "onet/input:0"
    private val oNetOutputName = arrayOf("onet/prob1:0", "onet/conv6-2/conv6-2:0", "onet/conv6-3/conv6-3:0")
    private val numOfChannels = 3L

     private fun bitmapResize(bm: Bitmap, scale: Float): Bitmap {
         val width = bm.width
         val height = bm.height
         // CREATE A MATRIX FOR THE MANIPULATION。matrix
         val matrix = Matrix()
         // RESIZE THE BIT MAP
         matrix.postScale(scale, scale)
         return Bitmap.createBitmap(
                 bm, 0, 0, width, height, matrix, true)
    }
    fun normalizeImage(bitmap: Bitmap): FloatArray {
        val w = bitmap.width
        val h = bitmap.height
        val floatValues = FloatArray(w * h * 3)
        val intValues = IntArray(w * h)
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        val imageMean = 127.5f
        val imageStd = 128f

        for (i in intValues.indices) {
            val `val` = intValues[i]
            floatValues[i * 3 + 0] = ((`val` shr 16 and 0xFF) - imageMean) / imageStd
            floatValues[i * 3 + 1] = ((`val` shr 8 and 0xFF) - imageMean) / imageStd
            floatValues[i * 3 + 2] = ((`val` and 0xFF) - imageMean) / imageStd
        }
        return floatValues
    }

    private fun PNetForward(bitmap: Bitmap, PNetOutProb: Array<FloatArray>, PNetOutBias: Array<Array<FloatArray>>): Int {
        val w = bitmap.width
        val h = bitmap.height

        val PNetIn = normalizeImage(bitmap)
        Utils.flip_diag(PNetIn, h, w, 3) //沿着对角线翻转
        tensorFlowInferenceInterface.feed(pNetInputName, PNetIn, 1L, w.toLong(), h.toLong(), numOfChannels)
        tensorFlowInferenceInterface.run(pNetOutputName, false)
        val PNetOutSizeW = Math.ceil(w * 0.5 - 5).toInt()
        val PNetOutSizeH = Math.ceil(h * 0.5 - 5).toInt()
        val PNetOutP = FloatArray(PNetOutSizeW * PNetOutSizeH * 2)
        val PNetOutB = FloatArray(PNetOutSizeW * PNetOutSizeH * 4)
        tensorFlowInferenceInterface.fetch(pNetOutputName[0], PNetOutP)
        tensorFlowInferenceInterface.fetch(pNetOutputName[1], PNetOutB)

        Utils.flip_diag(PNetOutP, PNetOutSizeW, PNetOutSizeH, 2)
        Utils.flip_diag(PNetOutB, PNetOutSizeW, PNetOutSizeH, 4)
        Utils.expand(PNetOutB, PNetOutBias)
        Utils.expandProb(PNetOutP, PNetOutProb)
        return 0
    }

 //Non-Maximum Suppression
     private fun nms(boxes: Vector<Box>, threshold: Float, method: String) {
         val cnt = 0
         for (i in boxes.indices) {
             val box = boxes[i]
             if (!box.deleted) {
                 for (j in i + 1 until boxes.size) {
                 val box2 = boxes[j]
                 if (!box2.deleted) {
                     val x1 = max(box.box[0], box2.box[0])
                     val y1 = max(box.box[1], box2.box[1])
                     val x2 = min(box.box[2], box2.box[2])
                     val y2 = min(box.box[3], box2.box[3])
                     if (x2 < x1 || y2 < y1) continue
                     val areaIoU = (x2 - x1 + 1) * (y2 - y1 + 1)
                     var iou = 0f
                     if (method == "Union")
                     iou = 1.0f * areaIoU / (box.area() + box2.area() - areaIoU)
                     else if (method == "Min") {
                     iou = 1.0f * areaIoU / min(box.area(), box2.area())
                     }
                     if (iou >= threshold) { //删除prob小的那个框
                         if (box.score > box2.score)
                             box2.deleted = true
                         else
                             box.deleted = true
                         //delete_cnt++;
                     }
                     }
                 }
             }
         }
     }

    private fun generateBoxes(prob: Array<FloatArray>, bias: Array<Array<FloatArray>>, scale: Float, threshold: Float, boxes: Vector<Box>): Int {
         val h = prob.size
         val w = prob[0].size

         for (y in 0 until h)
             for (x in 0 until w) {
                 val score = prob[y][x]
                 //only accept prob >threadshold(0.6 here)
                 if (score > threshold) {
                      val box = Box()
                      //score
                      box.score = score
                      //box
                      box.box[0] = Math.round(x * 2 / scale)
                      box.box[1] = Math.round(y * 2 / scale)
                      box.box[2] = Math.round((x * 2 + 11) / scale)
                      box.box[3] = Math.round((y * 2 + 11) / scale)
                      //bbr
                      for (i in 0..3)
                       box.bbr[i] = bias[y][x][i]
                      //add
                      boxes.addElement(box)
                 }
             }
         return 0
    }

    private fun BoundingBoxReggression(boxes: Vector<Box>) {
        for (i in boxes.indices)
           boxes[i].calibrate()
    }

    //Pnet + Bounding Box Regression + Non-Maximum Regression

    private fun PNet(bitmap: Bitmap, minSize: Int): Vector<Box> {
        val whMin = min(bitmap.width, bitmap.height)
        var currentFaceSize = minSize.toFloat()  //currentFaceSize=minSize/(factor^k) k=0,1,2... until excced whMin
        val totalBoxes = Vector<Box>()
        //【1】Image Paramid and Feed to Pnet
        while (currentFaceSize <= whMin) {
            val scale = 12.0f / currentFaceSize
            //(1)Image Resize
            val bm = bitmapResize(bitmap, scale)
            val w = bm.width
            val h = bm.height
            //(2)RUN CNN
            val PNetOutSizeW = (Math.ceil(w * 0.5 - 5) + 0.5).toInt()
            val PNetOutSizeH = (Math.ceil(h * 0.5 - 5) + 0.5).toInt()
            val PNetOutProb = Array(PNetOutSizeH) { FloatArray(PNetOutSizeW) }
            val PNetOutBias = Array(PNetOutSizeH) { Array(PNetOutSizeW) { FloatArray(4) } }
            PNetForward(bm, PNetOutProb, PNetOutBias)

            val curBoxes = Vector<Box>()
            generateBoxes(PNetOutProb, PNetOutBias, scale, PNetThreshold, curBoxes)
            //Log.i(TAG,"[*]CNN Output Box number:"+curBoxes.size()+" Scale:"+scale);
            //(4)nms 0.5
            nms(curBoxes, 0.5f, "Union")
            //(5)add to totalBoxes
            for (i in curBoxes.indices)
                if (!curBoxes[i].deleted)
                    totalBoxes.addElement(curBoxes[i])
            //Face Size等比递增
            currentFaceSize /= factor
        }
        //NMS 0.7
        nms(totalBoxes, 0.7f, "Union")
        //BBR
        BoundingBoxReggression(totalBoxes)
        return Utils.updateBoxes(totalBoxes)
    }

    private fun crop_and_resize(bitmap: Bitmap, box: Box, size: Int, data: FloatArray) {
         //(2)crop and resize
         val matrix = Matrix()
         val scale = 1.0f * size / box.width()
         matrix.postScale(scale, scale)
         val croped = Bitmap.createBitmap(bitmap, box.left(), box.top(), box.width(), box.height(), matrix, true)
         //(3)save
         val pixels_buf = IntArray(size * size)
         croped.getPixels(pixels_buf, 0, croped.width, 0, 0, croped.width, croped.height)
         val imageMean = 127.5f
         val imageStd = 128f
         for (i in pixels_buf.indices) {
             val `val` = pixels_buf[i]
             data[i * 3 + 0] = ((`val` shr 16 and 0xFF) - imageMean) / imageStd
             data[i * 3 + 1] = ((`val` shr 8 and 0xFF) - imageMean) / imageStd
             data[i * 3 + 2] = ((`val` and 0xFF) - imageMean) / imageStd
         }
    }

    private fun RNetForward(RNetIn: FloatArray, boxes: Vector<Box>) {
        val num = RNetIn.size / 24 / 24 / 3
        //feed & run
        tensorFlowInferenceInterface.feed(rNetInputName, RNetIn, num.toLong(), 24L, 24L, numOfChannels)
        tensorFlowInferenceInterface.run(rNetOutputName, false)
        //fetch
        val RNetP = FloatArray(num * 2)
        val RNetB = FloatArray(num * 4)
        tensorFlowInferenceInterface.fetch(rNetOutputName[0], RNetP)
        tensorFlowInferenceInterface.fetch(rNetOutputName[1], RNetB)
        for (i in 0 until num) {
            boxes[i].score = RNetP[i * 2 + 1]
            for (j in 0..3)
                boxes[i].bbr[j] = RNetB[i * 4 + j]
        }
    }

    //Refine Net
    private fun RNet(bitmap: Bitmap, boxes: Vector<Box>): Vector<Box> {
        //RNet Input Init
        val num = boxes.size
        val RNetIn = FloatArray(num * 24 * 24 * 3)
        val curCrop = FloatArray(24 * 24 * 3)
        var RNetInIdx = 0
        for (i in 0 until num) {
            crop_and_resize(bitmap, boxes[i], 24, curCrop)
            Utils.flip_diag(curCrop, 24, 24, 3)
            for (j in curCrop.indices) RNetIn[RNetInIdx++] = curCrop[j]
        }
        //Run RNet
        RNetForward(RNetIn, boxes)
        //RNetThreshold
        for (i in 0 until num)
            if (boxes[i].score < RNetThreshold)
                boxes[i].deleted = true
        //Nms
        nms(boxes, 0.7f, "Union")
        BoundingBoxReggression(boxes)
        return Utils.updateBoxes(boxes)
    }

    private fun ONetForward(ONetIn: FloatArray, boxes: Vector<Box>) {
        val num = ONetIn.size / 48 / 48 / 3
        //feed & run
        tensorFlowInferenceInterface.feed(oNetInputName, ONetIn, num.toLong(), 48L, 48L, numOfChannels)
        tensorFlowInferenceInterface.run(oNetOutputName, false)
        //fetch
        val ONetP = FloatArray(num * 2) //prob
        val ONetB = FloatArray(num * 4) //bias
        val ONetL = FloatArray(num * 10) //landmark
        tensorFlowInferenceInterface.fetch(oNetOutputName[0], ONetP)
        tensorFlowInferenceInterface.fetch(oNetOutputName[1], ONetB)
        tensorFlowInferenceInterface.fetch(oNetOutputName[2], ONetL)

        for (i in 0 until num) {
            //prob
            boxes[i].score = ONetP[i * 2 + 1]
            //bias
            for (j in 0..3)
                boxes[i].bbr[j] = ONetB[i * 4 + j]

            //landmark
            for (j in 0..4) {
                val x = boxes[i].left() + (ONetL[i * 10 + j] * boxes[i].width()).toInt()
                val y = boxes[i].top() + (ONetL[i * 10 + j + 5] * boxes[i].height()).toInt()
                boxes[i].landmark[j] = Point(x, y)
            }
        }
    }

 //ONet
    private fun ONet(bitmap: Bitmap, boxes: Vector<Box>): Vector<Box> {
        //ONet Input Init
        val num = boxes.size
        val ONetIn = FloatArray(num * 48 * 48 * 3)
        val curCrop = FloatArray(48 * 48 * 3)
        var ONetInIdx = 0
        for (i in 0 until num) {
            crop_and_resize(bitmap, boxes[i], 48, curCrop)
            Utils.flip_diag(curCrop, 48, 48, 3)
            for (j in curCrop.indices) ONetIn[ONetInIdx++] = curCrop[j]
        }
        //Run ONet
        ONetForward(ONetIn, boxes)
        //ONetThreshold
        for (i in 0 until num)
         if (boxes[i].score < ONetThreshold)
          boxes[i].deleted = true
        BoundingBoxReggression(boxes)
        //Nms
        nms(boxes, 0.7f, "Min")
        return Utils.updateBoxes(boxes)
    }

    private fun square_limit(boxes: Vector<Box>, w: Int, h: Int) {
        //square
        for (i in boxes.indices) {
            boxes[i].toSquareShape()
            boxes[i].limit_square(w, h)
        }
    }

    fun detectFaces(bitmap: Bitmap, minFaceSize: Int): Vector<Box> {
        //【1】PNet generate candidate boxes
        var boxes = PNet(bitmap, minFaceSize)
        square_limit(boxes, bitmap.width, bitmap.height)
        //【2】RNet
        boxes = RNet(bitmap, boxes)
        square_limit(boxes, bitmap.width, bitmap.height)
        //【3】ONet
        boxes = ONet(bitmap, boxes)
        //return
        return boxes
    }
}