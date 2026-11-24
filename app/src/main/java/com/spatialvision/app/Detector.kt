package com.spatialvision.app

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.media.Image
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.min

/** One named object: where it is on the screen (0..1) and how far (metres, from ARCore depth). */
data class Named(val label: String, val conf: Float, val screenX: Float, val screenY: Float, val meters: Float) {
    val sector: Int get() = min(2, max(0, (screenX * 3).toInt()))
}

/**
 * YOLOv8n (320 x 320, ONNX Runtime on the phone's CPU) - only gives NAMES to things.
 * Warnings never wait for it: the distance warning comes from ARCore depth, which sees every obstacle
 * (walls and doors too), named or not.
 */
class Detector(ctx: Context, private val onResult: (List<Named>) -> Unit) {
    companion object {
        const val SIZE = 320
        const val MIN_CONF = 0.45f
        const val IOU = 0.6f
        val NAMES = arrayOf("person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
            "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
            "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
            "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
            "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
            "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
            "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush")
    }

    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputName: String
    private val pool = Executors.newSingleThreadExecutor()
    @Volatile var busy = false
        private set
    @Volatile var lastMs = 0L
        private set

    init {
        val bytes = ctx.assets.open("yolov8n_320.onnx").use { it.readBytes() }
        val opts = OrtSession.SessionOptions().apply { setIntraOpNumThreads(4) }
        session = env.createSession(bytes, opts)
        inputName = session.inputNames.iterator().next()
    }

    /** Called on the camera (GL) thread with the latest camera image; the image is closed by the caller. */
    fun submit(img: Image, vm: ViewMap, depth: DepthSnapshot?) {
        if (busy) return
        busy = true
        val input: FloatBuffer
        val geo: Letterbox
        try {
            val r = toUprightTensor(img, vm)
            input = r.first
            geo = r.second
        } catch (e: Exception) {
            busy = false
            return
        }
        pool.execute {
            try {
                val t0 = System.currentTimeMillis()
                val dets = run(input, geo, vm, depth)
                lastMs = System.currentTimeMillis() - t0
                onResult(dets)
            } catch (e: Exception) {
                // keep going: naming is optional, the depth warnings do not depend on it
            } finally {
                busy = false
            }
        }
    }

    /** where the upright camera picture sits inside the 320 x 320 square */
    class Letterbox(val uprightW: Int, val uprightH: Int, val scale: Float, val padX: Float, val padY: Float)

    /**
     * YUV camera image -> upright RGB picture (as the user holds the phone), letterboxed into 320 x 320,
     * RGB 0..1 in CHW order (what YOLOv8 expects). Rotation comes from ARCore's ViewMap.
     */
    private fun toUprightTensor(img: Image, vm: ViewMap): Pair<FloatBuffer, Letterbox> {
        val iw = img.width
        val ih = img.height
        val uw = if (vm.imageXAlongScreenX) iw else ih   // upright picture size in pixels
        val uh = if (vm.imageXAlongScreenX) ih else iw
        val scale = min(SIZE.toFloat() / uw, SIZE.toFloat() / uh)
        val nw = uw * scale
        val nh = uh * scale
        val padX = (SIZE - nw) / 2f
        val padY = (SIZE - nh) / 2f

        val yP = img.planes[0]; val uP = img.planes[1]; val vP = img.planes[2]
        val yB = yP.buffer; val uB = uP.buffer; val vB = vP.buffer
        val yRow = yP.rowStride; val yPix = yP.pixelStride
        val uRow = uP.rowStride; val uPix = uP.pixelStride
        val vRow = vP.rowStride; val vPix = vP.pixelStride

        val n = SIZE * SIZE
        val out = FloatArray(3 * n)
        val pad = 114f / 255f
        for (oy in 0 until SIZE) for (ox in 0 until SIZE) {
            val o = oy * SIZE + ox
            val ux = (ox + 0.5f - padX) / nw   // 0..1 across the upright picture
            val uy = (oy + 0.5f - padY) / nh
            if (ux < 0f || ux >= 1f || uy < 0f || uy >= 1f) {
                out[o] = pad; out[n + o] = pad; out[2 * n + o] = pad
                continue
            }
            // upright picture -> screen coordinates -> camera image coordinates
            val sx = vm.minX + ux * (vm.maxX - vm.minX)
            val sy = vm.minY + uy * (vm.maxY - vm.minY)
            val (ixn, iyn) = vm.toImage(sx, sy)
            val x = (ixn * iw).toInt().coerceIn(0, iw - 1)
            val y = (iyn * ih).toInt().coerceIn(0, ih - 1)
            val yy = (yB.get(y * yRow + x * yPix).toInt() and 0xFF).toFloat()
            val u = (uB.get((y / 2) * uRow + (x / 2) * uPix).toInt() and 0xFF) - 128f
            val v = (vB.get((y / 2) * vRow + (x / 2) * vPix).toInt() and 0xFF) - 128f
            out[o] = ((yy + 1.402f * v) / 255f).coerceIn(0f, 1f)
            out[n + o] = ((yy - 0.344136f * u - 0.714136f * v) / 255f).coerceIn(0f, 1f)
            out[2 * n + o] = ((yy + 1.772f * u) / 255f).coerceIn(0f, 1f)
        }
        val fb = ByteBuffer.allocateDirect(out.size * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
        fb.put(out)
        fb.position(0)
        return Pair(fb, Letterbox(uw, uh, scale, padX, padY))
    }

    private class Cand(val cls: Int, val conf: Float, val x1: Float, val y1: Float, val x2: Float, val y2: Float)

    private fun iou(a: Cand, b: Cand): Float {
        val ix = max(0f, min(a.x2, b.x2) - max(a.x1, b.x1))
        val iy = max(0f, min(a.y2, b.y2) - max(a.y1, b.y1))
        val i = ix * iy
        return i / ((a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - i + 1e-6f)
    }

    private fun run(input: FloatBuffer, g: Letterbox, vm: ViewMap, depth: DepthSnapshot?): List<Named> {
        val shape = longArrayOf(1, 3, SIZE.toLong(), SIZE.toLong())
        OnnxTensor.createTensor(env, input, shape).use { tensor ->
            session.run(mapOf(inputName to tensor)).use { res ->
                @Suppress("UNCHECKED_CAST")
                val o = (res[0].value as Array<Array<FloatArray>>)[0]   // [84][anchors]
                val anchors = o[0].size
                val cands = ArrayList<Cand>()
                for (a in 0 until anchors) {
                    var best = 0f
                    var cls = -1
                    for (c in 4 until o.size) if (o[c][a] > best) { best = o[c][a]; cls = c - 4 }
                    if (best < MIN_CONF) continue
                    val cx = o[0][a]; val cy = o[1][a]; val w = o[2][a]; val h = o[3][a]
                    cands.add(Cand(cls, best, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
                }
                cands.sortByDescending { it.conf }
                val keep = ArrayList<Cand>()
                for (c in cands) {
                    if (keep.any { it.cls == c.cls && iou(it, c) > IOU }) continue
                    keep.add(c)
                    if (keep.size >= 20) break
                }
                return keep.map { c ->
                    // box centre: 320-square -> upright 0..1 -> screen 0..1 -> camera image 0..1 (for depth)
                    val ux = (((c.x1 + c.x2) / 2 - g.padX) / (g.uprightW * g.scale)).coerceIn(0f, 1f)
                    val uy = (((c.y1 + c.y2) / 2 - g.padY) / (g.uprightH * g.scale)).coerceIn(0f, 1f)
                    val sx = vm.minX + ux * (vm.maxX - vm.minX)
                    val sy = vm.minY + uy * (vm.maxY - vm.minY)
                    val (ix, iy) = vm.toImage(sx, sy)
                    val m = depth?.metersAt(ix, iy) ?: Float.NaN
                    Named(NAMES[c.cls], c.conf, sx.coerceIn(0f, 1f), sy, m)
                }
            }
        }
    }
}
