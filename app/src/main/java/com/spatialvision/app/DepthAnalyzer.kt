package com.spatialvision.app

import com.google.ar.core.Coordinates2d
import com.google.ar.core.Frame
import com.google.ar.core.exceptions.NotYetAvailableException
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * How camera-image coordinates (0..1) map to the phone screen as the user holds it (0..1, x = left->right,
 * y = top->bottom). The camera sensor is sideways in most phones, so this is a 90-degree rotation;
 * ARCore tells us the exact mapping, so nothing is guessed.
 */
class ViewMap(
    private val ox: Float, private val oy: Float,   // screen position of image (0,0)
    private val axX: Float, private val axY: Float, // screen change per +1 image x
    private val ayX: Float, private val ayY: Float, // screen change per +1 image y
) {
    // the whole camera image in screen coordinates (it is a bit larger than the screen)
    val minX = min(ox, min(ox + axX, min(ox + ayX, ox + axX + ayX)))
    val maxX = max(ox, max(ox + axX, max(ox + ayX, ox + axX + ayX)))
    val minY = min(oy, min(oy + axY, min(oy + ayY, oy + axY + ayY)))
    val maxY = max(oy, max(oy + axY, max(oy + ayY, oy + axY + ayY)))
    /** true when the image x axis runs along the screen x axis (no rotation) */
    val imageXAlongScreenX = abs(axX) > abs(ayX)

    fun toScreen(ix: Float, iy: Float): Pair<Float, Float> =
        Pair(ox + ix * axX + iy * ayX, oy + ix * axY + iy * ayY)

    fun toImage(sx: Float, sy: Float): Pair<Float, Float> {
        val det = axX * ayY - ayX * axY
        val dx = sx - ox
        val dy = sy - oy
        return Pair((dx * ayY - ayX * dy) / det, (axX * dy - dx * axY) / det)
    }

    companion object {
        fun from(frame: Frame): ViewMap {
            val pts = floatArrayOf(0f, 0f, 1f, 0f, 0f, 1f)
            val out = FloatArray(6)
            frame.transformCoordinates2d(Coordinates2d.IMAGE_NORMALIZED, pts, Coordinates2d.VIEW_NORMALIZED, out)
            return ViewMap(out[0], out[1], out[2] - out[0], out[3] - out[1], out[4] - out[0], out[5] - out[1])
        }
    }
}

/** Distances (metres) to the nearest obstacle in the three walking directions. NaN = nothing measured. */
data class Sectors(val left: Float, val ahead: Float, val right: Float)

class DepthSnapshot(val mm: ShortArray, val w: Int, val h: Int) {
    /** distance in metres around image point (ix, iy in 0..1): robust minimum of a small patch */
    fun metersAt(ix: Float, iy: Float): Float {
        val cx = (ix * w).toInt().coerceIn(0, w - 1)
        val cy = (iy * h).toInt().coerceIn(0, h - 1)
        val vals = ArrayList<Int>()
        for (y in max(0, cy - 2)..min(h - 1, cy + 2)) for (x in max(0, cx - 2)..min(w - 1, cx + 2)) {
            val v = mm[y * w + x].toInt() and 0xFFFF
            if (v in 1..8000) vals.add(v)
        }
        if (vals.isEmpty()) return Float.NaN
        vals.sort()
        return vals[vals.size / 2] / 1000f
    }
}

/**
 * ARCore depth image (real millimetres, about 160 x 120 pixels) -> nearest obstacle on the LEFT, AHEAD, RIGHT.
 * Only the band of the screen where things you can walk into appear is used (not the floor right in front
 * of the feet at the bottom of the screen), and a low percentile instead of the minimum, so a few noisy
 * pixels never cause a false warning.
 */
class DepthAnalyzer {
    companion object {
        const val BAND_TOP = 0.10f      // screen fraction
        const val BAND_BOTTOM = 0.72f   // below this is mostly floor
        const val MAX_MM = 8000
        const val PERCENTILE = 0.08f    // "nearest" = 8th percentile of the sector's distances
        const val MIN_PIXELS = 25
    }

    @Volatile var viewMap: ViewMap? = null
    @Volatile var snapshot: DepthSnapshot? = null
    @Volatile var geometryDirty = true

    private var mapW = 0
    private var mapH = 0
    private var sector = IntArray(0)   // per depth pixel: 0/1/2 = left/ahead/right, -1 = not used

    /** Called every camera frame: keeps the image->screen mapping current (needed by YOLO too). */
    fun updateGeometry(frame: Frame) {
        if (geometryDirty || viewMap == null) {
            viewMap = ViewMap.from(frame)
            geometryDirty = false
            mapW = 0   // rebuild the sector table with the new mapping
        }
    }

    fun process(frame: Frame): Sectors? {
        val img = try {
            frame.acquireDepthImage16Bits()
        } catch (e: NotYetAvailableException) {
            return null
        } catch (e: Exception) {
            return null
        }
        img.use {
            val w = it.width
            val h = it.height
            if (w != mapW || h != mapH) buildSectors(w, h)
            if (sector.size != w * h) return null   // mapping not ready yet: never index a wrong-size table
            val plane = it.planes[0]
            val buf = plane.buffer.order(ByteOrder.nativeOrder())
            val rowStride = plane.rowStride
            val pixStride = plane.pixelStride
            val mm = ShortArray(w * h)
            for (y in 0 until h) for (x in 0 until w) mm[y * w + x] = buf.getShort(y * rowStride + x * pixStride)
            snapshot = DepthSnapshot(mm, w, h)

            val lists = Array(3) { IntArray(w * h) }
            val counts = IntArray(3)
            for (i in mm.indices) {
                val s = sector[i]
                if (s < 0) continue
                val v = mm[i].toInt() and 0xFFFF
                if (v == 0 || v > MAX_MM) continue
                lists[s][counts[s]++] = v
            }
            fun near(s: Int): Float {
                val n = counts[s]
                if (n < MIN_PIXELS) return Float.NaN
                val a = lists[s].copyOf(n)
                a.sort()
                return a[(n * PERCENTILE).toInt()] / 1000f
            }
            return Sectors(near(0), near(1), near(2))
        }
    }

    private fun buildSectors(w: Int, h: Int) {
        val vm = viewMap ?: return
        mapW = w
        mapH = h
        sector = IntArray(w * h) { i ->
            val (sx, sy) = vm.toScreen((i % w + 0.5f) / w, (i / w + 0.5f) / h)
            if (sx < 0f || sx > 1f || sy < BAND_TOP || sy > BAND_BOTTOM) -1
            else min(2, (sx * 3).toInt())
        }
    }
}
