package com.spatialvision.app

import android.content.Context
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.os.VibrationEffect
import android.os.Vibrator
import android.speech.tts.TextToSpeech
import java.util.Locale
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Turns distances into what the user hears and feels. Runs on the main thread.
 *  - Vibration works like a car parking sensor: the closer the obstacle ahead, the faster the pulses.
 *  - Speech: a new obstacle is announced at once; while it stays, it is repeated every 4 s, at most 3 times
 *    (the vibration keeps going, so the user is never left without a warning).
 */
class Guide(ctx: Context) : TextToSpeech.OnInitListener {
    companion object {
        var WARN_AHEAD = 1.2f   // metres: speak when something in front is closer than this
        const val WARN_SIDE = 0.6f
        const val NEAR_TOO = 0.3f   // a side counts as "free" if its obstacle is this much farther
        const val REPEAT_MS = 4000L
        const val MAX_REPEATS = 3
        const val CLEAR_MS = 1500L
        val SIDE = arrayOf("on your left", "ahead", "on your right")
    }

    private val tts = TextToSpeech(ctx, this)
    private var ttsReady = false
    private val vibrator: Vibrator? = ctx.getSystemService(Vibrator::class.java)
    private val ui = Handler(Looper.getMainLooper())

    var paused = false
        private set
    var sectors: Sectors? = null
        private set
    var named: List<Named> = emptyList()
    private var lastSectorsAt = 0L
    private var lastKey = ""
    private var lastSpokenAt = 0L
    private var repeats = 0
    private var clearSince = 0L
    private var lastBuzz = 0L
    private var lastHintAt = 0L
    var lastAlert = ""
        private set
    val startedAt = SystemClock.uptimeMillis()

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            val r = tts.setLanguage(Locale("en", "IN"))
            if (r == TextToSpeech.LANG_MISSING_DATA || r == TextToSpeech.LANG_NOT_SUPPORTED) tts.setLanguage(Locale.US)
            tts.setSpeechRate(1.1f)
            ttsReady = true
            say("Spatial Vision started. Move the phone slowly left and right for a moment. " +
                "Tap the screen to hear what is around you.")
        }
    }

    fun say(text: String, urgent: Boolean = true) {
        if (!ttsReady) return
        tts.speak(text, if (urgent) TextToSpeech.QUEUE_FLUSH else TextToSpeech.QUEUE_ADD, null, "svai")
    }

    fun shutdown() {
        ui.removeCallbacksAndMessages(null)
        tts.stop()
        tts.shutdown()
    }

    fun setPaused(p: Boolean) {
        paused = p
        say(if (p) "Guidance paused." else "Guidance resumed.")
    }

    fun startHaptics() {
        ui.post(object : Runnable {
            override fun run() {
                hapticTick()
                ui.postDelayed(this, 50)
            }
        })
    }

    // ------------------------------------------------------------------ distances in
    fun update(s: Sectors) {
        sectors = s
        lastSectorsAt = SystemClock.uptimeMillis()
        if (paused) return
        val now = lastSectorsAt
        val key = buildString {
            if (s.left < WARN_SIDE) append("L")
            if (s.ahead < WARN_AHEAD) append("A")
            if (s.right < WARN_SIDE) append("R")
        }
        if (key.isEmpty()) {
            if (clearSince == 0L) clearSince = now
            if (now - clearSince >= CLEAR_MS) { lastKey = ""; repeats = 0 }
            return
        }
        clearSince = 0L
        val isNew = key != lastKey
        if (isNew || (now - lastSpokenAt >= REPEAT_MS && repeats < MAX_REPEATS)) {
            if (isNew && now - lastSpokenAt < 1200 && lastKey.isNotEmpty()) return   // do not cut every word off
            if (isNew) repeats = 0
            lastAlert = alertText(s)
            say(lastAlert)
            lastSpokenAt = now
            repeats++
            lastKey = key
        }
    }

    /** spoken when depth is not available yet (ARCore needs a little camera movement to start) */
    fun noDepth(hint: String) {
        val now = SystemClock.uptimeMillis()
        if (now - startedAt > 6000 && now - lastSectorsAt > 3000 && now - lastHintAt > 8000 && !paused) {
            lastHintAt = now
            say(hint)
        }
    }

    // ------------------------------------------------------------------ sentences
    private fun dist(m: Float): String = when {
        m.isNaN() -> "unknown distance"
        m < 1f -> "${((m * 100) / 10).roundToInt() * 10} centimeters"
        else -> String.format(Locale.US, "%.1f meters", m)
    }

    /** the name of what is in a sector at about that distance, else "obstacle" (walls, doors...) */
    private fun nameIn(sector: Int, m: Float): String {
        val fresh = named.filter { it.sector == sector && !it.meters.isNaN() && kotlin.math.abs(it.meters - m) < 0.5f }
        return fresh.minByOrNull { it.meters }?.label ?: "obstacle"
    }

    private fun far(v: Float) = v.isNaN() || v > 8f

    private fun alertText(s: Sectors): String {
        val parts = ArrayList<String>()
        if (s.ahead < WARN_AHEAD) {
            val l = if (far(s.left)) 99f else s.left
            val r = if (far(s.right)) 99f else s.right
            val hint = when {
                l < s.ahead + NEAR_TOO && r < s.ahead + NEAR_TOO -> "Stop."
                l >= r -> "Move left."
                else -> "Move right."
            }
            return "${nameIn(1, s.ahead).replaceFirstChar { it.uppercase() }} ahead, ${dist(s.ahead)}. $hint"
        }
        if (s.left < WARN_SIDE) parts.add("${nameIn(0, s.left)} on your left, ${dist(s.left)}")
        if (s.right < WARN_SIDE) parts.add("${nameIn(2, s.right)} on your right, ${dist(s.right)}")
        return parts.joinToString(". ").replaceFirstChar { it.uppercase() } + "."
    }

    /** "where am I" */
    fun describe(): String {
        val s = sectors
        if (s == null || SystemClock.uptimeMillis() - lastSectorsAt > 3000)
            return "I cannot measure distance yet. Move the phone slowly left and right."
        val vals = floatArrayOf(s.left, s.ahead, s.right)
        val order = intArrayOf(1, 0, 2)
        val parts = order.map { i ->
            val where = if (i == 1) "Ahead" else if (i == 0) "Left" else "Right"
            val m = vals[i]
            if (far(m) || m > 3f) "$where, clear" else "$where, ${nameIn(i, m)}, ${dist(m)}"
        }
        val extra = named.filter { !it.meters.isNaN() }.sortedBy { it.meters }.take(3)
            .filter { n -> parts.none { p -> p.contains(n.label) } }
            .joinToString(", ") { "${it.label} ${SIDE[it.sector]}, ${dist(it.meters)}" }
        return parts.joinToString(". ") + "." + if (extra.isNotEmpty()) " I also see $extra." else ""
    }

    // ------------------------------------------------------------------ vibration (parking-sensor style)
    private fun hapticTick() {
        val s = sectors ?: return
        val now = SystemClock.uptimeMillis()
        if (paused || now - lastSectorsAt > 1500) return
        var closest = if (s.ahead < WARN_AHEAD) s.ahead else Float.NaN
        if (s.left < WARN_SIDE) closest = if (closest.isNaN()) s.left + 0.3f else min(closest, s.left + 0.3f)
        if (s.right < WARN_SIDE) closest = if (closest.isNaN()) s.right + 0.3f else min(closest, s.right + 0.3f)
        if (closest.isNaN()) return
        // 0.3 m -> buzz every 120 ms ... 1.2 m -> every 900 ms
        val f = ((closest - 0.3f) / (WARN_AHEAD - 0.3f)).coerceIn(0f, 1f)
        val interval = (120 + f * 780).toLong()
        if (now - lastBuzz < interval) return
        lastBuzz = now
        val v = vibrator ?: return
        if (Build.VERSION.SDK_INT >= 26) v.vibrate(VibrationEffect.createOneShot(45, VibrationEffect.DEFAULT_AMPLITUDE))
        else {
            @Suppress("DEPRECATION")
            v.vibrate(45)
        }
    }
}
