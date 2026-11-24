package com.spatialvision.app

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.graphics.Typeface
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.os.Bundle
import android.os.SystemClock
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.view.Gravity
import android.view.ViewGroup
import android.view.WindowManager
import android.widget.Button
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.TextView
import com.google.ar.core.ArCoreApk
import com.google.ar.core.Config
import com.google.ar.core.Session
import com.google.ar.core.TrackingFailureReason
import com.google.ar.core.TrackingState
import com.google.ar.core.exceptions.NotYetAvailableException
import com.google.ar.core.exceptions.UnavailableDeviceNotCompatibleException
import com.google.ar.core.exceptions.UnavailableUserDeclinedInstallationException
import java.util.Locale
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

/**
 * Spatial Vision - assistive navigation, everything on the phone.
 *   ARCore Depth  -> real distance (metres) to ANY obstacle on the left / ahead / right (walls and doors too)
 *   YOLOv8n       -> the name of the obstacle when it is one of 80 known things (person, chair, car...)
 *   Guide         -> voice + parking-sensor vibration
 * Tap the screen = "where am I". Long-press or SPEAK = voice command. PAUSE button.
 */
class MainActivity : Activity(), GLSurfaceView.Renderer {
    private lateinit var glView: GLSurfaceView
    private lateinit var status: TextView
    private lateinit var pauseBtn: Button
    private lateinit var guide: Guide
    private var session: Session? = null
    private var installRequested = false
    private val background = BackgroundRenderer()
    private val depth = DepthAnalyzer()
    private var detector: Detector? = null
    private var viewW = 0
    private var viewH = 0
    private var viewportChanged = false
    private var lastDepthAt = 0L
    private var lastYoloAt = 0L
    private var lastUiAt = 0L
    private var frames = 0
    private var fpsT0 = 0L
    private var fps = 0f
    private var recognizer: SpeechRecognizer? = null
    private var depthSupported = true
    @Volatile private var trackingText = "starting"
    @Volatile private var depthOk = false

    // ------------------------------------------------------------------ setup
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        guide = Guide(this)

        glView = GLSurfaceView(this).apply {
            preserveEGLContextOnPause = true
            setEGLContextClientVersion(2)
            setEGLConfigChooser(8, 8, 8, 8, 16, 0)
            setRenderer(this@MainActivity)
            renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
            setWillNotDraw(false)
            contentDescription = "Camera view. Tap to hear what is around you. Long press for a voice command."
            setOnClickListener { guide.say(guide.describe()) }
            setOnLongClickListener { listen(); true }
        }
        status = TextView(this).apply {
            setTextColor(Color.rgb(255, 212, 0))
            setBackgroundColor(Color.argb(200, 0, 0, 0))
            textSize = 20f
            typeface = Typeface.DEFAULT_BOLD
            setPadding(32, 48, 32, 24)
            text = "Starting…"
        }
        pauseBtn = bigButton("PAUSE") {
            guide.setPaused(!guide.paused)
            pauseBtn.text = if (guide.paused) "RESUME" else "PAUSE"
        }
        val speakBtn = bigButton("SPEAK") { listen() }
        val buttons = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            setBackgroundColor(Color.argb(220, 0, 0, 0))
            setPadding(16, 16, 16, 40)
            addView(pauseBtn, LinearLayout.LayoutParams(0, 200, 1f).apply { setMargins(12, 0, 12, 0) })
            addView(speakBtn, LinearLayout.LayoutParams(0, 200, 1f).apply { setMargins(12, 0, 12, 0) })
        }
        val root = FrameLayout(this).apply {
            addView(glView, FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT))
            addView(status, FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT, Gravity.TOP))
            addView(buttons, FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT, Gravity.BOTTOM))
        }
        setContentView(root)

        try {
            detector = Detector(this) { list -> runOnUiThread { guide.named = list } }
        } catch (e: Throwable) {
            detector = null   // naming is optional; distance warnings still work
        }
        guide.startHaptics()

        val need = arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO)
            .filter { checkSelfPermission(it) != PackageManager.PERMISSION_GRANTED }
        if (need.isNotEmpty()) requestPermissions(need.toTypedArray(), 1)
    }

    private fun bigButton(text: String, onClick: () -> Unit) = Button(this).apply {
        this.text = text
        textSize = 22f
        typeface = Typeface.DEFAULT_BOLD
        setTextColor(Color.BLACK)
        setBackgroundColor(Color.rgb(255, 212, 0))
        setOnClickListener { onClick() }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            status.text = "Camera permission is needed. Open Settings > Apps > Spatial Vision > Permissions."
            guide.say("I need camera permission to work.")
        }   // when granted, onResume() runs by itself after the permission dialog closes
    }

    override fun onResume() {
        super.onResume()
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) return
        if (session == null) {
            try {
                when (ArCoreApk.getInstance().requestInstall(this, !installRequested)) {
                    ArCoreApk.InstallStatus.INSTALL_REQUESTED -> { installRequested = true; return }
                    else -> {}
                }
                val s = Session(this)
                val config = Config(s)
                if (s.isDepthModeSupported(Config.DepthMode.AUTOMATIC)) {
                    config.depthMode = Config.DepthMode.AUTOMATIC
                } else {
                    depthSupported = false
                    guide.say("Sorry, this phone does not support depth measurement.")
                }
                config.focusMode = Config.FocusMode.AUTO
                config.planeFindingMode = Config.PlaneFindingMode.DISABLED
                config.lightEstimationMode = Config.LightEstimationMode.DISABLED
                config.updateMode = Config.UpdateMode.LATEST_CAMERA_IMAGE
                s.configure(config)
                session = s
            } catch (e: UnavailableUserDeclinedInstallationException) {
                status.text = "Please install \"Google Play Services for AR\" from the Play Store."
                return
            } catch (e: UnavailableDeviceNotCompatibleException) {
                status.text = "This phone does not support ARCore."
                return
            } catch (e: Exception) {
                status.text = "Could not start ARCore: ${e.javaClass.simpleName}"
                return
            }
        }
        try {
            session?.resume()
        } catch (e: Exception) {
            status.text = "Camera is busy or unavailable. Close other camera apps and reopen."
            session = null
            return
        }
        glView.onResume()
        depth.geometryDirty = true
    }

    override fun onPause() {
        super.onPause()
        recognizer?.cancel()
        if (session != null) {
            glView.onPause()
            session?.pause()
        }
    }

    override fun onDestroy() {
        recognizer?.destroy()
        guide.shutdown()
        session?.close()
        session = null
        super.onDestroy()
    }

    // ------------------------------------------------------------------ camera loop (GL thread)
    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        GLES20.glClearColor(0f, 0f, 0f, 1f)
        background.createOnGlThread()
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        GLES20.glViewport(0, 0, width, height)
        viewW = width
        viewH = height
        viewportChanged = true
    }

    override fun onDrawFrame(gl: GL10?) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT or GLES20.GL_DEPTH_BUFFER_BIT)
        val s = session ?: return
        if (viewportChanged) {
            @Suppress("DEPRECATION")
            val rotation = windowManager.defaultDisplay.rotation
            s.setDisplayGeometry(rotation, viewW, viewH)
            viewportChanged = false
        }
        s.setCameraTextureName(background.textureId)
        val frame = try { s.update() } catch (e: Exception) { return }
        if (frame.hasDisplayGeometryChanged()) depth.geometryDirty = true
        if (frame.timestamp != 0L) depth.updateGeometry(frame)
        background.draw(frame)

        val now = SystemClock.uptimeMillis()
        frames++
        if (now - fpsT0 >= 1000) { fps = frames * 1000f / (now - fpsT0); frames = 0; fpsT0 = now }

        val cam = frame.camera
        val tracking = cam.trackingState == TrackingState.TRACKING
        val reason = cam.trackingFailureReason
        trackingText = if (tracking) "tracking" else "${cam.trackingState.name.lowercase()} (${reason.name.lowercase()})"
        if (now - lastDepthAt >= 100) {   // 10 distance updates per second
            lastDepthAt = now
            val r = if (tracking) depth.process(frame) else null
            depthOk = r != null
            runOnUiThread { if (r != null) guide.update(r) else guide.noDepth(hintFor(tracking, reason)) }
        }
        val det = detector
        val vm = depth.viewMap
        if (det != null && vm != null && !det.busy && now - lastYoloAt >= 200) {
            lastYoloAt = now
            try {
                frame.acquireCameraImage().use { img -> det.submit(img, vm, depth.snapshot) }
            } catch (e: NotYetAvailableException) {
            } catch (e: Exception) {
            }
        }
        if (now - lastUiAt >= 250) {
            lastUiAt = now
            runOnUiThread { showStatus(tracking) }
        }
    }

    private fun hintFor(tracking: Boolean, reason: TrackingFailureReason): String = when {
        !depthSupported -> "This phone cannot measure distance."
        tracking -> "Hold on, measuring distance. Move the phone slowly left and right."
        reason == TrackingFailureReason.INSUFFICIENT_LIGHT -> "It is too dark. Please turn on a light."
        reason == TrackingFailureReason.EXCESSIVE_MOTION -> "Moving too fast. Move the phone more slowly."
        reason == TrackingFailureReason.INSUFFICIENT_FEATURES -> "Point the camera at the room, not at a plain wall, and move it slowly."
        else -> "Move the phone slowly left and right."
    }

    // ------------------------------------------------------------------ screen text (for sighted helpers)
    private fun m(v: Float) = if (v.isNaN() || v > 8f) "  —  " else String.format(Locale.US, "%.1f m", v)

    private fun showStatus(tracking: Boolean) {
        val s = guide.sectors
        val lines = StringBuilder()
        if (guide.paused) lines.append("PAUSED\n")
        if (!depthSupported) lines.append("This phone does not support ARCore Depth\n")
        else if (s == null || !tracking) lines.append("Move the phone slowly left and right…\n")
        else lines.append("LEFT ${m(s.left)}   AHEAD ${m(s.ahead)}   RIGHT ${m(s.right)}\n")
        if (guide.lastAlert.isNotEmpty()) lines.append(guide.lastAlert).append('\n')
        val names = guide.named.joinToString(", ") { "${it.label} ${m(it.meters).trim()}" }
        if (names.isNotEmpty()) lines.append("Seen: ").append(names).append('\n')
        lines.append(String.format(Locale.US, "%.0f FPS · YOLO %s · AR %s · depth %s", fps,
            if (detector == null) "off" else "${detector?.lastMs ?: 0} ms", trackingText,
            if (!depthSupported) "unsupported" else if (depthOk) "ok" else "waiting"))
        status.text = lines.toString()
        val danger = s != null && (s.ahead < Guide.WARN_AHEAD) && !guide.paused
        status.setBackgroundColor(if (danger) Color.argb(220, 200, 30, 30) else Color.argb(200, 0, 0, 0))
        status.setTextColor(if (danger) Color.WHITE else Color.rgb(255, 212, 0))
    }

    // ------------------------------------------------------------------ voice commands
    private fun listen() {
        if (!SpeechRecognizer.isRecognitionAvailable(this)) {
            guide.say("Voice commands are not available on this phone.")
            return
        }
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), 2)
            return
        }
        val r = recognizer ?: SpeechRecognizer.createSpeechRecognizer(this).also { recognizer = it }
        r.setRecognitionListener(object : RecognitionListener {
            override fun onResults(results: Bundle?) {
                val text = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.firstOrNull() ?: ""
                handleCommand(text)
            }
            override fun onError(error: Int) { guide.say("I did not hear a command.") }
            override fun onReadyForSpeech(params: Bundle?) {}
            override fun onBeginningOfSpeech() {}
            override fun onRmsChanged(rmsdB: Float) {}
            override fun onBufferReceived(buffer: ByteArray?) {}
            override fun onEndOfSpeech() {}
            override fun onPartialResults(partialResults: Bundle?) {}
            override fun onEvent(eventType: Int, params: Bundle?) {}
        })
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, "en-IN")
            putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
        }
        guide.say("Listening.")
        status.postDelayed({ r.startListening(intent) }, 700)   // let "Listening" finish first
    }

    private fun handleCommand(raw: String) {
        val t = raw.lowercase(Locale.US)
        when {
            Regex("where am i|what'?s around|around me|describe|what do you see").containsMatchIn(t) -> guide.say(guide.describe())
            Regex("\\b(pause|stop|quiet)\\b").containsMatchIn(t) -> { if (!guide.paused) pauseBtn.performClick() }
            Regex("\\b(resume|start|continue|go)\\b").containsMatchIn(t) -> { if (guide.paused) pauseBtn.performClick() }
            Regex("\\b(repeat|again)\\b").containsMatchIn(t) -> guide.say(guide.lastAlert.ifEmpty { "No warning right now." })
            Regex("\\b(far|farther|early|earlier)\\b").containsMatchIn(t) -> {
                Guide.WARN_AHEAD = 1.8f; guide.say("I will warn you earlier, at 1.8 meters.")
            }
            Regex("\\b(near|nearer|late|later|normal)\\b").containsMatchIn(t) -> {
                Guide.WARN_AHEAD = 1.2f; guide.say("Normal warning distance, 1.2 meters.")
            }
            Regex("\\bhelp\\b").containsMatchIn(t) -> guide.say(
                "Tap the screen to hear what is around you. Say: where am I, pause, resume, repeat, " +
                "warn earlier, or normal distance.")
            else -> guide.say("Sorry, I did not understand. Say help to hear the commands.")
        }
    }
}
