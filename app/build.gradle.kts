plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.spatialvision.app"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.spatialvision.app"
        minSdk = 24
        targetSdk = 34
        versionCode = 2
        versionName = "1.1"
    }

    // Fixed key, so a newer APK installs over the old one without uninstalling.
    signingConfigs {
        getByName("debug") {
            storeFile = file("debug.keystore")
            storePassword = "android"
            keyAlias = "androiddebugkey"
            keyPassword = "android"
        }
    }
    buildTypes {
        getByName("debug") { signingConfig = signingConfigs.getByName("debug") }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    androidResources { noCompress += "onnx" }
}

dependencies {
    implementation("com.google.ar:core:1.45.0")                          // ARCore: real distance in metres
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.20.0") // YOLOv8n: names of objects
}
