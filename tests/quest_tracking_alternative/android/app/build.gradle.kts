plugins { id("com.android.application") }
android {
    namespace = "org.neura.questtracking"
    compileSdk = 34
    ndkVersion = "27.2.12479018"
    defaultConfig {
        applicationId = "org.neura.questtracking"
        minSdk = 29
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"
        ndk { abiFilters += "arm64-v8a" }
        externalNativeBuild { cmake { arguments += "-DANDROID_STL=c++_static" } }
    }
    buildFeatures { prefab = true }
    externalNativeBuild { cmake { path = file("src/main/cpp/CMakeLists.txt"); version = "3.22.1" } }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}
dependencies { implementation("org.khronos.openxr:openxr_loader_for_android:1.1.53") }
