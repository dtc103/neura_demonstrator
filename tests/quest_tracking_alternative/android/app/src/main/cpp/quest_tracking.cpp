// Controller-only OpenXR app. Starts streaming automatically when launched.
#include <EGL/egl.h>
#include <GLES3/gl3.h>
#include <jni.h>
#include <android/log.h>
#include <android_native_app_glue.h>
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
#include <array>
#include <chrono>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(XrResult result, const char* operation) {
    if (XR_FAILED(result)) throw std::runtime_error(std::string(operation) + ": " + std::to_string(result));
}
#define XR(call) check((call), #call)
int64_t nowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
XrPosef identity() { XrPosef pose{}; pose.orientation.w = 1; return pose; }

struct Stream {
    int listener = -1, client = -1;
    void start() {
        listener = socket(AF_INET, SOCK_STREAM, 0);
        if (listener < 0) throw std::runtime_error("socket failed");
        int yes = 1;
        setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
        sockaddr_in addr{};
        addr.sin_family = AF_INET; addr.sin_port = htons(9201);
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        if (bind(listener, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) || listen(listener, 1))
            throw std::runtime_error("cannot listen on port 9201");
        fcntl(listener, F_SETFL, fcntl(listener, F_GETFL) | O_NONBLOCK);
    }
    void send(const std::string& line) {
        if (client < 0) {
            client = accept(listener, nullptr, nullptr);
            if (client < 0) return;
            int yes = 1;
            setsockopt(client, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));
            fcntl(client, F_SETFL, fcntl(client, F_GETFL) | O_NONBLOCK);
        }
        std::string message = line + "\n";
        // Drop the connection on partial writes; never queue stale poses.
        if (::send(client, message.data(), message.size(), MSG_NOSIGNAL | MSG_DONTWAIT)
            != static_cast<ssize_t>(message.size())) { close(client); client = -1; }
    }
    ~Stream() { if (client >= 0) close(client); if (listener >= 0) close(listener); }
};

struct App {
    android_app* android;
    XrInstance instance = XR_NULL_HANDLE;
    XrSession session = XR_NULL_HANDLE;
    XrSystemId system = XR_NULL_SYSTEM_ID;
    XrSpace reference = XR_NULL_HANDLE;
    XrReferenceSpaceType referenceType = XR_REFERENCE_SPACE_TYPE_LOCAL;
    XrActionSet actions = XR_NULL_HANDLE;
    XrAction poseAction = XR_NULL_HANDLE, triggerAction = XR_NULL_HANDLE, selectAction = XR_NULL_HANDLE;
    std::array<XrPath, 2> hands{};
    std::array<XrSpace, 2> grips{};
    EGLDisplay display = EGL_NO_DISPLAY;
    EGLContext context = EGL_NO_CONTEXT;
    EGLSurface surface = EGL_NO_SURFACE;
    EGLConfig config{};
    XrSwapchain swapchain = XR_NULL_HANDLE;
    std::vector<XrSwapchainImageOpenGLESKHR> images;
    GLuint framebuffer = 0;
    uint32_t width = 0, height = 0;
    bool running = false, exit = false;
    XrSessionState state = XR_SESSION_STATE_UNKNOWN;
    uint64_t epoch = 0, sequence = 0;
    std::string source = "native-" + std::to_string(nowNs());
    Stream stream;

    XrPath path(const char* value) { XrPath p; XR(xrStringToPath(instance, value, &p)); return p; }
    void init() {
        PFN_xrInitializeLoaderKHR initialize = nullptr;
        XR(xrGetInstanceProcAddr(XR_NULL_HANDLE, "xrInitializeLoaderKHR", reinterpret_cast<PFN_xrVoidFunction*>(&initialize)));
        XrLoaderInitInfoAndroidKHR loader{XR_TYPE_LOADER_INIT_INFO_ANDROID_KHR};
        loader.applicationVM = android->activity->vm;
        loader.applicationContext = android->activity->clazz;
        XR(initialize(reinterpret_cast<XrLoaderInitInfoBaseHeaderKHR*>(&loader)));
        uint32_t count = 0;
        XR(xrEnumerateInstanceExtensionProperties(nullptr, 0, &count, nullptr));
        std::vector<XrExtensionProperties> extensions(count, {XR_TYPE_EXTENSION_PROPERTIES});
        XR(xrEnumerateInstanceExtensionProperties(nullptr, count, &count, extensions.data()));
        std::vector<const char*> enabled{XR_KHR_ANDROID_CREATE_INSTANCE_EXTENSION_NAME, XR_KHR_OPENGL_ES_ENABLE_EXTENSION_NAME};
        bool touchPlus = false;
        for (const auto& e : extensions) if (!strcmp(e.extensionName, "XR_META_touch_controller_plus")) touchPlus = true;
        if (touchPlus) enabled.push_back("XR_META_touch_controller_plus");
        XrInstanceCreateInfoAndroidKHR androidInfo{XR_TYPE_INSTANCE_CREATE_INFO_ANDROID_KHR};
        androidInfo.applicationVM = android->activity->vm; androidInfo.applicationActivity = android->activity->clazz;
        XrInstanceCreateInfo create{XR_TYPE_INSTANCE_CREATE_INFO}; create.next = &androidInfo;
        strcpy(create.applicationInfo.applicationName, "Neura Controller Tracking");
        create.applicationInfo.applicationVersion = 1; create.applicationInfo.apiVersion = XR_CURRENT_API_VERSION;
        create.enabledExtensionCount = enabled.size(); create.enabledExtensionNames = enabled.data();
        XR(xrCreateInstance(&create, &instance));
        XrSystemGetInfo get{XR_TYPE_SYSTEM_GET_INFO}; get.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
        XR(xrGetSystem(instance, &get, &system));
        PFN_xrGetOpenGLESGraphicsRequirementsKHR requirements = nullptr;
        XR(xrGetInstanceProcAddr(instance, "xrGetOpenGLESGraphicsRequirementsKHR", reinterpret_cast<PFN_xrVoidFunction*>(&requirements)));
        XrGraphicsRequirementsOpenGLESKHR req{XR_TYPE_GRAPHICS_REQUIREMENTS_OPENGL_ES_KHR};
        XR(requirements(instance, system, &req));
        display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        if (!eglInitialize(display, nullptr, nullptr)) throw std::runtime_error("eglInitialize failed");
        eglBindAPI(EGL_OPENGL_ES_API);
        const EGLint attrs[]{EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT, EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
            EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8, EGL_NONE};
        EGLint configs = 0;
        if (!eglChooseConfig(display, attrs, &config, 1, &configs) || !configs) throw std::runtime_error("EGL config unavailable");
        const EGLint ctx[]{EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
        context = eglCreateContext(display, config, EGL_NO_CONTEXT, ctx);
        const EGLint pbuffer[]{EGL_WIDTH, 16, EGL_HEIGHT, 16, EGL_NONE};
        surface = eglCreatePbufferSurface(display, config, pbuffer);
        if (!eglMakeCurrent(display, surface, surface, context)) throw std::runtime_error("eglMakeCurrent failed");
        XrGraphicsBindingOpenGLESAndroidKHR graphics{XR_TYPE_GRAPHICS_BINDING_OPENGL_ES_ANDROID_KHR};
        graphics.display = display; graphics.config = config; graphics.context = context;
        XrSessionCreateInfo createSession{XR_TYPE_SESSION_CREATE_INFO}; createSession.next = &graphics; createSession.systemId = system;
        XR(xrCreateSession(instance, &createSession, &session));
        XR(xrEnumerateReferenceSpaces(session, 0, &count, nullptr));
        std::vector<XrReferenceSpaceType> spaces(count);
        XR(xrEnumerateReferenceSpaces(session, count, &count, spaces.data()));
        for (auto type : spaces) if (type == XR_REFERENCE_SPACE_TYPE_STAGE) referenceType = type;
        XrReferenceSpaceCreateInfo ref{XR_TYPE_REFERENCE_SPACE_CREATE_INFO}; ref.referenceSpaceType = referenceType;
        ref.poseInReferenceSpace = identity(); XR(xrCreateReferenceSpace(session, &ref, &reference));
        makeActions(touchPlus);
        makeSwapchain();
        stream.start();
        __android_log_print(ANDROID_LOG_INFO, "NeuraQuest", "Streaming automatically on TCP 9201, fixed %s reference",
            referenceType == XR_REFERENCE_SPACE_TYPE_STAGE ? "stage" : "local");
    }

    XrAction action(const char* name, XrActionType type) {
        XrActionCreateInfo info{XR_TYPE_ACTION_CREATE_INFO}; info.actionType = type;
        strcpy(info.actionName, name); strcpy(info.localizedActionName, name);
        info.countSubactionPaths = hands.size(); info.subactionPaths = hands.data();
        XrAction result; XR(xrCreateAction(actions, &info, &result)); return result;
    }
    void makeActions(bool plus) {
        hands = {path("/user/hand/left"), path("/user/hand/right")};
        XrActionSetCreateInfo info{XR_TYPE_ACTION_SET_CREATE_INFO};
        strcpy(info.actionSetName, "controllers"); strcpy(info.localizedActionSetName, "Controllers");
        XR(xrCreateActionSet(instance, &info, &actions));
        poseAction = action("grip_pose", XR_ACTION_TYPE_POSE_INPUT);
        triggerAction = action("trigger", XR_ACTION_TYPE_FLOAT_INPUT);
        selectAction = action("selection", XR_ACTION_TYPE_BOOLEAN_INPUT);
        std::vector<XrActionSuggestedBinding> bindings{
            {poseAction, path("/user/hand/left/input/grip/pose")}, {poseAction, path("/user/hand/right/input/grip/pose")},
            {triggerAction, path("/user/hand/left/input/trigger/value")}, {triggerAction, path("/user/hand/right/input/trigger/value")},
            {selectAction, path("/user/hand/left/input/x/click")}, {selectAction, path("/user/hand/right/input/a/click")}};
        auto suggest = [&](const char* profile) {
            XrInteractionProfileSuggestedBinding binding{XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING};
            binding.interactionProfile = path(profile); binding.countSuggestedBindings = bindings.size(); binding.suggestedBindings = bindings.data();
            XR(xrSuggestInteractionProfileBindings(instance, &binding));
        };
        suggest("/interaction_profiles/oculus/touch_controller");
        if (plus) suggest("/interaction_profiles/meta/touch_controller_plus");
        XrSessionActionSetsAttachInfo attach{XR_TYPE_SESSION_ACTION_SETS_ATTACH_INFO}; attach.countActionSets = 1; attach.actionSets = &actions;
        XR(xrAttachSessionActionSets(session, &attach));
        for (int i = 0; i < 2; ++i) {
            XrActionSpaceCreateInfo grip{XR_TYPE_ACTION_SPACE_CREATE_INFO}; grip.action = poseAction;
            grip.subactionPath = hands[i]; grip.poseInActionSpace = identity();
            XR(xrCreateActionSpace(session, &grip, &grips[i]));
        }
    }
    void makeSwapchain() {
        uint32_t count = 0;
        XR(xrEnumerateViewConfigurationViews(instance, system, XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, 0, &count, nullptr));
        if (count != 2) throw std::runtime_error("Expected two stereo views");
        std::array<XrViewConfigurationView, 2> views{{{XR_TYPE_VIEW_CONFIGURATION_VIEW}, {XR_TYPE_VIEW_CONFIGURATION_VIEW}}};
        XR(xrEnumerateViewConfigurationViews(instance, system, XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, 2, &count, views.data()));
        width = views[0].recommendedImageRectWidth; height = views[0].recommendedImageRectHeight;
        XR(xrEnumerateSwapchainFormats(session, 0, &count, nullptr));
        std::vector<int64_t> formats(count); XR(xrEnumerateSwapchainFormats(session, count, &count, formats.data()));
        if (formats.empty()) throw std::runtime_error("No swapchain formats");
        int64_t format = formats[0];
        for (auto f : formats) if (f == GL_SRGB8_ALPHA8) format = f;
        XrSwapchainCreateInfo info{XR_TYPE_SWAPCHAIN_CREATE_INFO}; info.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT;
        info.format = format; info.sampleCount = 1; info.width = width; info.height = height;
        info.faceCount = 1; info.arraySize = 2; info.mipCount = 1;
        XR(xrCreateSwapchain(session, &info, &swapchain));
        XR(xrEnumerateSwapchainImages(swapchain, 0, &count, nullptr));
        images.resize(count, {XR_TYPE_SWAPCHAIN_IMAGE_OPENGL_ES_KHR});
        XR(xrEnumerateSwapchainImages(swapchain, count, &count, reinterpret_cast<XrSwapchainImageBaseHeader*>(images.data())));
        glGenFramebuffers(1, &framebuffer);
    }
    void events() {
        XrEventDataBuffer event{XR_TYPE_EVENT_DATA_BUFFER};
        while (xrPollEvent(instance, &event) == XR_SUCCESS) {
            if (event.type == XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED) {
                state = reinterpret_cast<XrEventDataSessionStateChanged*>(&event)->state;
                if (state == XR_SESSION_STATE_READY && !running) {
                    XrSessionBeginInfo begin{XR_TYPE_SESSION_BEGIN_INFO}; begin.primaryViewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
                    XR(xrBeginSession(session, &begin)); running = true;
                } else if (state == XR_SESSION_STATE_STOPPING && running) { XR(xrEndSession(session)); running = false; }
                else if (state == XR_SESSION_STATE_EXITING || state == XR_SESSION_STATE_LOSS_PENDING) exit = true;
            } else if (event.type == XR_TYPE_EVENT_DATA_REFERENCE_SPACE_CHANGE_PENDING) {
                auto* changed = reinterpret_cast<XrEventDataReferenceSpaceChangePending*>(&event);
                if (changed->referenceSpaceType == referenceType) ++epoch;
            } else if (event.type == XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING) exit = true;
            event = {XR_TYPE_EVENT_DATA_BUFFER};
        }
    }
    void telemetry(XrTime time) {
        bool focused = state == XR_SESSION_STATE_FOCUSED;
        if (focused) {
            XrActiveActionSet active{actions, XR_NULL_PATH};
            XrActionsSyncInfo sync{XR_TYPE_ACTIONS_SYNC_INFO}; sync.countActiveActionSets = 1; sync.activeActionSets = &active;
            XR(xrSyncActions(session, &sync));
        }
        std::ostringstream out; out.setf(std::ios::fixed); out.precision(7);
        out << "{\"schema\":\"neura.quest.controllers.v1\",\"source\":\"" << source
            << "\",\"sequence\":" << sequence++ << ",\"epoch\":" << epoch
            << ",\"sample_time_ns\":" << time << ",\"reference_space\":\""
            << (referenceType == XR_REFERENCE_SPACE_TYPE_STAGE ? "stage" : "local") << "\",\"controllers\":{";
        for (int i = 0; i < 2; ++i) {
            if (i) out << ','; out << (i == 0 ? "\"left\":" : "\"right\":");
            XrActionStateGetInfo get{XR_TYPE_ACTION_STATE_GET_INFO}; get.action = poseAction; get.subactionPath = hands[i];
            XrActionStatePose pose{XR_TYPE_ACTION_STATE_POSE};
            XrSpaceLocation location{XR_TYPE_SPACE_LOCATION};
            bool tracked = false;
            if (focused) {
                XR(xrGetActionStatePose(session, &get, &pose));
                if (pose.isActive) {
                    XR(xrLocateSpace(grips[i], reference, time, &location));
                    const XrSpaceLocationFlags required = XR_SPACE_LOCATION_POSITION_VALID_BIT | XR_SPACE_LOCATION_ORIENTATION_VALID_BIT
                        | XR_SPACE_LOCATION_POSITION_TRACKED_BIT | XR_SPACE_LOCATION_ORIENTATION_TRACKED_BIT;
                    tracked = (location.locationFlags & required) == required;
                }
            }
            if (!tracked) { out << "null"; continue; }
            get.action = triggerAction; XrActionStateFloat trigger{XR_TYPE_ACTION_STATE_FLOAT}; XR(xrGetActionStateFloat(session, &get, &trigger));
            get.action = selectAction; XrActionStateBoolean button{XR_TYPE_ACTION_STATE_BOOLEAN}; XR(xrGetActionStateBoolean(session, &get, &button));
            const auto& p = location.pose.position; const auto& q = location.pose.orientation;
            out << "{\"input_type\":\"controller\",\"position\":[" << p.x << ',' << p.y << ',' << p.z
                << "],\"orientation\":[" << q.x << ',' << q.y << ',' << q.z << ',' << q.w
                << "],\"trigger\":" << (trigger.isActive ? trigger.currentState : 0)
                << ",\"select\":" << (button.isActive && button.currentState ? "true" : "false") << '}';
        }
        out << "}}"; stream.send(out.str());
    }
    void frame() {
        XrFrameWaitInfo wait{XR_TYPE_FRAME_WAIT_INFO}; XrFrameState frame{XR_TYPE_FRAME_STATE}; XR(xrWaitFrame(session, &wait, &frame));
        XrFrameBeginInfo begin{XR_TYPE_FRAME_BEGIN_INFO}; XR(xrBeginFrame(session, &begin));
        telemetry(frame.predictedDisplayTime);
        XrCompositionLayerProjection layer{XR_TYPE_COMPOSITION_LAYER_PROJECTION}; layer.space = reference;
        std::array<XrCompositionLayerProjectionView, 2> projection{{{XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW}, {XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW}}};
        const XrCompositionLayerBaseHeader* layers[] = {reinterpret_cast<XrCompositionLayerBaseHeader*>(&layer)};
        uint32_t layerCount = 0;
        if (frame.shouldRender) {
            XrViewLocateInfo locate{XR_TYPE_VIEW_LOCATE_INFO}; locate.viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
            locate.displayTime = frame.predictedDisplayTime; locate.space = reference;
            XrViewState viewsState{XR_TYPE_VIEW_STATE}; uint32_t count = 0;
            std::array<XrView, 2> views{{{XR_TYPE_VIEW}, {XR_TYPE_VIEW}}};
            XR(xrLocateViews(session, &locate, &viewsState, 2, &count, views.data()));
            if (count == 2 && (viewsState.viewStateFlags & XR_VIEW_STATE_ORIENTATION_VALID_BIT)) {
                uint32_t index; XrSwapchainImageAcquireInfo acquire{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
                XR(xrAcquireSwapchainImage(swapchain, &acquire, &index));
                XrSwapchainImageWaitInfo imageWait{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO}; imageWait.timeout = XR_INFINITE_DURATION;
                XR(xrWaitSwapchainImage(swapchain, &imageWait));
                glBindFramebuffer(GL_FRAMEBUFFER, framebuffer); glViewport(0, 0, width, height);
                for (int i = 0; i < 2; ++i) {
                    glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, images[index].image, 0, i);
                    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) throw std::runtime_error("Incomplete XR framebuffer");
                    glClearColor(.035f, .055f, .09f, 1); glClear(GL_COLOR_BUFFER_BIT);
                    projection[i].pose = views[i].pose; projection[i].fov = views[i].fov;
                    projection[i].subImage.swapchain = swapchain;
                    projection[i].subImage.imageRect.extent = {static_cast<int32_t>(width), static_cast<int32_t>(height)};
                    projection[i].subImage.imageArrayIndex = i;
                }
                glBindFramebuffer(GL_FRAMEBUFFER, 0);
                XrSwapchainImageReleaseInfo release{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO}; XR(xrReleaseSwapchainImage(swapchain, &release));
                layer.viewCount = 2; layer.views = projection.data(); layerCount = 1;
            }
        }
        XrFrameEndInfo end{XR_TYPE_FRAME_END_INFO}; end.displayTime = frame.predictedDisplayTime;
        end.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE; end.layerCount = layerCount; end.layers = layers;
        XR(xrEndFrame(session, &end));
    }
    ~App() {
        if (framebuffer) glDeleteFramebuffers(1, &framebuffer);
        if (swapchain) xrDestroySwapchain(swapchain);
        for (auto grip : grips) if (grip) xrDestroySpace(grip);
        if (reference) xrDestroySpace(reference);
        if (session) xrDestroySession(session);
        if (actions) xrDestroyActionSet(actions);
        if (instance) xrDestroyInstance(instance);
        if (display != EGL_NO_DISPLAY) {
            eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
            if (surface != EGL_NO_SURFACE) eglDestroySurface(display, surface);
            if (context != EGL_NO_CONTEXT) eglDestroyContext(display, context);
            eglTerminate(display);
        }
    }
};
}

extern "C" void android_main(android_app* android) {
    app_dummy();
    JNIEnv* environment = nullptr;
    if (android->activity->vm->AttachCurrentThread(&environment, nullptr) != JNI_OK) {
        __android_log_print(ANDROID_LOG_ERROR, "NeuraQuest", "Cannot attach native tracking thread to Java VM");
        ANativeActivity_finish(android->activity);
        return;
    }
    try {
        App app{}; app.android = android; app.init();
        while (!android->destroyRequested && !app.exit) {
            int events; android_poll_source* source = nullptr;
            while (ALooper_pollOnce(app.running ? 0 : 10, nullptr, &events, reinterpret_cast<void**>(&source)) >= 0) {
                if (source) source->process(android, source);
                if (android->destroyRequested) break;
                if (!app.running) break;
            }
            if (android->destroyRequested) break;
            app.events();
            if (app.running && !app.exit) app.frame();
        }
    } catch (const std::exception& e) {
        __android_log_print(ANDROID_LOG_ERROR, "NeuraQuest", "Tracking failed: %s", e.what());
    }
    ANativeActivity_finish(android->activity);
    android->activity->vm->DetachCurrentThread();
}
