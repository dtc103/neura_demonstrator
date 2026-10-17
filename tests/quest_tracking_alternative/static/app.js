const status = document.getElementById('status');
const start = document.getElementById('start');
const stop = document.getElementById('stop');
const diagnostics = document.getElementById('diagnostics');
diagnostics.textContent = `Client: 20261014-recording\nURL: ${location.href}\nSecure context: ${window.isSecureContext}\nWebXR API: ${!!navigator.xr}\nBrowser: ${navigator.userAgent}`;
let session = null, space = null, gl = null, source = '', epoch = 0, sequence = 0;
let lastSend = 0, sending = false;

let clockSync = null;
async function syncClock() {
  const before = performance.now();
  try {
    const response = await fetch('/clock', {cache:'no-store', signal:AbortSignal.timeout(1000)});
    const sample = await response.json();
    const after = performance.now();
    const candidate = {host_offset_ns:sample.host_monotonic_ns - (before+after)*5e5,
      round_trip_ms:after-before, at:after};
    if (!clockSync || after-clockSync.at > 10000 || candidate.round_trip_ms < clockSync.round_trip_ms)
      clockSync=candidate;
  } catch (_) { /* Missing synchronization is explicit in recorded timing. */ }
}
setInterval(syncClock, 2000);
syncClock();

function payload(frame, frameTime) {
  const controllers = {left:null, right:null};
  const controller_buttons = {left:null, right:null};
  if (session.visibilityState === 'visible') {
    for (const input of session.inputSources) {
      // Do not let a hand-tracking source replace a physical controller.
      if (input.hand || !input.gamepad || !input.gripSpace ||
          input.gamepad.mapping !== 'xr-standard' || !(input.handedness in controllers)) continue;
      // Buttons remain usable when the physical controller has no tracked pose.
      controller_buttons[input.handedness] = {
        input_type:'controller',
        select:input.gamepad.buttons[4]?.pressed ?? false,
        secondary:input.gamepad.buttons[5]?.pressed ?? false
      };
      const pose = frame.getPose(input.gripSpace, space);
      if (!pose || pose.emulatedPosition) continue;
      const p = pose.transform.position, q = pose.transform.orientation;
      controllers[input.handedness] = {
        axes:Array.from(input.gamepad.axes),
        buttons:Array.from(input.gamepad.buttons, b=>({value:b.value, pressed:b.pressed, touched:b.touched})),
        input_type:'controller', position:[p.x,p.y,p.z], orientation:[q.x,q.y,q.z,q.w],
        trigger:input.gamepad.buttons[0]?.value ?? 0,
        squeeze:input.gamepad.buttons[1]?.value ?? 0,
        thumbstick:input.gamepad.buttons[3]?.pressed ?? false,
        select:input.gamepad.buttons[4]?.pressed ?? false,
        secondary:input.gamepad.buttons[5]?.pressed ?? false,
        emulatedPosition:!!pose.emulatedPosition
      };
    }
  }
  return {reference_space:'local-floor', source, epoch, sequence:sequence++, controllers, controller_buttons,
    timing:{frame_time_ms:frameTime, sample_time_ms:performance.now(), time_origin_ms:performance.timeOrigin,
      host_offset_ns:clockSync?.host_offset_ns ?? null,
      round_trip_ms:clockSync?.round_trip_ms ?? null,
      sync_age_ms:clockSync ? performance.now()-clockSync.at : null}};
}

async function send(data) {
  if (sending) return;
  sending = true;
  try {
    const response = await fetch('/pose', {method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify(data), signal:AbortSignal.timeout(1000)});
    if (!response.ok) throw new Error(await response.text());
    const hands = Object.entries(data.controllers).filter(([,pose])=>pose).map(([hand])=>hand);
    status.textContent = `Streaming local-floor controller poses\nTracked: ${hands.join(', ') || 'none'}\nReference epoch: ${epoch}\nRight B: ${data.controllers.right?.secondary ?? "not tracked"}`;
  } catch (error) { status.textContent = `Relay error: ${error.message}`; }
  finally { sending = false; }
}

function onFrame(time, frame) {
  if (!session) return;
  session.requestAnimationFrame(onFrame);
  gl.bindFramebuffer(gl.FRAMEBUFFER, session.renderState.baseLayer.framebuffer);
  gl.clearColor(.05,.08,.15,1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  if (time-lastSend >= 1000/30 && !sending) {
    lastSend=time;
    send(payload(frame, time));
  }
}

start.onclick = async () => {
  start.disabled = true;
  try {
    const canvas = document.createElement('canvas');
    gl = canvas.getContext('webgl', {xrCompatible:true});
    if (!gl) throw new Error('WebGL unavailable');
    session = await navigator.xr.requestSession('immersive-vr', {
      requiredFeatures:['local-floor'], optionalFeatures:['dom-overlay'], domOverlay:{root:document.body}
    });
    session.updateRenderState({baseLayer:new XRWebGLLayer(session,gl)});
    // Never fall back to viewer/head space.
    space = await session.requestReferenceSpace('local-floor');
    source = crypto.randomUUID(); epoch=0; sequence=0; lastSend=0;
    space.addEventListener('reset', () => {epoch++;});
    session.addEventListener('end', () => {
      const endedSource=source;
      session=null; space=null; start.disabled=false; stop.disabled=true;
      send({reference_space:'local-floor',source:endedSource,epoch,sequence:sequence++,controllers:{left:null,right:null}});
    });
    stop.disabled=false;
    session.requestAnimationFrame(onFrame);
  } catch (error) {
    status.textContent=`Cannot start: ${error.message}`;
    if (session) await session.end();
    session=null; start.disabled=false;
  }
};
stop.onclick = () => session?.end();
if (!window.isSecureContext || !navigator.xr) {
  start.disabled=true;
  status.textContent='WebXR needs the Quest browser and a secure context. Over USB use http://localhost:8443 after adb reverse.';
} else {
  navigator.xr.isSessionSupported('immersive-vr').then(supported => {
    start.disabled=!supported;
    diagnostics.textContent += `\nImmersive VR supported: ${supported}`;
    status.textContent=supported ? 'Ready. Start tracking, allow VR access, and keep the controllers awake.' :
      'This browser cannot start immersive VR. Open http://localhost:8443/ in Meta Quest Browser INSIDE the headset, not the PC or phone browser. If already there, close other VR sessions, restart Quest Browser, and reload. Browser diagnostics below show the detected environment.';
  }).catch(error => {
    start.disabled=true;
    status.textContent=`WebXR support check failed: ${error.name}: ${error.message}. Check the browser permissions and reload in Meta Quest Browser.`;
    diagnostics.textContent += `\nSupport check error: ${error.name}: ${error.message}`;
  });
}
