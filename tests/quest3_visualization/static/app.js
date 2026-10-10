import { relativePose } from './math.js';
import { Renderer } from './renderer.js';

const $ = id => document.getElementById(id);
const sourceMode = new URLSearchParams(location.search).has('source');
let renderer;
try { renderer = new Renderer($('scene')); }
catch (error) { $('status').textContent=error.message; throw error; }
let session=null, referenceSpace=null, floorY=0, relayConnected=false, epoch=null;
let controllers={left:null,right:null}, receivedAt=0, origin=null, lastSend=0;
let transportStatus='Connecting…', originPending=true, starting=false;
const sourceId=Array.from(crypto.getRandomValues(new Uint8Array(16)),v=>v.toString(16).padStart(2,'0')).join('');
let pending=null, posting=false;

function clearTracking() {
  controllers={left:null,right:null}; receivedAt=0;
}

function connect() {
  const events = new EventSource('/api/events');
  events.addEventListener('open', () => {
    relayConnected=true;
    transportStatus=sourceMode?'Relay connected. Start Quest tracking.':'Waiting for Quest controller…';
    origin=null; originPending=true;
  });
  events.addEventListener('message', event => {
    const message=JSON.parse(event.data);
    if (message.type==='error') { transportStatus=message.message; return; }
    if (message.type==='pose' && !sourceMode) {
      controllers=message.controllers; receivedAt=performance.now();
      if (message.epoch!==undefined && message.epoch!==epoch) {
        origin=null; originPending=true; epoch=message.epoch;
      }
    }
  });
  events.addEventListener('error', () => {
    relayConnected=false;
    transportStatus='Relay disconnected. Reconnecting…'; clearTracking();
  });
}

async function send(message) {
  // One in-flight request, one latest sample; never queue increasingly old poses.
  if (pending?.reset) message.reset=true;
  pending=message;
  if (posting) return;
  posting=true;
  try {
    while (pending) {
      const sample=pending; pending=null;
      const response=await fetch('/api/pose',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({...sample,source:sourceId}),signal:AbortSignal.timeout(1000)});
      if (!response.ok) throw new Error(await response.text());
    }
  } catch (error) {
    pending=null;
    if (session) await session.end();
    pending=null;
    transportStatus=`Pose relay failed: ${error.message}`;
  } finally { posting=false; }
}

function recenter() {
  origin=null; originPending=true;
}
$('recenter').addEventListener('click',recenter);
$('hand').addEventListener('change',recenter);
$('demo').addEventListener('change',recenter);

function simulatedPose(time) {
  const t=time/1000;
  return {position:[0.35*Math.sin(t),0.18*Math.sin(t*0.7),0.35*Math.cos(t)],
    orientation:[Math.sin(t/2)*0.35,Math.sin(t/2)*Math.sqrt(1-0.35**2),0,Math.cos(t/2)]};
}

function desktopFrame(time) {
  requestAnimationFrame(desktopFrame);
  if (session || starting) return;
  const demo=$('demo').checked;
  const pose=demo?simulatedPose(time):(time-receivedAt<1000?controllers[$('hand').value]:null);
  let displayed=null;
  if (pose) {
    if (originPending) { origin=structuredClone(pose); originPending=false; }
    displayed=relativePose(pose,origin);
    $('status').textContent=demo?'SIMULATION — synthetic movement, no controller data':
      pose.emulatedPosition?'Rotation tracked; position is estimated by the headset.':`${$('hand').value} controller tracking live`;
    const p=displayed.position.map(v=>v.toFixed(3)).join(', ');
    const q=displayed.orientation.map(v=>v.toFixed(3)).join(', ');
    $('detail').textContent=`Relative position (m): [${p}] · Quaternion (x, y, z, w): [${q}]`;
  } else {
    $('status').textContent=relayConnected?
      (sourceMode?transportStatus:'No tracked controller. Start tracking in the Quest browser; wake the selected controller.'):transportStatus;
    $('detail').textContent='Position in meters · X right · Y up · −Z forward';
  }
  renderer.desktop(displayed);
}

function resetReference() {
  recenter(); clearTracking();
  send({type:'pose',controllers,reset:true});
}

function xrFrame(time, frame) {
  if (!session || session!==frame.session) return;
  session.requestAnimationFrame(xrFrame);
  controllers={left:null,right:null};
  if (session.visibilityState!=='visible') {
    send({type:'pose',controllers}); return;
  }
  // A fixed local reference space preserves world translation when the head moves.
  for (const input of session.inputSources) {
    if (!input.gripSpace || input.hand || !['left','right'].includes(input.handedness)) continue;
    const pose=frame.getPose(input.gripSpace,referenceSpace);
    if (!pose) continue;
    const {position:p,orientation:q}=pose.transform;
    controllers[input.handedness]={position:[p.x,p.y,p.z],orientation:[q.x,q.y,q.z,q.w],
      emulatedPosition:pose.emulatedPosition};
  }
  receivedAt=performance.now();
  if (time-lastSend>=1000/30) { send({type:'pose',controllers}); lastSend=time; }
  renderer.xr(frame,referenceSpace,Object.values(controllers),floorY);
}

async function startTracking() {
  if (session) { await session.end(); return; }
  if (!relayConnected) {
    transportStatus='Wait for the relay connection before starting.'; return;
  }
  starting=true; $('start').disabled=true;
  try {
    // requestSession is called directly from the button's user gesture.
    session=await navigator.xr.requestSession('immersive-vr',{optionalFeatures:['local-floor']});
    session.addEventListener('end', () => {
      session=null; starting=false; clearTracking();
      send({type:'pose',controllers});
      $('start').textContent='Start Quest tracking'; transportStatus='Tracking stopped. Start again to resume.';
    });
    await renderer.gl.makeXRCompatible();
    session.updateRenderState({baseLayer:new XRWebGLLayer(session,renderer.gl)});
    try { referenceSpace=await session.requestReferenceSpace('local-floor'); floorY=0; }
    catch { referenceSpace=await session.requestReferenceSpace('local'); floorY=-1.6; }
    referenceSpace.addEventListener('reset',resetReference);
    session.addEventListener('visibilitychange', () => {
      if (session && session.visibilityState!=='visible') {
        clearTracking(); send({type:'pose',controllers});
      }
    });
    lastSend=0; resetReference();
    $('start').textContent='Stop Quest tracking';
    session.requestAnimationFrame(xrFrame);
  } catch (error) {
    if (session) await session.end();
    transportStatus=`Cannot start WebXR: ${error.message}. Check Quest browser permissions and secure context.`;
  } finally { starting=false; $('start').disabled=false; }
}

if (sourceMode) {
  $('start').hidden=false; $('demo-control').hidden=true; $('recenter').hidden=true;
  $('intro').textContent='Start tracking here on the Quest. Open the desktop viewer on your computer.';
  $('start').addEventListener('click',startTracking);
  if (!window.isSecureContext || !navigator.xr) {
    $('start').disabled=true;
    $('intro').textContent='WebXR needs the Quest browser and a secure origin. Use USB adb forwarding with http://localhost:8000, or trusted HTTPS.';
  }
}
connect(); requestAnimationFrame(desktopFrame);
