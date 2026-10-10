// Column-major matrices and WebXR quaternions [x, y, z, w].
export const identity = () => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

export function multiply(a, b) {
  const out = new Array(16).fill(0);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    for (let k = 0; k < 4; k++) out[c * 4 + r] += a[k * 4 + r] * b[c * 4 + k];
  }
  return out;
}

export function quaternionMultiply([x, y, z, w], [a, b, c, d]) {
  return [w*a + x*d + y*c - z*b, w*b - x*c + y*d + z*a,
    w*c + x*b - y*a + z*d, w*d - x*a - y*b - z*c];
}

export function rotateVector(q, vector) {
  return quaternionMultiply(quaternionMultiply(q, [...vector, 0]), [-q[0], -q[1], -q[2], q[3]]).slice(0, 3);
}

export function relativePose(pose, origin) {
  const inverse = [-origin.orientation[0], -origin.orientation[1], -origin.orientation[2], origin.orientation[3]];
  return {
    position: pose.position.map((v, i) => v - origin.position[i]),
    orientation: quaternionMultiply(pose.orientation, inverse),
  };
}

export function poseMatrix(position, [x, y, z, w], scale = 1) {
  return [
    (1-2*y*y-2*z*z)*scale, (2*x*y+2*z*w)*scale, (2*x*z-2*y*w)*scale, 0,
    (2*x*y-2*z*w)*scale, (1-2*x*x-2*z*z)*scale, (2*y*z+2*x*w)*scale, 0,
    (2*x*z+2*y*w)*scale, (2*y*z-2*x*w)*scale, (1-2*x*x-2*y*y)*scale, 0,
    ...position, 1,
  ];
}

export function perspective(aspect) {
  const f = 1 / Math.tan(Math.PI / 8), near = 0.01, far = 100;
  return [f/aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far+near)/(near-far), -1,
    0, 0, 2*far*near/(near-far), 0];
}

export function lookAt(eye, target) {
  const unit = v => { const n = Math.hypot(...v); return v.map(x => x/n); };
  const cross = (a,b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
  const dot = (a,b) => a.reduce((s, x, i) => s + x*b[i], 0);
  const z = unit(eye.map((x,i) => x-target[i]));
  const x = unit(cross([0,1,0], z)), y = cross(z,x);
  return [x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0,
    -dot(x,eye), -dot(y,eye), -dot(z,eye), 1];
}
