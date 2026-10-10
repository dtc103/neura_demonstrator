import { multiply, poseMatrix, perspective, lookAt } from './math.js';

export class Renderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.gl = canvas.getContext('webgl', { antialias: true, xrCompatible: true });
    if (!this.gl) throw new Error('WebGL is unavailable. Enable hardware acceleration in your browser.');
    const gl = this.gl;
    const shader = (type, source) => {
      const s = gl.createShader(type);
      gl.shaderSource(s, source); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      return s;
    };
    this.program = gl.createProgram();
    gl.attachShader(this.program, shader(gl.VERTEX_SHADER,
      'attribute vec3 p; attribute vec3 color; uniform mat4 mvp; varying vec3 c; void main(){c=color;gl_Position=mvp*vec4(p,1.0);}'));
    gl.attachShader(this.program, shader(gl.FRAGMENT_SHADER,
      'precision mediump float; varying vec3 c; void main(){gl_FragColor=vec4(c,1.0);}'));
    gl.linkProgram(this.program);
    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(this.program));
    this.p = gl.getAttribLocation(this.program, 'p');
    this.color = gl.getAttribLocation(this.program, 'color');
    this.mvp = gl.getUniformLocation(this.program, 'mvp');
    const vertex = (p, c) => [...p, ...c];
    const corners = [[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]];
    const faces = [[0,1,2,3],[5,4,7,6],[4,0,3,7],[1,5,6,2],[3,2,6,7],[4,5,1,0]];
    const colors = [[0.08,0.3,0.48],[0.15,0.6,0.75],[0.08,0.4,0.52],[0.1,0.5,0.65],[0.4,0.8,0.9],[0.05,0.2,0.32]];
    this.cube = this.mesh(faces.flatMap((f,i) => [0,1,2,0,2,3].flatMap(j => vertex(corners[f[j]], colors[i]))), gl.TRIANGLES);
    const axes = [[1,0,0],[0,1,0],[0,0,1]];
    this.axes = this.mesh(axes.flatMap(c => [...vertex([0,0,0],c), ...vertex(c.map(v=>v*3),c)]), gl.LINES);
    const grid = [];
    for (let i=-12; i<=12; i++) {
      const c = i === 0 ? [0.3,0.4,0.5] : [0.16,0.22,0.29];
      grid.push(...vertex([i/4,0,-3],c), ...vertex([i/4,0,3],c),
        ...vertex([-3,0,i/4],c), ...vertex([3,0,i/4],c));
    }
    this.grid = this.mesh(grid, gl.LINES);
    this.yaw = 0.55; this.pitch = 0.35; this.distance = 2.8;
    let drag = null;
    canvas.addEventListener('pointerdown', e => { drag=[e.clientX,e.clientY]; canvas.setPointerCapture(e.pointerId); });
    canvas.addEventListener('pointermove', e => {
      if (!drag) return;
      this.yaw -= (e.clientX-drag[0])*0.008;
      this.pitch = Math.max(-1.3,Math.min(1.3,this.pitch+(e.clientY-drag[1])*0.008));
      drag=[e.clientX,e.clientY];
    });
    for (const name of ['pointerup','pointercancel']) canvas.addEventListener(name, () => { drag=null; });
    canvas.addEventListener('wheel', e => {
      e.preventDefault(); this.distance = Math.max(0.3,Math.min(15,this.distance*Math.exp(e.deltaY*0.001)));
    }, { passive: false });
  }

  mesh(vertices, mode) {
    const gl=this.gl, buffer=gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER,buffer);
    gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(vertices),gl.STATIC_DRAW);
    return {buffer, mode, count:vertices.length/6};
  }

  draw(mesh, matrix) {
    const gl=this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER,mesh.buffer);
    gl.enableVertexAttribArray(this.p); gl.enableVertexAttribArray(this.color);
    gl.vertexAttribPointer(this.p,3,gl.FLOAT,false,24,0);
    gl.vertexAttribPointer(this.color,3,gl.FLOAT,false,24,12);
    gl.uniformMatrix4fv(this.mvp,false,new Float32Array(matrix));
    gl.drawArrays(mesh.mode,0,mesh.count);
  }

  scene(projection, view, poses, floorY) {
    const gl=this.gl;
    gl.useProgram(this.program); gl.enable(gl.DEPTH_TEST);
    const vp=multiply(projection,view);
    this.draw(this.grid,multiply(vp,poseMatrix([0,floorY,0],[0,0,0,1])));
    for (const pose of poses.filter(Boolean)) {
      const mvp=multiply(vp,poseMatrix(pose.position,pose.orientation,0.06));
      this.draw(this.cube,mvp); this.draw(this.axes,mvp);
    }
  }

  clear() {
    const gl=this.gl;
    gl.clearColor(0.04,0.07,0.1,1); gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
  }

  desktop(pose) {
    const gl=this.gl, canvas=this.canvas;
    const ratio=Math.min(devicePixelRatio || 1,2);
    const width=Math.round(canvas.clientWidth*ratio), height=Math.max(1,Math.round(canvas.clientHeight*ratio));
    if (canvas.width!==width || canvas.height!==height) { canvas.width=width; canvas.height=height; }
    gl.bindFramebuffer(gl.FRAMEBUFFER,null); gl.viewport(0,0,width,height); this.clear();
    const d=this.distance, p=this.pitch, y=this.yaw;
    const eye=[d*Math.cos(p)*Math.sin(y), d*Math.sin(p), d*Math.cos(p)*Math.cos(y)];
    this.scene(perspective(width/height),lookAt(eye,[0,0,0]),[pose],-0.5);
  }

  xr(frame, space, poses, floorY) {
    const viewer=frame.getViewerPose(space);
    if (!viewer) return;
    const gl=this.gl, layer=frame.session.renderState.baseLayer;
    gl.bindFramebuffer(gl.FRAMEBUFFER,layer.framebuffer); this.clear();
    for (const view of viewer.views) {
      const viewport=layer.getViewport(view);
      gl.viewport(viewport.x,viewport.y,viewport.width,viewport.height);
      this.scene(view.projectionMatrix,view.transform.inverse.matrix,poses,floorY);
    }
  }
}
