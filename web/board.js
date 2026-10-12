/** Dependency-free isometric renderer. World coordinates keep picking and drawing aligned. */
const TAU = Math.PI * 2;
const palette = { top: '#faf9f2', left: '#d2dfe0', right: '#e7ece6', dome: '#3c83c7' };

export function demoGame() {
  const heights = Array(25).fill(0);
  [[1, 1], [3, 2], [6, 2], [7, 1], [9, 1], [11, 3], [12, 1], [14, 4], [16, 1], [17, 2], [19, 1], [22, 1]].forEach(([i, h]) => heights[i] = h);
  return { heights, workers: [[8, 15], [2, 18]], current: 0, phase: 'move', selected: -1, winner: null, turn: 1, revision: 0 };
}

export class IslandBoard {
  constructor(canvas, onPick, onHover) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.onPick = onPick;
    this.onHover = onHover;
    this.angle = Math.PI / 4;
    this.game = demoGame();
    this.targets = new Set();
    this.workerCells = new Set();
    this.selected = -1;
    this.hover = -1;
    this.surfaces = [];
    this.observer = new ResizeObserver(() => this.resize());
    this.observer.observe(canvas.parentElement);
    canvas.addEventListener('pointermove', event => {
      const cell = this.pick(event);
      if (cell !== this.hover) {
        this.hover = cell;
        this.canvas.style.cursor = cell >= 0 ? 'pointer' : 'default';
        this.onHover(cell);
        this.draw();
      }
    });
    canvas.addEventListener('pointerleave', () => { this.hover = -1; this.onHover(-1); this.draw(); });
    canvas.addEventListener('click', event => {
      const cell = this.pick(event);
      if (cell >= 0) this.onPick(cell);
    });
    this.resize();
  }

  resize() {
    const rectangle = this.canvas.getBoundingClientRect();
    this.width = rectangle.width;
    this.height = rectangle.height;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    this.canvas.width = Math.round(this.width * ratio);
    this.canvas.height = Math.round(this.height * ratio);
    this.ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    this.unit = Math.min(this.width / 8.5, this.height / 6.7);
    this.draw();
  }

  setState(game, targets = [], selected = -1, workerCells = []) {
    this.game = game;
    this.targets = new Set(targets);
    this.workerCells = new Set(workerCells);
    this.selected = selected;
    this.draw();
  }

  rotate(direction) { this.angle += direction * Math.PI / 2; this.draw(); }
  resetView() { this.angle = Math.PI / 4; this.draw(); }

  project(x, y, z) {
    const cos = Math.cos(this.angle), sin = Math.sin(this.angle);
    return [this.width * .5 + (x * cos - z * sin) * this.unit,
      this.height * .60 + (x * sin + z * cos) * this.unit * .48 - y * this.unit];
  }

  polygon(points, fill, stroke = null, cell = -1) {
    const ctx = this.ctx;
    ctx.beginPath();
    points.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
    ctx.closePath();
    ctx.fillStyle = fill;
    ctx.fill();
    if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = .7; ctx.stroke(); }
    if (cell >= 0) this.surfaces.push({ cell, points });
  }

  cube(x, y, z, width, height, colors = palette, cell = -1) {
    const half = width / 2;
    const corners = [[x-half,z-half],[x+half,z-half],[x+half,z+half],[x-half,z+half]];
    const bottom = corners.map(([px,pz]) => this.project(px,y,pz));
    const top = corners.map(([px,pz]) => this.project(px,y+height,pz));
    const faces = corners.map((corner, i) => ({ i, depth: ((corner[0]+corners[(i+1)%4][0])/2)*Math.sin(this.angle)+((corner[1]+corners[(i+1)%4][1])/2)*Math.cos(this.angle) }));
    faces.sort((a,b) => a.depth-b.depth);
    for (const face of faces.slice(2)) {
      const i=face.i,j=(i+1)%4;
      const averageX = (bottom[i][0]+bottom[j][0])/2;
      const sideColor = averageX < this.project(x,y,z)[0] ? colors.left : colors.right;
      this.polygon([bottom[i],bottom[j],top[j],top[i]],sideColor,null,cell);
    }
    this.polygon(top,colors.top,colors.stroke,cell);
  }

  ellipse(x, y, rx, ry, fill, stroke = null) {
    const ctx=this.ctx;
    ctx.beginPath();ctx.ellipse(x,y,rx,ry,0,0,TAU);ctx.fillStyle=fill;ctx.fill();
    if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=1;ctx.stroke();}
  }

  ring(x,y,z,radius,color) {
    const ctx=this.ctx;const p=this.project(x,y,z);
    ctx.beginPath();ctx.ellipse(p[0],p[1],radius*this.unit,radius*this.unit*.48,0,0,TAU);
    ctx.strokeStyle=color;ctx.lineWidth=2;ctx.stroke();
  }

  dome(x,y,z,cell) {
    const p=this.project(x,y,z),r=this.unit*.32,ctx=this.ctx;
    const gradient=ctx.createRadialGradient(p[0]-r*.4,p[1]-r*.55,0,p[0],p[1]-.1*r,r*1.2);
    gradient.addColorStop(0,'#76b7e6');gradient.addColorStop(.5,'#438cc9');gradient.addColorStop(1,'#22659f');
    ctx.beginPath();ctx.moveTo(p[0]-r,p[1]);
    ctx.bezierCurveTo(p[0]-r,p[1]-r*1.35,p[0]+r,p[1]-r*1.35,p[0]+r,p[1]);
    ctx.bezierCurveTo(p[0]+r*.8,p[1]+r*.45,p[0]-r*.8,p[1]+r*.45,p[0]-r,p[1]);
    ctx.fillStyle=gradient;ctx.fill();
    this.surfaces.push({cell,points:[[p[0]-r,p[1]],[p[0]-r*.7,p[1]-r],[p[0]+r*.7,p[1]-r],[p[0]+r,p[1]+r*.3]]});
    this.ellipse(p[0]-r*.22,p[1]-r*.91,r*.06,r*.035,'#c4e3f4');
  }

  worker(x,y,z,player,cell) {
    const ctx=this.ctx,unit=this.unit;
    const colors=player===0?['#e88b70','#c26553','#f4b695']:['#f0ce79','#c29b45','#ffe3a0'];
    let p=this.project(x+.04,y+.025,z+.035);
    this.ellipse(p[0]+unit*.07,p[1]+unit*.025,unit*.24,unit*.10,'#20434c23');
    p=this.project(x,y+.03,z);this.ellipse(p[0],p[1],unit*.19,unit*.095,colors[1]);
    const bottom=this.project(x,y+.09,z),top=this.project(x,y+.33,z);
    const gradient=ctx.createLinearGradient(bottom[0]-unit*.14,0,bottom[0]+unit*.15,0);
    gradient.addColorStop(0,colors[1]);gradient.addColorStop(.4,colors[0]);gradient.addColorStop(1,colors[2]);
    this.polygon([[bottom[0]-unit*.15,bottom[1]],[top[0]-unit*.085,top[1]],[top[0]+unit*.085,top[1]],[bottom[0]+unit*.15,bottom[1]]],gradient,null,cell);
    this.ellipse(bottom[0],bottom[1],unit*.15,unit*.058,colors[0]);
    const head=this.project(x,y+.43,z);
    const skin=ctx.createRadialGradient(head[0]-unit*.035,head[1]-unit*.035,0,head[0],head[1],unit*.12);
    skin.addColorStop(0,'#ffe4b9');skin.addColorStop(1,'#d5ad7f');
    this.ellipse(head[0],head[1],unit*.105,unit*.115,skin);
    const brim=this.project(x,y+.5,z);this.ellipse(brim[0],brim[1],unit*.16,unit*.054,colors[1]);
    this.ellipse(brim[0],brim[1]-unit*.04,unit*.1,unit*.075,colors[2]);
    this.surfaces.push({cell,points:[[bottom[0]-unit*.2,bottom[1]+unit*.12],[head[0]-unit*.2,head[1]-unit*.13],[head[0]+unit*.2,head[1]-unit*.13],[bottom[0]+unit*.2,bottom[1]+unit*.12]]});
  }

  drawIsland() {
    const ctx=this.ctx,u=this.unit;
    const p=this.project(0,-.55,0);
    const water=ctx.createRadialGradient(p[0],p[1],u*1.3,p[0],p[1],u*4.3);
    water.addColorStop(0,'#88c4cc55');water.addColorStop(.65,'#a8d7db44');water.addColorStop(1,'#e8f2f500');
    this.ellipse(p[0],p[1],u*4.2,u*2.4,water);
    for(let i=0;i<3;i++) {
      ctx.beginPath();ctx.ellipse(p[0],p[1]+u*.18,u*(3.8+i*.25),u*(1.75+i*.17),0,0,TAU);
      ctx.strokeStyle=['#b5d9dd70','#b5d9dd40','#b5d9dd30'][i];ctx.lineWidth=1;ctx.stroke();
    }
    this.ellipse(p[0]+u*.1,p[1]+u*.3,u*3.25,u*1.55,'#397e8b16');
    const outline=[[-2.8,-2.6],[-1.8,-2.95],[.5,-2.9],[2.65,-2.7],[2.95,-1.3],[2.87,1.8],[2.45,2.85],[.8,3.02],[-1.65,2.9],[-2.9,2.0],[-3.02,-.5]];
    const faces=outline.map(([x,z],i)=>({i,depth:x*Math.sin(this.angle)+z*Math.cos(this.angle)})).sort((a,b)=>a.depth-b.depth);
    for(const {i} of faces){
      const [x,z]=outline[i],[nx,nz]=outline[(i+1)%outline.length];
      this.polygon([this.project(x,-.13,z),this.project(nx,-.13,nz),this.project(nx*.96,-.65,nz*.96),this.project(x*.96,-.65,z*.96)],i%2?'#b0c5ba':'#c4d0bf');
      this.polygon([this.project(x*.96,-.65,z*.96),this.project(nx*.96,-.65,nz*.96),this.project(nx*.84,-.82,nz*.84),this.project(x*.84,-.82,z*.84)],i%2?'#94b3ab':'#a1bdb0');
    }
    this.polygon(outline.map(([x,z])=>this.project(x,-.13,z)),'#e5dfc2');
    const grass=[[-2.7,1.7],[2.7,-1.7],[1.5,2.72],[-1.9,-2.72],[-2.75,-.8],[2.6,2.45]];
    for(const [x,z] of grass){let q=this.project(x,-.07,z);this.ellipse(q[0],q[1],u*.18,u*.09,'#a2b893');this.ellipse(q[0]-u*.03,q[1]-u*.025,u*.105,u*.07,'#bbca9f');}
    this.cube(0,-.1,0,5.32,.22,{top:'#b6c6b6',left:'#97afa4',right:'#a9bbae'});
    this.cube(0,.12,0,5.16,.09,{top:'#e1e8d3',left:'#d0d9c2',right:'#d6dfc9'});
  }

  draw() {
    if(!this.width||!this.height)return;
    const ctx=this.ctx;ctx.clearRect(0,0,this.width,this.height);this.surfaces=[];
    this.drawIsland();
    const cells=Array.from({length:25},(_,cell)=>({cell,x:cell%5-2,z:Math.floor(cell/5)-2}));
    cells.sort((a,b)=>(a.x*Math.sin(this.angle)+a.z*Math.cos(this.angle))-(b.x*Math.sin(this.angle)+b.z*Math.cos(this.angle)));
    for(const {cell,x,z} of cells){
      const level=this.game.heights[cell];
      let color=(Math.floor(cell/5)+cell%5)%2?'#d9e1ca':'#e2e8d3';
      if(this.targets.has(cell))color='#b9d7bc';
      if(this.hover===cell)color=this.targets.has(cell)?'#a5cbb0':'#ecedda';
      this.cube(x,.21,z,.965,.055,{top:color,left:'#c3d0b7',right:'#ccd6be',stroke:'#ced8bf'},cell);
      let elevation=.28;
      if(level>0){
        const shade=this.project(x+.06,.275,z+.12);this.ellipse(shade[0],shade[1],this.unit*.48,this.unit*.24,'#2545531c');
        for(let story=0;story<Math.min(level,3);story++){
          const size=[.83,.68,.55][story],height=[.43,.39,.34][story];
          this.cube(x,elevation,z,size,height,{...palette,top:story===2?'#f7f5e9':palette.top},cell);
          this.cube(x,elevation+height-.035,z,size+.045,.055,{top:'#fffff7',left:'#dde6e1',right:'#edf0e7'},cell);
          elevation+=height+.025;
        }
        if(level===4)this.dome(x,elevation,z,cell);
      }
      if(this.targets.has(cell)){
        this.ring(x,elevation+.045,z,level? .22:.30,'#438c77');
        const q=this.project(x,elevation+.05,z);this.ellipse(q[0],q[1],this.unit*.043,this.unit*.024,'#438c77');
      }
      for(let player=0;player<2;player++)for(let worker=0;worker<2;worker++){
        if(this.game.workers[player][worker]===cell){
          if(this.selected===cell)this.ring(x,elevation+.04,z,.32,player===0?'#b75e49':'#a78332');
          else if(this.workerCells.has(cell))this.ring(x,elevation+.035,z,.28,'#789b8d');
          this.worker(x,elevation+.03,z,player,cell);
        }
      }
    }
    ctx.textAlign='center';ctx.textBaseline='middle';ctx.font='10px "Segoe UI", sans-serif';ctx.fillStyle='#819c98';
    for(let i=0;i<5;i++){
      let p=this.project(i-2,.25,2.83);ctx.fillText(String.fromCharCode(65+i),p[0],p[1]);
      p=this.project(-2.83,.25,i-2);ctx.fillText(String(i+1),p[0],p[1]);
    }
  }

  pick(event) {
    const rect=this.canvas.getBoundingClientRect();const x=event.clientX-rect.left,y=event.clientY-rect.top;
    for(let i=this.surfaces.length-1;i>=0;i--){
      const {points,cell}=this.surfaces[i];let inside=false;
      for(let j=0,k=points.length-1;j<points.length;k=j++){
        const [xj,yj]=points[j],[xk,yk]=points[k];
        if(((yj>y)!==(yk>y))&&(x<(xk-xj)*(y-yj)/(yk-yj)+xj))inside=!inside;
      }
      if(inside)return cell;
    }
    return -1;
  }
}
