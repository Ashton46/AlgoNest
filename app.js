// Charts + simple heatmap renderer using mock data — runs entirely client‑side.


(function(){
const byId = (id)=> document.getElementById(id);
const heatEl = byId('heatmap');
const avgEl = byId('avgPositions');
const speedEl= byId('speedChart');
const zoneEl = byId('zonesChart');
const mockBtn= byId('mockProcess');


if (mockBtn) {
mockBtn.addEventListener('click', () => runDemo());
}


function runDemo(){
const {points, averages, speeds, zones} = generateMockData();
drawHeatmap(heatEl, points);
drawAvgPositions(avgEl, averages);
drawSpeedChart(speedEl, speeds);
drawZonesChart(zoneEl, zones);
}


function generateMockData(){
// Field space: 100x100 logical units
const players = 11;
const frames = 800;
const points = []; // {x,y}


// Random walk around two clusters (attack / defense)
for (let i=0;i<players;i++){
let x = 30 + Math.random()*40;
let y = 30 + Math.random()*40;
for (let f=0; f<frames; f++){
x += (Math.random()-.5)*3;
y += (Math.random()-.5)*3;
x = Math.max(5, Math.min(95, x));
y = Math.max(5, Math.min(95, y));
points.push({x,y, w: 1});
}
}


// Averages: cluster around typical 4‑3‑3 layout
const averages = [
{x:12,y:50},{x:25,y:25},{x:25,y:75}, // back line
{x:45,y:20},{x:45,y:50},{x:45,y:80}, // midfield
{x:70,y:25},{x:70,y:50},{x:70,y:75}, // forwards
{x:15,y:15},{x:15,y:85} // keeper + spare
];


// Speeds over time (mock sinusoid + noise)
})();
