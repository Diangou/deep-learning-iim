let session;
let correctCount = 0;


async function loadModel() {
  session = await ort.InferenceSession.create("model.onnx");
  console.log("Modèle chargé !");
  console.log("Entrées:", session.inputNames);
  console.log("Sorties:", session.outputNames);
}
loadModel();

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
let drawing = false;
let lastX = 0;
let lastY = 0;

canvas.addEventListener("mousedown", (e) => {
  drawing = true;
  lastX = e.offsetX;
  lastY = e.offsetY;
});

canvas.addEventListener("mouseup", () => (drawing = false));
canvas.addEventListener("mouseout", () => (drawing = false));
canvas.addEventListener("mousemove", draw);

function draw(e) {
  if (!drawing) return;

  ctx.strokeStyle = "white"; // couleur du trait
  ctx.lineWidth = 24;        // taille du pinceau
  ctx.lineCap = "round";     // bords arrondis

  ctx.beginPath();
  ctx.moveTo(lastX, lastY);
  ctx.lineTo(e.offsetX, e.offsetY);
  ctx.stroke();

  lastX = e.offsetX;
  lastY = e.offsetY;
}

document.getElementById("clear").onclick = () => {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  document.getElementById("result").innerText = "---";
};

document.getElementById("predictBtn").onclick = () => {
  predict();
};

document.getElementById("correctBtn").onclick = () => {
  correctCount++;
  document.getElementById("correctCount").innerText = correctCount;
};

async function predict() {
  if (!session) return;

  const small = document.createElement("canvas");
  small.width = 28;
  small.height = 28;
  const smallCtx = small.getContext("2d");
  smallCtx.drawImage(canvas, 0, 0, 28, 28);

  const imageData = smallCtx.getImageData(0, 0, 28, 28).data;
  const input = new Float32Array(28 * 28);

  const mean = 0.1307;
  const std = 0.3081;

  for (let i = 0; i < 28 * 28; i++) {
    const r = imageData[i * 4]; 
    const pixel = r / 255; 
    input[i] = (pixel - mean) / std; 
  }

  const tensor = new ort.Tensor("float32", input, [1, 1, 28, 28]);

  const feeds = { [session.inputNames[0]]: tensor };
  const results = await session.run(feeds);
  const output = results[session.outputNames[0]].data;

  const probs = softmax(Array.from(output));
  console.log("Probabilités:", probs);

  const maxIndex = probs.indexOf(Math.max(...probs));
  document.getElementById("result").innerText = maxIndex;
}

function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map(v => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(v => v / sum);
}