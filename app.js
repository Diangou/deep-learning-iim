let session;

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

canvas.addEventListener("mousedown", () => (drawing = true));
canvas.addEventListener("mouseup", () => (drawing = false));
canvas.addEventListener("mousemove", draw);

function draw(e) {
  if (!drawing) return;
  ctx.fillStyle = "white";
  ctx.beginPath();
  ctx.arc(e.offsetX, e.offsetY, 12, 0, Math.PI * 2);
  ctx.fill();
}

document.getElementById("clear").onclick = () => {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  document.getElementById("result").innerText = "---";
};

// 🔗 Bouton prédire
document.getElementById("predictBtn").onclick = () => {
  predict();
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

  for (let i = 0; i < 28 * 28; i++) {
    const r = imageData[i * 4];
    input[i] = r / 255; // ⚠️ plus d’inversion
  }

  const tensor = new ort.Tensor("float32", input, [1, 1, 28, 28]);

  const feeds = { [session.inputNames[0]]: tensor };
  const results = await session.run(feeds);
  const output = results[session.outputNames[0]].data;

  // Softmax pour voir les probabilités
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