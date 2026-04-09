importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/ort.min.js");

const MODEL_INPUT_SIZE = 320;

let session = null;
let inputName = "";
let outputName = "";

// Preprocessing buffers (allocated once per worker lifetime)
const offscreen = new OffscreenCanvas(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
const octx = offscreen.getContext("2d", { willReadFrequently: true });
const float32Data = new Float32Array(3 * MODEL_INPUT_SIZE * MODEL_INPUT_SIZE);
const pixelCount = MODEL_INPUT_SIZE * MODEL_INPUT_SIZE;

self.onmessage = async (e) => {
  try {
    if (e.data.type === "init") {
      ort.env.wasm.wasmPaths =
        "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/";
      session = await ort.InferenceSession.create(e.data.modelBuffer, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
      inputName = session.inputNames[0];
      outputName = session.outputNames[0];
      self.postMessage({ type: "ready" });
    } else if (e.data.type === "infer") {
      // Draw the transferred ImageBitmap onto the offscreen canvas
      const frame = e.data.frame;
      octx.drawImage(frame, 0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
      frame.close();

      // Read pixels and convert to CHW float32
      const { data } = octx.getImageData(0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
      for (let i = 0; i < pixelCount; i++) {
        float32Data[i]                  = data[i * 4]     / 255.0;
        float32Data[pixelCount + i]     = data[i * 4 + 1] / 255.0;
        float32Data[2 * pixelCount + i] = data[i * 4 + 2] / 255.0;
      }

      // Run inference — fresh copy for tensor to avoid shared-buffer issues
      const dims = [1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE];
      const inputData = new Float32Array(float32Data);
      self.postMessage({ type: "log", message: `tensor: ${inputData.length} elems, dims=${JSON.stringify(dims)}` });
      const tensor = new ort.Tensor("float32", inputData, dims);
      const results = await session.run({ [inputName]: tensor });
      const output = results[outputName];

      // Copy output before disposing
      const outData = new Float32Array(output.data);
      const outDims = Array.from(output.dims);
      tensor.dispose();
      for (const t of Object.values(results)) t.dispose();

      self.postMessage(
        { type: "result", data: outData.buffer, dims: outDims },
        [outData.buffer]
      );
    }
  } catch (err) {
    self.postMessage({ type: "error", message: err.message });
  }
};
