importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/ort.min.js");

let session = null;
let inputName = "";
let outputName = "";

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
      const tensor = new ort.Tensor(
        "float32",
        new Float32Array(e.data.inputBuffer),
        e.data.dims
      );
      const results = await session.run({ [inputName]: tensor });
      const output = results[outputName];
      const data = new Float32Array(output.data);
      const dims = Array.from(output.dims);
      tensor.dispose();
      for (const t of Object.values(results)) t.dispose();
      self.postMessage({ type: "result", data: data.buffer, dims }, [
        data.buffer,
      ]);
    }
  } catch (err) {
    self.postMessage({ type: "error", message: err.message });
  }
};
