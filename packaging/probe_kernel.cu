// Compilation-only smoke test: no GPU execution or model weights.
extern "C" __global__ void hq_probe(float *out) {
    out[threadIdx.x] = static_cast<float>(threadIdx.x);
}
