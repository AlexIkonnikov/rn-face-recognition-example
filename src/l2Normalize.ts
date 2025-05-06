export function l2Normalize(embedding: Float32Array) {
  'worklet';
  const norm = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
  return embedding.map(val => val / norm);
}
