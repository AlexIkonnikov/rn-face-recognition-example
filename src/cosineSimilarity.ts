export function cosineSimilarity(
  vecA: Float32Array,
  vecB: Float32Array,
): number {
  'worklet';

  let dot = 0.0;
  for (let i = 0; i < vecB.length; i++) {
    dot += vecA[i]! * vecB[i]!;
  }
  return dot;
}
