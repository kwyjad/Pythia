const buildMatrix = (rows: number, cols: number, initial: number) =>
  Array.from({ length: rows }, () => Array(cols).fill(initial));

export function jenksBreaks(values: number[], k: number): number[] {
  if (values.length === 0 || k <= 0) {
    return [];
  }

  const sorted = values.slice().sort((a, b) => a - b);
  const unique = Array.from(new Set(sorted));
  if (unique.length <= k) {
    const breaks = unique.slice();
    while (breaks.length < k + 1) {
      breaks.push(unique[unique.length - 1]);
    }
    return breaks;
  }

  const n = sorted.length;
  const lower = buildMatrix(k + 1, n + 1, 0);
  const variance = buildMatrix(k + 1, n + 1, Infinity);

  for (let i = 1; i <= k; i++) {
    lower[i][1] = 1;
    variance[i][1] = 0;
  }

  for (let l = 2; l <= n; l++) {
    let sum = 0;
    let sumSquares = 0;
    let weight = 0;
    let currentVariance = 0;

    for (let m = 1; m <= l; m++) {
      const i = l - m + 1;
      const value = sorted[i - 1];
      weight += 1;
      sum += value;
      sumSquares += value * value;
      const mean = sum / weight;
      currentVariance = sumSquares - mean * sum;
      const i4 = i - 1;

      if (i4 !== 0) {
        for (let j = 2; j <= k; j++) {
          const candidate = currentVariance + variance[j - 1][i4];
          if (candidate < variance[j][l]) {
            lower[j][l] = i;
            variance[j][l] = candidate;
          }
        }
      }
    }

    lower[1][l] = 1;
    variance[1][l] = currentVariance;
  }

  const breaks = Array(k + 1).fill(0);
  breaks[k] = sorted[n - 1];
  breaks[0] = sorted[0];

  let count = k;
  let idx = n;
  while (count > 1) {
    const id = lower[count][idx] - 1;
    breaks[count - 1] = sorted[id];
    idx = id;
    count -= 1;
  }

  return breaks;
}

export function classifyJenks(value: number, breaks: number[]): number {
  if (!breaks.length) {
    return -1;
  }
  for (let i = 1; i < breaks.length; i++) {
    if (value <= breaks[i]) {
      return i - 1;
    }
  }
  return Math.max(0, breaks.length - 2);
}
