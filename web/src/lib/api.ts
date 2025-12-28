type QueryParams = Record<string, string | number | boolean | null | undefined>;

const API_BASE =
  process.env.NEXT_PUBLIC_PYTHIA_API_BASE ?? "http://localhost:8000/v1";

const buildUrl = (path: string, params?: QueryParams) => {
  const endpoint = path.startsWith("http") ? path : `${API_BASE}${path}`;
  const url = new URL(endpoint);
  if (params) {
    Object.entries(params).forEach(([key, value]) => {
      if (value === null || value === undefined) {
        return;
      }
      url.searchParams.set(key, String(value));
    });
  }
  return url.toString();
};

export const apiGet = async <T>(path: string, params?: QueryParams): Promise<T> => {
  const url = buildUrl(path, params);
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`API request failed (${response.status})`);
  }
  return (await response.json()) as T;
};
