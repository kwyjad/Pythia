/** @type {import('next').NextConfig} */

// The browser talks to two origins: this one, and the API the pages fetch
// from on the client side. The CSP names both and nothing else, so a
// script injected through model-written text has nowhere to send what it
// reads.
const apiBase = process.env.NEXT_PUBLIC_PYTHIA_API_BASE ?? "http://localhost:8000/v1";
let apiOrigin = "";
try {
  apiOrigin = new URL(apiBase).origin;
} catch {
  apiOrigin = "";
}

const isDev = process.env.NODE_ENV !== "production";

// Next.js 14 writes inline bootstrap scripts and inline styles without a
// nonce, so 'unsafe-inline' stays for those two; everything else is closed.
const csp = [
  "default-src 'self'",
  `script-src 'self' 'unsafe-inline'${isDev ? " 'unsafe-eval'" : ""}`,
  "style-src 'self' 'unsafe-inline'",
  "img-src 'self' data: blob:",
  "font-src 'self' data:",
  `connect-src 'self'${apiOrigin ? ` ${apiOrigin}` : ""}${isDev ? " ws:" : ""}`,
  "object-src 'none'",
  "base-uri 'self'",
  "form-action 'self'",
  "frame-ancestors 'none'",
].join("; ");

const securityHeaders = [
  { key: "Content-Security-Policy", value: csp },
  { key: "X-Content-Type-Options", value: "nosniff" },
  { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
  { key: "X-Frame-Options", value: "DENY" },
  { key: "Strict-Transport-Security", value: "max-age=31536000; includeSubDomains" },
  { key: "Permissions-Policy", value: "camera=(), microphone=(), geolocation=()" },
];

const nextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
  // The dashboard uses no next/image. With the optimiser off, /_next/image
  // answers 404, which closes the route every image-optimiser advisory
  // against Next 14 goes through (14.2.35 is the last 14.x release).
  images: { unoptimized: true },
  async headers() {
    return [{ source: "/:path*", headers: securityHeaders }];
  },
};

export default nextConfig;
