import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

// Shared deployment settings: ../aegis.config.json (same file the backend uses).
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const CONFIG_PATH = path.resolve(__dirname, "..", "aegis.config.json");

function loadAegisConfig() {
  const defaults = {
    services: {
      api_host: "127.0.0.1",
      api_port: 8000,
      ui_host: "0.0.0.0",
      ui_port: 5173,
    },
  };
  try {
    const raw = JSON.parse(fs.readFileSync(CONFIG_PATH, "utf8"));
    return {
      ...defaults,
      ...raw,
      services: { ...defaults.services, ...(raw.services || {}) },
    };
  } catch {
    return defaults;
  }
}

const aegis = loadAegisConfig();
const apiHost = process.env.VITE_API_PROXY_HOST || aegis.services.api_host;
const apiPort = Number(
  process.env.VITE_API_PROXY_PORT || aegis.services.api_port || 8000,
);
const API_PROXY_TARGET =
  process.env.VITE_API_PROXY_TARGET || `http://${apiHost}:${apiPort}`;
const uiHost = process.env.VITE_UI_HOST || aegis.services.ui_host || "0.0.0.0";
const uiPort = Number(
  process.env.VITE_UI_PORT || aegis.services.ui_port || 5173,
);

export default defineConfig({
  plugins: [react()],
  server: {
    host: uiHost,
    port: uiPort,
    // Long-lived clinical RAG + guard calls
    proxy: {
      "/api": {
        target: API_PROXY_TARGET,
        changeOrigin: true,
        timeout: 300_000,
        proxyTimeout: 300_000,
      },
      "/health": {
        target: API_PROXY_TARGET,
        changeOrigin: true,
      },
    },
  },
  preview: {
    host: uiHost,
    port: uiPort,
    proxy: {
      "/api": {
        target: API_PROXY_TARGET,
        changeOrigin: true,
        timeout: 300_000,
        proxyTimeout: 300_000,
      },
    },
  },
});
