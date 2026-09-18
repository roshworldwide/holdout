import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  base: "/",
  build: {
    outDir: "../src/holdout/dashboard_dist",
    emptyOutDir: true,
    sourcemap: false,
    assetsInlineLimit: 8192,
  },
  server: {
    proxy: {
      "/api": "http://127.0.0.1:4517",
    },
  },
});
