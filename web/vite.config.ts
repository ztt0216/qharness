import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// 开发时接口转给 qharness serve（默认 8765）；构建产物由 qharness serve 直接托管
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { "/api": "http://127.0.0.1:8765" } },
});
