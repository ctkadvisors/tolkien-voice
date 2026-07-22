import { defineConfig } from "vite";
import topLevelAwait from "vite-plugin-top-level-await";


export default defineConfig({
  server: {
    host: "127.0.0.1",
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8788",
        ws: true,
      },
    },
  },
  plugins: [
    topLevelAwait({
      promiseExportName: "__tla",
      promiseImportName: (index) => `__tla_${index}`,
    }),
  ],
});
