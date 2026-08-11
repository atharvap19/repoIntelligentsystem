import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'node:path'

// The backend has no CORS middleware, so dev requests are proxied through
// Vite instead of hitting http://localhost:8000 directly from the browser.
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { '@': path.resolve(__dirname, './src') },
  },
  build: {
    rollupOptions: {
      output: {
        // The markdown + syntax-highlighting stack is most of the bundle and
        // changes far less often than app code.
        manualChunks: {
          markdown: ['react-markdown', 'remark-gfm', 'rehype-highlight'],
          react: ['react', 'react-dom'],
        },
      },
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: process.env.VITE_BACKEND_URL ?? 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ''),
      },
    },
  },
})
