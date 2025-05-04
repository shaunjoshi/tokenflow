// next.config.js
module.exports = {
  async rewrites() {
    return [
      {
        source: '/api/:path*', // Match requests starting with /api/
        destination: 'http://127.0.0.1:8000/api/:path*', // Proxy to your backend
      },
    ]
  },
}