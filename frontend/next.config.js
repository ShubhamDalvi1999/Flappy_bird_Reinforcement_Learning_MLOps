/** @type {import('next').NextConfig} */
const nextConfig = {
  // Only use static export for production builds
  ...(process.env.NODE_ENV === 'production' ? {
    output: 'export',
    distDir: 'out',
    trailingSlash: true,
    images: {
      unoptimized: true,
    },
  } : {
    // Development config
    async rewrites() {
      return [
        {
          source: '/api/:path*',
          destination: 'http://backend:5000/api/:path*',
        },
        {
          source: '/socket.io/:path*',
          destination: 'http://backend:5000/socket.io/:path*',
        },
      ]
    },
  })
}

module.exports = nextConfig 