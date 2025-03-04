# Build stage
FROM node:20-alpine AS builder

WORKDIR /app

# Copy package files
COPY package.json ./
COPY next.config.js ./

# Install dependencies
RUN npm install

# Copy frontend code
COPY . .

# Build Next.js app
RUN npm run build

# Production stage using nginx
FROM nginx:alpine

# Copy the nginx configuration
RUN rm /etc/nginx/conf.d/default.conf
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy the build output to replace the default nginx contents.
COPY --from=builder /app/out /usr/share/nginx/html

# Expose port
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"] 