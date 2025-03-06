# Simple development Dockerfile for Next.js
FROM node:18-alpine

WORKDIR /app

# Copy package.json and package-lock.json
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy frontend code
COPY . ./

# Set environment variables
ENV NODE_ENV=development
ENV NEXT_PUBLIC_API_URL=http://localhost:5000

# Expose port
EXPOSE 3000

# Start the application in development mode
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]