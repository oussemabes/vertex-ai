FROM node:alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY ./ ./

# Create public directory inside dist
RUN mkdir -p dist/public

# Create images directory inside public
RUN mkdir -p dist/public/images

EXPOSE 3001
RUN npm run build
CMD ["node", "dist/index.js"]

