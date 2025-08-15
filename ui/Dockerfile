FROM node:18-alpine AS builder

WORKDIR /app 

COPY client client/
WORKDIR /app/client 

RUN npm i && npm run build

FROM node:18-alpine

WORKDIR /app

COPY --from=builder /app/client/build client/build

COPY server server/

WORKDIR /app/server/

RUN npm install && npm run build

ENV NODE_ENV=production
ENV HOST=0.0.0.0 PORT=8080

EXPOSE 8080/tcp

LABEL name="grick8s/ai-applied/ai-applied-demo-ui" \
      vendor="IBM" \
      version="1.0" \
      author="gricas" \
      summary="A template for quickly building full stack generative AI applications" \
      description="This template provides a robust framework using React, TypeScript, Express, and FastAPI to streamline the development of full stack generative AI applications, with easy deployment on Kubernetes."

CMD ["npm", "run", "prod"]

