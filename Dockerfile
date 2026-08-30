FROM golang:1.22-alpine AS builder
WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY main.go .
RUN CGO_ENABLED=0 GOOS=linux go build -o server main.go


FROM alpine:3.19
WORKDIR /app


RUN apk add --no-cache ca-certificates

COPY --from=builder /app/server .
COPY data/train.csv ./data/train.csv

ENV TRAIN_CSV=/app/data/train.csv
EXPOSE 8080

CMD ["./server"]
