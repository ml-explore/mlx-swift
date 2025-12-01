FROM swift:6.2.1-jammy AS base

RUN apt-get update && apt-get install -y \
    libblas-dev \
    liblapack-dev \
    liblapacke-dev \
    libopenblas-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

FROM base AS builder

COPY . .
RUN swift build -c release --product Example1 --static-swift-stdlib -Xlinker -s -v

# Final image
FROM base

# Copy executable from SwiftPM build directory
COPY --from=builder /app/.build/*/release/Example1 /app/Example1

CMD ["./Example1"]