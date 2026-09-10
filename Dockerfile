FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
  git \
  curl \
  btrfs-progs \
  containernetworking-plugins \
  gcc \
  git \
  go-md2man \
  iptables \
  libassuan-dev \
  libbtrfs-dev \
  libc6-dev \
  libdevmapper-dev \
  libglib2.0-dev \
  libgpgme-dev \
  libgpg-error-dev \
  libprotobuf-dev \
  libprotobuf-c-dev \
  libseccomp-dev \
  libselinux1-dev \
  libsystemd-dev \
  make \
  pkg-config \
  runc \
  uidmap \
&& apt-get clean && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
  podman \
&& apt-get clean && rm -rf /var/lib/apt/lists/*

RUN curl -L -O https://golang.org/dl/go1.23.1.linux-amd64.tar.gz
RUN tar -C /usr/local -xzf go1.23.1.linux-amd64.tar.gz
ENV PATH="/usr/local/go/bin:${PATH}"
RUN go version
