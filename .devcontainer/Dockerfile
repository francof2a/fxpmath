FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    bash-completion \
    build-essential \
    ca-certificates \
    curl \
    git \
    openssh-client \
    python3 \
    python3-pip \
    python3-venv \
    sudo \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

RUN if ! id -u ubuntu >/dev/null 2>&1; then useradd -m -s /bin/bash ubuntu; fi \
    && usermod -aG sudo ubuntu

# Python dev dependencies in a venv owned by the remote user
COPY requirements.txt requirements-dev.txt /tmp/req/
USER ubuntu
RUN python3 -m venv /home/ubuntu/.venv \
    && /home/ubuntu/.venv/bin/pip install --no-cache-dir -r /tmp/req/requirements-dev.txt
USER root

COPY .devcontainer/init-user.sh /usr/local/share/devcontainer/init-user.sh
RUN chmod 755 /usr/local/share/devcontainer/init-user.sh

WORKDIR /workspace

ENTRYPOINT ["/usr/local/share/devcontainer/init-user.sh"]
CMD ["sleep", "infinity"]
