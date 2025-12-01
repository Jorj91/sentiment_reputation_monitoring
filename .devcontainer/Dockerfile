# dev container - minimal for development (CPU)
FROM mcr.microsoft.com/vscode/devcontainers/python:3.10

# avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# system deps for fasttext and others
RUN apt-get update && apt-get install -y build-essential git curl libsndfile1 && rm -rf /var/lib/apt/lists/*

# create workspace user already exists in base image
USER vscode
WORKDIR /workspace

# Copy repo files for installing requirements in devcontainer build context
COPY . /workspace/

# Install poetry/pip packages will be run via postCreateCommand
