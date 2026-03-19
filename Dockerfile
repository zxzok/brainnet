# BrainNet Docker Image
# Includes nilearn, nibabel, and optional neuroimaging tool wrappers
FROM python:3.11-slim

LABEL maintainer="BrainNet <brainnet@example.com>"
LABEL description="BrainNet fMRI 脑网络动力学分析计算平台"

# System dependencies for matplotlib, HDF5, and nilearn
RUN apt-get update && apt-get install -y --no-install-recommends \
    libhdf5-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY pyproject.toml .
COPY requirements.txt .
RUN pip install --no-cache-dir -e ".[dev]" matplotlib

# Copy application code
COPY . .

# Default port
EXPOSE 6525

# Environment
ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg

CMD ["python", "web_app.py"]
