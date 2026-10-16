# The base image must provide Python 3.12 and CUDA-enabled PyTorch, plus numpy,
# safetensors and huggingface_hub. Replace it with an image built for your GPU.
FROM qwen38-flash-dgx:v0.31

# The timestral package, pinned. --no-deps keeps the base image's torch and numpy.
# A GitHub archive because the image has no git; pip records the URL, which is
# how /v1/info reports the commit.
ARG TIMESTRAL_COMMIT=a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff
RUN pip install --no-deps --no-cache-dir --root-user-action=ignore \
        "https://github.com/mistralai/timestral/archive/${TIMESTRAL_COMMIT}.tar.gz" \
    && python3 -c "import timestral"

# HOME=/tmp: the container runs as an unprivileged uid (see docker-compose.yml)
# that may have no home directory in the base image.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app
COPY timestral_abel/ timestral_abel/
COPY scripts/ scripts/
EXPOSE 18320

# /healthz answers 503 until the model is loaded and warmed up; urlopen raises
# (exit 1) on that and on connection errors. Port must match listen_port.
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD ["python3", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:18320/healthz', timeout=4)"]

# Replace the base image's entrypoint.
ENTRYPOINT ["python3", "-m", "timestral_abel"]
CMD []
