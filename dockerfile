# Use the official TensorFlow image (includes Python + TF preinstalled) [cite: 1]
FROM tensorflow/tensorflow:2.15.0

# Set a non-root user for security
RUN groupadd --system appuser && useradd --system --gid appuser appuser

# Set working directory to the user's home
WORKDIR /home/appuser/app

# Create a new user with a specific ID to ensure consistent permissions
RUN chown -R appuser /home/appuser/app

# Copy the `uv` binary from its official image. [cite: 2]
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set up the Python environment (no need for a separate builder stage)
ENV VIRTUAL_ENV=/home/appuser/app/.venv \
    PATH="/home/appuser/app/.venv/bin:$PATH"

# Copy dependency files first to leverage Docker's layer caching
# You need `pyproject.toml` and `uv.lock` in your local directory
COPY --chown=appuser:appuser pyproject.toml uv.lock ./


# Temporary marker to stop the build here
#FROM tensorflow/tensorflow:2.15.0 AS debug-stage


# Ensure cache dir exists and is writable
RUN mkdir -p /home/appuser/.cache/uv && \
    chown -R appuser:appuser /home/appuser/.cache

# Switch temporary to the root user 
USER root
# Install packages into a virtual environment using `uv sync`
RUN --mount=type=cache,target=/home/appuser/.cache/uv \
    uv sync

# Switch to the non-root user for subsequent commands
USER appuser
# Copy the rest of your application code
COPY --chown=appuser:appuser . . 

# Run the script using the correct Python interpreter from the virtual environment
#CMD ["/home/appuser/app/.venv/bin/python", "main.py"]

#CMD ["uv", "run", "python", "main.py", "--task", "summarize",  "--movie_id", "550"]
CMD ["uv", "run", "python", "main.py", "--task", "sentiment",  "--text", "Bad acting and terrible plot."]