FROM openfabric/tee-python-cpu:dev

WORKDIR /app

RUN apt-get update && apt-get install -y libmagic1 && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir --upgrade poetry

COPY pyproject.toml poetry.lock ./

# Install ALL project dependencies using Poetry
# --only main: installs only main dependencies (defined under [tool.poetry.dependencies])
# --no-root: if your project isn't structured as a standard installable package itself
# -vvv: for verbose output during build, can be removed for final version if desired
RUN python3 -m poetry install --only main --no-interaction --no-ansi --no-root -vvv


COPY . .

EXPOSE 8888

RUN chmod +x ./start.sh

# CMD ["sh", "./start.sh"]
CMD ["python3", "./ignite.py"]