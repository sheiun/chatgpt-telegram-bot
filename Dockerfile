FROM python:3.12-alpine

ENV PYTHONFAULTHANDLER=1 \
     PYTHONUNBUFFERED=1 \
     PYTHONDONTWRITEBYTECODE=1

RUN apk --no-cache add ffmpeg curl
RUN pip install uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

COPY . .

CMD ["uv", "run", "python", "bot/main.py"]