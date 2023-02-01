# syntax=docker/dockerfile:1

ARG  PY_VERSION=3.10
FROM python:${PY_VERSION}-buster as builder

WORKDIR /py4ai-analytics

RUN apt-get update && apt-get upgrade -y

COPY LICENSE MANIFEST.in versioneer.py setup.py pyproject.toml README.md Makefile ./
COPY requirements requirements
COPY py4ai py4ai
COPY tests tests

RUN addgroup --system tester && adduser --system --group tester
RUN chown -R tester:tester /py4ai-analytics
ENV PATH ${PATH}:/home/tester/.local/bin
USER tester

# change to the tester user: switch to a non-root user is a best practice.
RUN make checks

FROM python:${PY_VERSION}-slim-buster
WORKDIR /py4ai-analytics
COPY --from=builder /py4ai-analytics/dist /py4ai-analytics/dist

RUN addgroup --system runner && adduser --system --group runner
RUN chown -R runner:runner /py4ai-analytics
ENV PATH ${PATH}:/home/runner/.local/bin
USER runner

RUN ls -t ./dist/*.tar.gz | xargs pip install
ENTRYPOINT ["python"]
