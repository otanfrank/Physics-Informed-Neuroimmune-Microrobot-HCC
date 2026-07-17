FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime
WORKDIR /workspace
COPY requirements.txt pyproject.toml ./
COPY code ./code
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-deps .
ENTRYPOINT ["ni-pinn-train"]
