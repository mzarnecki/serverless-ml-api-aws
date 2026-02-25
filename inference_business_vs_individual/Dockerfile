FROM public.ecr.aws/lambda/python:3.10

# TensorFlow (CPU) + runtime deps
# Pinning versions avoids surprises
RUN pip install --no-cache-dir \
    "tensorflow-cpu==2.13.1" \
    "numpy==1.24.3" \
    "h5py==3.9.0"

# Your inference code + model
COPY inference.py ${LAMBDA_TASK_ROOT}/inference.py
COPY trained_model.h5 ${LAMBDA_TASK_ROOT}/trained_model.h5

CMD ["inference.lambda_handler"]