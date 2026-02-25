FROM public.ecr.aws/lambda/python:3.10

# Native runtime libs needed by MXNet
RUN yum -y install libgomp libquadmath && yum clean all

# Pin NumPy to a version compatible with mxnet 1.9.x
RUN pip install --no-cache-dir "numpy==1.23.5"

# Install MXNet (will then use the pinned NumPy)
RUN pip install --no-cache-dir "mxnet==1.9.1"

COPY inference.py ${LAMBDA_TASK_ROOT}/inference.py
COPY model.tar.gz ${LAMBDA_TASK_ROOT}/model.tar.gz

CMD ["inference.lambda_handler"]
