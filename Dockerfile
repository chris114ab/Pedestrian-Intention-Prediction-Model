# FROM tells Docker which base image to use for this build 
FROM bitnami/pytorch:2.2.0
# COPY is used to copy tftest.py from the local machine 
# to a location inside the image
USER root
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Update package lists and install libgl1-mesa-glx
COPY main.py /tmp
COPY data /tmp/data
COPY ml /tmp/ml
COPY tracker /tmp/tracker
COPY utils.py /tmp
COPY data.py /tmp
COPY movenet_thunder.tflite /tmp

# ADD data /tmp/data
# ENV can be used to set environment variables
ENV TFTEST_ENV_VAR 12345
# RUN is used to execute a command in the image

# WORKDIR configures the current working directory that 
# the CMD will be executed within. Since the COPY
# command above puts the script under /tmp, we'll set 
# it to the same location so we can run the script with 
# no path prefix
WORKDIR /tmp
RUN pip install opencv-python-headless
RUN pip install transformers
RUN pip install pillow
RUN pip install scikit-learn
RUN pip install tensorflow


# CMD defines the command that containers will run when # created from this image
CMD ["python", "main.py"]