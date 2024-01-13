# Use the latest Ubuntu image as the base image
FROM ubuntu:latest

# Details about the image
LABEL org.opencontainers.image.title="ML Lab Image"
LABEL org.opencontainers.image.description="This image contains \
all the packages required for ML laboratory preinstalled."
LABEL org.opencontainers.image.source="https://github.com/kbdharun/ML-Lab"
LABEL org.opencontainers.image.authors="K.B.Dharun Krishna mail@kbdharun.dev"
LABEL org.opencontainers.image.vendor="kbdharun.dev"
LABEL org.opencontainers.image.licenses="GPL-3.0-only"

# Update and upgrade packages
RUN apt-get update && apt-get upgrade -y

# Install Python, pip, git, and Jupyter Notebook
RUN apt-get install -y python3 python3-pip git

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install Python packages from requirements file
RUN pip3 install -r /app/requirements.txt

# Set the working directory
WORKDIR /app

# Command to run when the container starts
CMD ["/bin/sh"]
