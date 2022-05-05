#!/bin/bash

sudo apt update
sudo apt install -y docker.io nano

sudo tee /etc/docker/daemon.json > /dev/null <<EOT
{
  "data-root": "/mydata/docker-data"
}
EOT

sudo systemctl restart docker
sudo chmod 666 /var/run/docker.sock