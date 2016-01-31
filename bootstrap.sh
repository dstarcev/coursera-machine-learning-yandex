#!/usr/bin/env bash
apt-get install -y git

apt-get install -y --upgrade python python-pip python-dev \
  ipython-notebook python-numpy python-scipy python-pandas python-matplotlib
  
pip install -U sklearn