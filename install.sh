#!/usr/bin/env bash

#############################################################
# INSTALL **Anti-"Donut Fairy" Intrusion System** ON RODETE #
#############################################################

# 2. INSTALL THE DEPENDENCIES

sudo apt install -y libopencv-dev python-numpy python-opencv
sudo pip install -r requirements.txt
echo "alias adfis='$PWD/anti-donut_fairy.py'" >> ~/.bashrc
