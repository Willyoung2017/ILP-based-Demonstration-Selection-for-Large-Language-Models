#!/bin/bash

#  download and unzip data files:
wget https://www.cs.cmu.edu/~pengchey/reg_attn_data.zip
unzip reg_attn_data.zip

# Copy smcalflow_cs from reg_atten_data/data/smcalflow_cs to the data directory
cp -R reg_atten_data/data/smcalflow_cs data/