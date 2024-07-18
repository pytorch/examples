#!/bin/bash

# Create the "data" folder if it doesn't exist
mkdir -p data

# Download the files into the "data" folder
wget -P data https://public-nlp-datasets.s3.us-west-2.amazonaws.com/wikihowAll.csv
wget -P data https://public-nlp-datasets.s3.us-west-2.amazonaws.com/wikihowSep.csv
