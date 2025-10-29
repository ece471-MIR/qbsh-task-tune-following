#!/bin/bash

REPOROOT=$(git rev-parse --show-toplevel)
echo "root of repo is $REPOROOT"

if [ ! -f /tmp/data.zip ]; then
    echo "downloading dataset"

    wget \
        -O /tmp/data.zip \
        http://mirlab.org/dataSet/public/MIR-QBSH.zip
else
    echo "dataset already downloaded"
fi
echo "dataset saved to /tmp/data.zip"

echo 'checking...'
if ! sha256sum --check $REPOROOT/data.sha; then
    echo 'Download failed!'
    exit 1
fi

echo "extracting to $REPOROOT/data, this will take a couple minutes"
mkdir -p $REPOROOT/data
unzip /tmp/data.zip -d $REPOROOT/data >> /dev/null
