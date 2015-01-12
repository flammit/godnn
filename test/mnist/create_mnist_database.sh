#!/usr/bin/env sh
# This scripts downloads the mnist data and unzips it.

echo "Creating Training Database"
go run create_db.go train-images-idx3-ubyte train-labels-idx1-ubyte train.db

echo "Creating Test Database"
go run create_db.go t10k-images-idx3-ubyte t10k-labels-idx1-ubyte t10k.db

echo "Done."