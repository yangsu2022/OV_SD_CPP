#!/usr/bin/env bash

# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# exit when any command fails
set -e

usage() {
    echo "SD Preparation in two steps: Python environment and C++ dependencies "

    exit 1
}

echo "---Part 1: prepare Python env for Lora enabing---"
# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda is not installed. Please install Conda before proceeding."
    exit 1
else
    echo "Conda is already installed."
fi

# Find the directory containing conda executable
conda_bin_dir=$(dirname $(which conda))
# Define the name of the conda environment
environment_name="SD-CPP"


# Check if the 'SD-CPP' conda environment exists
if ! conda env list | grep -q "SD-CPP"; then
    echo "The 'SD-CPP' conda environment does not exist."
    echo "Creating 'SD-CPP' environment and installing Python 3.10, numpy, safetensors, and pybind11..."
    conda create -n SD-CPP python==3.10 -y
    # Activate the conda environment using source
    source "${conda_bin_dir}/../etc/profile.d/conda.sh"
    conda activate "${environment_name}"
    conda install numpy safetensors pybind11 -y
    echo "Environment 'SD-CPP' created and packages installed."
else
    echo "The 'SD-CPP' conda environment already exists."
fi

# Check if "numpy" "safetensors" "pybind11" is successfully installed in the 'SD-CPP' environment
# List the packages to check
packages=("numpy" "safetensors" "pybind11")

# Initialize a variable to keep track of missing packages
missing_packages=""

# Check if each package is installed
for package in "${packages[@]}"; do
    if ! conda list -n SD-CPP | grep -q "$package"; then
        missing_packages+=" $package"
    fi
done

# Check if any packages are missing
if [ -z "$missing_packages" ]; then
    echo ""numpy" "safetensors" "pybind11" are installed in the 'SD-CPP' environment."
else
    echo "The following packages are missing in the 'SD-CPP' environment:$missing_packages"
    echo "Now install packages"
   
   # Activate the conda environment using source
    source "${conda_bin_dir}/../etc/profile.d/conda.sh"
    conda activate "${environment_name}"
    conda install numpy safetensors pybind11 -y
fi

echo
echo "---Part 2: prepare the dependencies: CMake, OpenCV, Boost and OpenVINO---"
# Check if CMake is installed
if ! command -v cmake &> /dev/null
then
    echo "CMake is not installed. Installing CMake..."
    sudo apt update
    sudo apt install cmake -y
else
    echo "CMake is already installed."
fi

# Check if OpenCV C++ libraries are installed using dpkg
if dpkg -l | grep libopencv &> /dev/null; then
    echo "OpenCV C++ libraries are already installed using dpkg."
elif [ -d "/usr/local/include/opencv4" ]; then
    echo "OpenCV C++ libraries are already installed in /usr/local/include/opencv4."
else
    echo "OpenCV C++ libraries are not installed. Installing OpenCV via apt..."
    sudo apt update
    sudo apt install libopencv-dev -y
fi

# Check if Boost is installed
if ! dpkg -l | grep libboost &> /dev/null; then
    echo "Boost is not installed."
    echo "Installing Boost..."
    sudo apt update
    sudo apt install libboost-all-dev -y
else
    echo "Boost is already installed."
fi

# Check if boost::math::quadrature::trapezoidal is supported
if g++ -E -x c++ - -I /usr/include -I /usr/local/include -I /usr/include/boost - <<<'#include <boost/math/quadrature/trapezoidal.hpp>' &> /dev/null; then
    echo "boost::math::quadrature::trapezoidal is supported."
else
    echo "boost::math::quadrature::trapezoidal is not supported."
fi


# Prompt the user if they want to install OpenVINO
read -p "Do you want to install OpenVINO Toolkit 2023.1.0 (C++) with conda-forge? (yes/no): " choice

if [ "$choice" = "yes" ]; then

    conda install -c conda-forge openvino=2023.1.0 -y

else
    echo "### OpenVINO 2023.1.0 install canceled. You can manually activate your OpenVINO environment."
fi

echo "### Finished all the preparation"  
echo "### Please activate the Python env and OpenVINO manually with command 'conda activate SD-CPP'"
echo "### Then build the pipeline with CMake following the README's guide"