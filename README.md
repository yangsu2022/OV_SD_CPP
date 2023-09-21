# OV_SD_CPP
C++ pipeline with OpenVINO native API for Stable Diffusion v1.5 with LMS Discrete Scheduler

## Step 1: Prepare environment
C++ pipeline loads the Lora safetensors via Pybind
```shell
conda create -n SD-CPP python==3.10
conda activate SD-CPP
conda pip install numpy safetensors pybind11
```
C++ Packages:
* OpenVINO:
Tested with [OpenVINO 2023.1.0.dev20230811 pre-release](https://storage.openvinotoolkit.org/repositories/openvino/packages/master/2023.1.0.dev20230811/)
* Boost: Install with `sudo apt-get install libboost-all-dev` for LMSDiscreteScheduler's integration
* OpenCV: Install with `sudo apt install libopencv-dev` for image saving

Notice: 

SD Preparation in two steps above could be auto implemented with build_dependencies.sh in the scripts directory 
```shell
chmod +x build_dependencies.sh
./build_dependencies.sh
```

## Step 2: Prepare SD model and Tokenizer model
* SD v1.5 model:

Refer [this link](https://github.com/intel-innersource/frameworks.ai.openvino.llm.bench/blob/main/public/convert.py#L124-L184) to generate SD v1.5 model, reshape to (1,3,512,512) for best performance.

With downloaded models, the model conversion from PyTorch model to OpenVINO IR could be done with script convert_model.py in the scripts directory 

```shell
python -m convert_model.py -b 1 -t <INT8|FP16|FP32> -sd Path_to_your_SD_model
```
Notice: Now the pipeline support batch size = 1 only

Lora enabling with safetensors, refer [this blog](https://blog.openvino.ai/blog-posts/enable-lora-weights-with-stable-diffusion-controlnet-pipeline) 

SD model [dreamlike-anime-1.0](https://huggingface.co/dreamlike-art/dreamlike-anime-1.0) and Lora [soulcard](https://civitai.com/models/67927?modelVersionId=72591) are tested in this pipeline

* Tokenizer model:
  
1. The script convert_sd_tokenizer.py in the scripts dir could serialize the tokenizer model IR

2. Build OV extension:
 
```git clone https://github.com/apaniukov/openvino_contrib/  -b tokenizer-fix-decode```

Refer to PR OpenVINO [custom extension](https://github.com/openvinotoolkit/openvino_contrib/pull/687) ( new feature still in experiments )

3. read model with extension in the SD pipeline 

Notice:

Tokenizer Model IR and built extension file are provided in this repo

![image](https://github.com/yangsu2022/OV_SD_CPP/assets/102195992/bac14f96-69c9-4ec4-b694-21a62a3176f4)

## Step 3: Build Pipeline

```shell
source /Path_to_your_OpenVINO_package/setupvars.sh
conda activate SD-CPP
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```
Notice:

Use the same OpenVINO environment as the tokenizer extension

## Step 4: Run Pipeline
```shell
 ./SD-generate -t <text> -n <negPrompt> -s <seed> --height <output image> --width <output image> -d <debugLogger> -e <useOVExtension> -r <readNPLatent> -m <modelPath> -p <precision> -l <lora.safetensors> -a <alpha> -h <help>
```

Usage:
  OV_SD_CPP [OPTION...]

* `- t, --text arg`     Initial positive prompt for SD  (default: cyberpunk cityscape like Tokyo New York  with tall buildings at dusk golden hour cinematic lighting)
* `-n, --negPrompt arg` Defaut negative prompt is empty with space (default: )
* `-s, --seed arg`      Number of random seed to generate latent (default: 42)
* `--height arg`        Height of output image (default: 512)
* `--width arg`         Width of output image (default: 512)
* `-d, --debugLogger`   Generate logging into log.txt for debug
* `-e, --useOVExtension`Use OpenVINO extension for tokenizer
* `-r, --readNPLatent`  Read numpy generated latents from file
* `-m, --modelPath arg` Specify path of SD model IR (default: /home/openvino/fiona/SD/SD_ctrlnet/dreamlike-anime-1.0)
* `-p, --precision arg` Specify precision of SD model IR (default: FP16_static)
* `-l, --loraPath arg`  Specify path of lora file. (*.safetensors). (default: /home/openvino/fiona/SD/Stable-Diffusion-NCNN/assets/lora/soulcard.safetensors)
* `-a, --alpha arg`     alpha for lora (default: 0.75)
* `-h, --help`          Print usage

Example:

Positive prompt: cyberpunk cityscape like Tokyo New York  with tall buildings at dusk golden hour cinematic lighting

Negative prompt: (empty, here couldn't use OV tokenizer, check the issues for details)  

Read the numpy latent instead of C++ std lib for the alignment with Python pipeline 

* Generate image without lora ` ./SD-generate -r -l "" `

![image](https://github.com/intel-sandbox/OV_SD_CPP/assets/102195992/66047d66-08a3-4272-abdc-7999d752eea0)

* Generate image with soulcard lora ` ./SD-generate -r `

![image](https://github.com/intel-sandbox/OV_SD_CPP/assets/102195992/0f6e2e3e-74fe-4bd4-bb86-df17cb4bf3f8)

* Generate the debug logging into log.txt: ` ./SD-generate -d`

## Benchmark:
The performance and image quality of C++ pipeline are aligned with Python

To align the performance with [Python SD pipeline](https://github.com/FionaZZ92/OpenVINO_sample/tree/master/SD_controlnet),
C++ pipeline will print the duration of each model inferencing only

For the diffusion part, the duration is for all the steps of Unet inferencing, which is the bottleneck

For the generation quality, be careful with the negative prompt and random latent generation

## Limitation:
* Pipeline features:
```shell
- Batch size 1
- LMS Discrete Scheduler
- Text to image
- CPU
```
* Program optimization:
now parallel optimization with std::for_each only and add_compile_options(-O3 -march=native -Wall) with CMake
* The pipeline with INT8 model IR not improve the performance  
* Lora enabling only for FP16
* Random generation fails to align, C++ random with MT19937 results is differ from numpy.random.randn(). Hence, please use `-r, --readNPLatent` for the alignment with Python 
* OV extension tokenizer cannot recognize the special character, like “.”, ”,”, “”, etc. When write prompt, need to use space to split words, and cannot accept empty negative prompt.
So use default tokenizer without config `-e, --useOVExtension`, when negative prompt is empty
  
## Setup in Windows 10 with VS2019:
1. Python env: Setup Conda env SD-CPP with the anaconda prompt terminal
2. C++ dependencies:
* OpenVINO and OpenCV:

Download and setup Environment Variable: add the path of bin and lib
(System Properties -> System Properties -> Environment Variables -> System variables -> Path )
* Boost:
```shell
1. Download from https://sourceforge.net/projects/boost/files/boost-binaries/1.83.0/
2. Unzip
3. Setup: bootstrap.bat 
4. Build: b2.exe
5. Install: b2.exe install
```
Installed boost in the path C:/Boost, add CMakeList with `SET(BOOST_ROOT"C:/Boost")`
3. Setup of conda env SD-CPP and Setup OpenVINO with setupvars.bat
4. CMake with build.bat like:
```shell
rmdir /Q /S build
mkdir build
cd build
cmake -G "Visual Studio 16 2019" -A x64 ^
 -DCMAKE_BUILD_TYPE=Release ^
      ..
cmake --build . --config Release
cd ..
```
5. Setup of Visual Studio with release and x64, and build: open .sln file in the build Dir
6. Run the `SD_generate.exe`
