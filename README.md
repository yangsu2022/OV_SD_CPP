# OV_SD_CPP
The pure C++ text-to-image pipeline, driven by the OpenVINO native API for Stable Diffusion v1.5 with LMS Discrete Scheduler, supports both static and dynamic model inference. It includes advanced features like Lora integration with safetensors and OpenVINO extension for tokenizer. This demo has been tested on the Windows platform.
## Step 1: Prepare environment

C++ Packages:
* CMake: Cross-platform build tool
* OpenVINO: Model inference
* Eigen3: Lora enabling

SD Preparation could be auto implemented with `build_dependencies.sh`. This script provides 2 ways to install `OpenVINO 2023.1.0`: [conda-forge](https://anaconda.org/conda-forge/openvino) and [Download archives](https://storage.openvinotoolkit.org/repositories/openvino/packages/2023.1/windows/).
```shell
cd scripts
chmod +x build_dependencies.sh
./build_dependencies.sh
...
"use conda-forge to install OpenVINO Toolkit 2023.1.0 (C++), or download from archives? (yes/no): "
```

Notice: Use Intel sample [writeOutputBmp function](https://github.com/openvinotoolkit/openvino/blob/539b5a83ba7fcbbd348e4dc308e4a0f2dee8343c/samples/cpp/common/utils/include/samples/common.hpp#L155) instead of OpenCV for image saving.


## Step 2: Prepare SD model and Tokenizer model
#### SD v1.5 model:
1. Prepare a conda python env and install dependencies:
    ```shell
    cd scripts
    conda create -n SD-CPP python==3.10
    pip install -r requirements.txt
    ```
    **Notice: tested with transformers=4.35.2 and 4.34.1. If has issues with converting model, could downgrade the pkg.
    
2. Download a huggingface SD v1.5 model like [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5), here another model [dreamlike-anime-1.0](https://huggingface.co/dreamlike-art/dreamlike-anime-1.0) is used to test for the lora enabling. Ref to the official website for [model downloading](https://huggingface.co/docs/hub/models-downloading). Here, you could also use the api to download model with model_id(like "runwayml/stable-diffusion-v1-5" for config "-sd").
    ```shell
    cd scripts
    python -m convert_sd_model.py -b 1 -t FP16 -sd runwayml/stable-diffusion-v1-5
    ```

3. Model conversion from PyTorch model to OpenVINO IR via [optimum-intel](https://github.com/huggingface/optimum-intel). Please use the script convert_sd_model.py to convert the model into `FP16_static` or `FP16_dyn`, which will be saved into the SD folder.  
    ```shell
    cd scripts
    python -m convert_sd_model.py -b 1 -t FP16 -sd Path_to_your_SD_model
    python -m convert_sd_model.py -b 1 -t FP16 -sd Path_to_your_SD_model -dyn
    ```
    **Notice: Now the pipeline support batch size = 1 only, ie. static model (1,3,512,512)

#### LCM model:
Download model `SimianLuo/LCM_Dreamshaper_v7` and convert to openvino FP16_dyn IR `../models/lcm/dreamshaper_v7/FP16_dyn/` via one script `convert_lcm_model.py`, which is based on the openvino notebook [263-latent-consistency-models-image-generation](https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/263-latent-consistency-models-image-generation/263-latent-consistency-models-image-generation.ipynb). For proxy issue, if failed to use the script to download model, try the `huggingface-cli` tool with `hf-mirror`.
    ```shell
    cd scripts
    python -m convert_lcm_model.py
    ```
#### Lora enabling with safetensors

Refer [this blog for python pipeline](https://blog.openvino.ai/blog-posts/enable-lora-weights-with-stable-diffusion-controlnet-pipeline), the safetensor model is loaded via [src/safetensors.h](https://github.com/hsnyder/safetensors.h). The layer name and weight are modified with `Eigen Lib` and inserted into the SD model with `ov::pass::MatcherPass` in the file `src/lora_cpp.hpp`. 

SD model [dreamlike-anime-1.0](https://huggingface.co/dreamlike-art/dreamlike-anime-1.0) and Lora [soulcard](https://civitai.com/models/67927?modelVersionId=72591) are tested in this pipeline. Here, Lora enabling only for FP16. 

Download and put safetensors and model IR into the models folder. 

#### Tokenizer model:
3 steps for OpenVINO extension for tokenizer:
  1. The script `convert_sd_tokenizer.py` in the scripts folder could serialize the tokenizer model IR
  2. Build OV extension:
      ```git clone https://github.com/apaniukov/openvino_contrib/  -b tokenizer-fix-decode```
      Refer to PR OpenVINO [custom extension](https://github.com/openvinotoolkit/openvino_contrib/pull/687) ( new feature still in experiments )
  3. Read model with extension in the SD pipeline 

Important Notes:  
- Ensure you are using the same OpenVINO environment as the tokenizer extension.
- When the negative prompt is empty, use the default tokenizer without any configuration (`-e` or `--useOVExtension`).
- You can find the Tokenizer Model IR and the built extension file in this repository:
`extensions/libuser_ov_extensions.so`
`models/tokenizer/`

## Step 3: Build Pipeline

```shell
conda activate SD-CPP
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

## Step 4: Run Pipeline
```shell
 ./SD-generate -t <text> -n <negPrompt> -s <seed> --height <output image> --width <output image> -d <debugLogger> -e <useOVExtension> -r <readNPLatent> -m <modelPath> -p <precision> -l <lora.safetensors> -a <alpha> -h <help>
```

Usage:
  OV_SD_CPP [OPTION...]

* `-p, --posPrompt arg` Initial positive prompt for SD  (default: cyberpunk cityscape like Tokyo New York  with tall buildings at dusk golden hour cinematic lighting)
* `-n, --negPrompt arg` Default is empty with space (default: )
* `-d, --device arg`    AUTO, CPU, or GPU (default: CPU)
* `--step arg`          Number of diffusion step ( default: 20)
* `-s, --seed arg`      Number of random seed to generate latent (default: 42)
* `--num arg`           Number of image output(default: 1)
* `--height arg`        Height of output image (default: 512)
* `--width arg`         Width of output image (default: 512)
* `--log arg`           Generate logging into log.txt for debug
* `--lcm arg`           Use LCM diffusion pipeline with LCM scheduler
* `-c, --useCache`      Use model caching
* `-e, --useOVExtension`Use OpenVINO extension for tokenizer
* `-r, --readNPLatent`  Read numpy generated latents from file
* `-m, --modelPath arg` Specify path of SD model IR (default: ../models/sd/dreamlike-anime-1.0)
* `-t, --type arg`      Specify the type of SD model IR (FP16_static or FP16_dyn) (default: FP16_static)
* `-l, --loraPath arg`  Specify path of lora file. (*.safetensors). (default: )
* `-a, --alpha arg`     alpha for lora (default: 0.75)
* `-h, --help`          Print usage

Example:

Positive prompt: cyberpunk cityscape like Tokyo New York  with tall buildings at dusk golden hour cinematic lighting

Negative prompt: (empty, here couldn't use OV tokenizer, check the issues for details)  

Read the numpy latent instead of C++ std lib for the alignment with Python pipeline 

* Using SDv1.5 model to generate image without lora ` ./SD-generate -r -l "" `

![image](https://github.com/intel-sandbox/OV_SD_CPP/assets/102195992/66047d66-08a3-4272-abdc-7999d752eea0)

* Using SDv1.5 model to generate image with soulcard lora ` ./SD-generate -r `

![image](https://github.com/intel-sandbox/OV_SD_CPP/assets/102195992/0f6e2e3e-74fe-4bd4-bb86-df17cb4bf3f8)

* Using LCM model and LCM scheduler to generate image without lora (reading noise from files)
 ` ./SD-generate -m ../models/lcm/dreamshaper_v7/ -r -l "" --lcm --step 4 -p "a beautiful pink unicorn"`

![lcm_read_noise](https://github.com/yangsu2022/OV_SD_CPP/assets/102195992/fb9e1c1b-ae60-4f9c-8aad-79951bbf3f8b)

* Using SDv1.5 model to generate the debug logging into log.txt: ` ./SD-generate --log`
* Using SDv1.5 model to generate different size image with dynamic model(C++ lib generated latent): ` ./SD-generate -m Your_Own_Path/sd/dreamlike-anime-1.0 -l '' -t FP16_dyn --height 448 --width 704 `

![image](https://github.com/yangsu2022/OV_SD_CPP/assets/102195992/9bd58b64-6688-417e-b435-c0991247b97b)

## Benchmark:
The performance and image quality of C++ pipeline are aligned with Python

To align the performance with [Python SD pipeline](https://github.com/FionaZZ92/OpenVINO_sample/tree/master/SD_controlnet), C++ pipeline will print the duration of each model inferencing only

For the diffusion part, the duration is for all the steps of Unet inferencing, which is the bottleneck

For the generation quality, be careful with the negative prompt and random latent generation. C++ random generation with MT19937 results is differ from numpy.random.randn(). Hence, please use -r, --readNPLatent for the alignment with Python(this latent file is for output image 512X512 only)

Program optimization: In addition to inference optimization, now parallel optimization with std::for_each only and add_compile_options(-O3 -march=native -Wall) with CMake 
  
## Setup in Windows 11 with VS2022:
1. Setup of C++ dependencies:
* OpenVINO:
To deployment without Conda: [Download archives* with OpenVINO](https://storage.openvinotoolkit.org/repositories/openvino/packages/2023.1/windows/), unzip and setup environment vars: run `setupvars.bat` within the terminal Command Prompt for VS

* Eigen:
```shell
1. Download from https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip 
2. Unzip to path C:/Eigen3/eigen-3.4.0 
3. Run next step's build.bat will report error: not found Eigen3Config.cmake/eigen3-config.cmake
- Create build folder for Eigen and Open VS in this path C:/Eigen3/eigen-3.4.0/build
- Open VS's developer PS terminal to do "cmake .." and redo the CMake 
```
Ref: [not found Eigen3Config.cmake/eigen3-config.cmake](https://stackoverflow.com/questions/48144415/not-found-eigen3-dir-when-configuring-a-cmake-project-in-windows)

2. CMake with Visual Studio and release config, run the script build.bat:

```shell
cd scripts
build.bat
```

3. Put safetensors and model IR into the models folder with the following default path:
`models\sd\dreamlike-anime-1.0\FP16_static` 
`models\soulcard.safetensors`

4. Run with prompt:  

```shell
cd PROJECT_SOURCE_DIR\build
.\Release\SD-generate.exe -l ''  // without lora
.\Release\SD-generate.exe -l ../models/soulcard.safetensors
```
```shell
Notice: 
* must run command line within path of build, or .exe could not find the models
* .exe is in the Release folder 
```

5. Run within Visual Studio (open .sln file in the `build` folder)

Notice: 
* has issue to build OpenVINO custom extension on Windows platform, so use the default tokenizer.
* VS Debugging needs to [build OpenVINO source code](https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/build_windows.md) for Debug version.  
