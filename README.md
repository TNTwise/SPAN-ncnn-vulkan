# SPAN-ncnn-vulkan
### Custom models:
- <a href="https://openmodeldb.info/models/4x-ClearRealityV1">ClearRealityV1 (4X)</a> by Kim2091
- <a href="https://github.com/terrainer/AI-Upscaling-Models/tree/main/4xSPANkendata">SPANkendata (4X)</a> by terrainer
- <a href="https://openmodeldb.info/models/2x-span-anime-pretrain">2x-span-anime-pretrain (2X)</a> by Kim2091
- <a href="https://github.com/Phhofm/models"> 2xHFA2kSpan (2x)</a> by Phhofm
- <a href="https://openmodeldb.info/models/4x-Nomos8k-span-otf-strong"> 4x-Nomos8k-span-otf-strong (4x)</a> by Helaman
- <a href="https://openmodeldb.info/models/4x-Nomos8k-span-otf-medium"> 4x-Nomos8k-span-otf-medium (4x)</a> by Helaman
- <a href="https://openmodeldb.info/models/4x-Nomos8k-span-otf-weak"> 4x-Nomos8k-span-otf-weak (4x)</a> by Helaman <br/>
- <a href="https://github.com/Sirosky/Upscale-Hub/releases/tag/AniSD"> AniSD models (4x)</a> by SiroSky <br/>




ncnn implementation of SPAN, Swift Parameter-free Attention Network for Efficient Super-Resolution

span-ncnn-vulkan uses [ncnn project](https://github.com/Tencent/ncnn) as the universal neural network inference framework.


Download Windows/Linux/MacOS Executable for Intel/AMD/Nvidia GPU

**https://github.com/tntwise/span-ncnn-vulkan/releases**

This package includes all the binaries and models required. It is portable, so no CUDA or PyTorch runtime environment is needed :)

## About SPAN

SPAN (Swift Parameter-free Attention Network for Efficient Super-Resolution)

https://github.com/hongyuanyu/SPAN

Cheng Wan Hongyuan Yu Zhiqi Li Yihang Chen Yajun Zou Yuqing Liu Xuanwu Yin Kunlong Zu


## Usages

Input one image, output one upscaled frame image.

### Example Commands

```shell
./span-ncnn-vulkan -m models/ -n spanx4_ch48 -s 4 -i 0.jpg  -o 01.jpg
./span-ncnn-vulkan -m models/ -n spanx4_ch48 -s 4 -i input_frames/ -o output_frames/
```

Example below runs on CPU, Discrete GPU, and Integrated GPU all at the same time. Uses 2 threads for image decoding, 4 threads for one CPU worker, 4 threads for another CPU worker, 2 threads for discrete GPU, 1 thread for integrated GPU, and 4 threads for image encoding.
```shell
./span-ncnn-vulkan -m models/ -n spanx4_ch48 -s 4 -i input_frames/ -o output_frames/ -g -1,-1,0,1 -j 2:4,4,2,1:4
```

### Video Upscaling with FFmpeg

```shell
mkdir input_frames
mkdir output_frames

# find the source fps and format with ffprobe, for example 24fps, AAC
ffprobe input.mp4

# extract audio
ffmpeg -i input.mp4 -vn -acodec copy audio.m4a

# decode all frames
ffmpeg -i input.mp4 input_frames/frame_%08d.png

# upscale 4x resolution
./span-ncnn-vulkan -m models/ -n spanx4_ch48 -s 4 -i input_frames -o output_frames

# encode interpolated frames in 48fps with audio
ffmpeg -framerate 24 -i output_frames/%08d.png -i audio.m4a -c:a copy -crf 20 -c:v libx264 -pix_fmt yuv420p output.mp4
```

### Full Usages

```console
Usage: span-ncnn-vulkan -i infile -o outfile [options]...

  -h                   show this help
  -i input-path        input image path (jpg/png/webp) or directory
  -o output-path       output image path (jpg/png/webp) or directory
  -s scale             upscale ratio (can be 2, 3, 4. default=4)
  -t tile-size         tile size (>=32/0=auto, default=0) can be 0,0,0 for multi-gpu
  -m model-path        folder path to the pre-trained models. default=models
  -n model-name        model name (default=spanx4_ch48, can be spanx4_ch48 | spanx2_ch52 | spanx4_ch48 | spanx4_ch52)
  -g gpu-id            gpu device to use (default=auto) can be 0,1,2 for multi-gpu
  -c cpu-only          use only CPU for upscaling, instead of vulkan
  -j load:proc:save    thread count for load/proc/save (default=1:2:2) can be 1:2,2,2:2 for multi-gpu
  -x                   enable tta mode
  -f format            output image format (jpg/png/webp, default=ext/png)
  -v                   verbose output
```

- `input-path` and `output-path` accept file directory
- `load:proc:save` = thread count for the three stages (image decoding + upscaling + image encoding), using larger values may increase GPU usage and consume more GPU memory. You can tune this configuration with "4:4:4" for many small-size images, and "2:2:2" for large-size images. The default setting usually works fine for most situations. If you find that your GPU is hungry, try increasing thread count to achieve faster processing.
- `pattern-format` = the filename pattern and format of the image to be output, png is better supported, however webp generally yields smaller file sizes, both are losslessly encoded
- `scale` = upscale multiplier, must match model.

If you encounter a crash or error, try upgrading your GPU driver:

- Intel: https://downloadcenter.intel.com/product/80939/Graphics-Drivers
- AMD: https://www.amd.com/en/support
- NVIDIA: https://www.nvidia.com/Download/index.aspx

## Build from Source

1. Download and setup the Vulkan SDK from https://vulkan.lunarg.com/
  - For Linux distributions, you can either get the essential build requirements from package manager
```shell
dnf install vulkan-headers vulkan-loader-devel
```
```shell
apt-get install libvulkan-dev
```
```shell
pacman -S vulkan-headers vulkan-icd-loader
```

2. Clone this project with all submodules

```shell
git clone https://github.com/tntwise/span-ncnn-vulkan.git
cd span-ncnn-vulkan
git submodule update --init --recursive
```

3. Build with CMake
  - You can pass -DUSE_STATIC_MOLTENVK=ON option to avoid linking the vulkan loader library on MacOS

```shell
mkdir build
cd build
cmake ../src
cmake --build . -j 4
```

### Model

| model | upstream version |
|---|---|
| spanx2_ch48 | spanx2_ch48 |
| spanx2_ch52 | spanx2_ch52 |
| spanx4_ch48 | spanx4_ch48 |
| spanx4_ch52 | spanx4_ch52 |


## Sample Images

### Original Image

![origin0](images/in0.png)


### Upscale 4X with spanx4_ch48 model

```shell
./span-ncnn-vulkan -m models/ -n spanx4_ch48 -s 4 -i 0.png -o out.png
```

![span](images/out0.png)

### Upscale 2X with 2xHFA2kSPAN model

```shell
./span-ncnn-vulkan -m custom_models/ -n 2xHFA2kSPAN_27k -s 2 -i 0.png -o 2xHFA2kSPAN.png
```

![2xHFA2xSPAN](images/2xHFA2kSPAN.png)

## Original SPAN Project

- https://github.com/hongyuanyu/SPAN

## Other Open-Source Code Used

- https://github.com/Tencent/ncnn for fast neural network inference on ALL PLATFORMS
- https://github.com/webmproject/libwebp for encoding and decoding Webp images on ALL PLATFORMS
- https://github.com/nothings/stb for decoding and encoding image on Linux / MacOS
- https://github.com/tronkko/dirent for listing files in directory on Windows
