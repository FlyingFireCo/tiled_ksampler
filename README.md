# tiled_ksampler

# Installation

`cd into ComfyUI/custom_nodes`

`git clone https://github.com/FlyingFireCo/tiled_ksampler.git`

Start comfyui


# Tiled KSampler:

add Tiled KSampler Node and use like KSampler


when tiling = 1 it will tile

when tiling = 0 it will behave like a normal KSampler


# Asymmetric Tiled KSampler:

add Asymmetric Tiled KSampler Node and use like KSampler


when tileX = 1 it will tile in the X direction

when tileY = 1 it will tile in the Y direction

# Circular VAE Decoder:

You'll need to use this when decoding the image, otherwise you'll get bleeding around the edges. It basically applies the Circular padding during decoding as well.

# Credit

Based off WAS Studio 

using the example for tiling from Automatic1111

GBJI for pointing to https://github.com/tjm35/asymmetric-tiling-sd-webui/ for asymmetric tiling

hrkljus1 for pointing out information about VAE Decoder removing bleeding


