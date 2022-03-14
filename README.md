# cmsc-630

# Output:

The outputs can be found here: https://drive.google.com/file/d/1tcV00SBeyuHNdiT8-YuYh82sU1zCRcnH/view?usp=sharing

The directory includes:

For each image:
* Converting to RGB/Grayscale
* Histogram
* Histogram equalization
* Salt and pepper noise
* Gaussian noise
* Linear filtered image (On the output image after applying salt and pepper noise)
* Median filtered image (On the output image after applying Gaussian noise)

For each image class:
* Averaged histograms

# How to run:

The project runs in python 3.6. Required libraries are included in requirements.txt. To run the project:
* Setup virtual env
* run `pip install -r requirements.txt`
* sh run.sh

The user specified variables can be modified in `run.sh`
