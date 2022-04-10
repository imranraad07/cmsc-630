# cmsc-630

# Part 2
The outputs can be found here: https://drive.google.com/file/d/1h6SEr7DAYG2fhPsNZehrI8fc0Vwd7YCl/view?usp=sharing

The directory includes:
For each image:
* Edge detection (Edge filter can be changed from run.h)
* Dilated image
* Eroded image
* Image after histogram thresholding
* Image after kmeans clustering (k can be changed from run.sh)


# Part 1
## Output:

The outputs can be found here: https://drive.google.com/file/d/1tcV00SBeyuHNdiT8-YuYh82sU1zCRcnH/view?usp=sharing

The directory includes:

For each image:
* Converting to RGB/Grayscale
* Histogram
* Histogram equalization
* Image based on histogram equalization
* Quantized image
* Salt and pepper noise
* Gaussian noise
* Linear filtered image (On the output image after applying salt and pepper noise)
* Median filtered image (On the output image after applying Gaussian noise)

For each image class:
* Averaged histograms

## How to run:

The project runs in python 3.6. Required libraries are included in requirements.txt. To run the project:
* Setup virtual env
* run `pip install -r requirements.txt`
* sh run.sh

The user specified variables can be modified in `run.sh`
