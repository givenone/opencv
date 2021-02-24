## cv2.dnn.blobFromImage

Creates 4-dimensional blob from image. Optionally resizes and crops image from center, subtract mean values, scales values by scalefactor, swap Blue and Red channels.

Parameters
image	input image (with 1-, 3- or 4-channels).
size	spatial size for output image
mean	scalar with mean values which are subtracted from channels. Values are intended to be in (mean-R, mean-G, mean-B) order if image has BGR ordering and swapRB is true.
scalefactor	multiplier for image values.
swapRB	flag which indicates that swap first and last channels in 3-channel image is necessary.
crop	flag which indicates whether image will be cropped after resize or not
ddepth	Depth of output blob. Choose CV_32F or CV_8U.
if crop is true, input image is resized so one side after resize is equal to corresponding dimension in size and another one is equal or larger. Then, crop from the center is performed. If crop is false, direct resize without cropping and preserving aspect ratio is performed.

Returns
4-dimensional Mat with NCHW dimensions order.
Examples:
samples/dnn/classification.cpp, samples/dnn/colorization.cpp, samples/dnn/object_detection.cpp, samples/dnn/openpose.cpp, and samples/dnn/segmentation.cpp.

## <font style="color:rgb(50,120,229)"> OpenCV DNN Module</font>
Using the DNN module in OpenCV comprises of the following steps : 
1. Load the network in memory using `readNet` function.
2. Convert input image to blob using `blobFromImage` function
3. Set the blob as the input of network.
4. Make a `forward` pass and get the network output.
5. Process the output

Let us see the major functions and their arguments in more detail.
### <font style="color:rgb(8,133,37)">2.1. Loading the Network</font>
The network is loaded into memory using different functions for different frameworks. For example, `readNetFromTensorflow()` for Tensorflow models, `readNetFromCaffe()` for caffe models and so on. From OpenCV > 3.4.1, there is another function `readNet()`, which figures out the framework from its arguments and loads the network. 

#### <font style = "color:rgb(8,133,37)">Tensorflow</font>
```python
retval	=	cv.dnn.readNetFromTensorflow(	model[, config]	)
```

- `model` is the file which contains the weights of the network. 
- `Config` is the file which defines the network.

#### <font style = "color:rgb(8,133,37)">Caffe </font>

```python
retval	=	cv.dnn.readNetFromCaffe( prototxt, model)
```

- `model` is the file which contains the weights of the network. 
- `prototxt` is the file which defines the network.


**Note** that the model and config is in opposite order for Caffe and Tensorflow.

### <font style="color:rgb(8,133,37)">2.2. Convert image to blob</font>
```python
retval	=	cv.dnn.blobFromImage(	image[, scalefactor[, size[, mean[, swapRB[, crop]]]]]	)
```
Creates 4-dimensional blob from image. Optionally resizes and crops image from center, subtract mean values, scales values by scalefactor, swap Blue and Red channels

**Parameters**
* `image` : input image (with 1-, 3- or 4-channels).
* `size` : spatial size for output image
* `mean` : scalar with mean values which are subtracted from channels. Values are intended to be in (mean-R, mean-G, mean-B) order if image has BGR ordering and swapRB is true.
* `scalefactor` : multiplier for image values.
* `swapRB` : flag which indicates that swap first and last channels in 3-channel image is necessary.
* `crop` : flag which indicates whether image will be cropped after resize or not

If crop is true, input image is resized so one side after resize is equal to corresponding dimension in size and another one is equal or larger. Then, crop from the center is performed. 

If crop is false, direct resize without cropping and preserving aspect ratio is performed.


### <font style="color:rgb(8,133,37)">2.3. Generate output</font>
```python
outputBlobs	=	net.forward(	[, outputBlobs[, outputName]])
```

The function runs a forward pass to compute the output of the layer with name *outputName*.

**Parameters**
* `outputBlobs`	: contains all output blobs for specified layer.
* `outputName` : name for layer which output is needed to get


Let us see if the models recognize a Panda.