## <font style="color:rgb(50,120,229)">Thresholding using OpenCV function</font>
We discussed about using loops and vector operations for performing thresholding in the previous units. Let us see how we can use the OpenCV function `cv2.threshold` to perform thresholding and then we'll also discuss whether we should use OpenCV functions or write our own functions.  

### <font style="color:rgb(8,133,37)">Function Syntax </font>

[**cv2.threshold**](https://docs.opencv.org/4.1.0/d7/d1b/group__imgproc__misc.html#gae8a4a146d1ca78c626a53577199e9c57) has the following syntax : 

`
retval, dst = cv.threshold(src, thresh, maxval, type[, dst])
`

Where,

Input:
- `src` is the input array ot image (multiple-channel, 8-bit or 32-bit floating point).
- `thresh` is the threshold value.
- `maxval` is the maximum value to use with the THRESH_BINARY and THRESH_BINARY_INV thresholding types.
- `type` is thethresholding type ( THRESH_BINARY, THRESH_BINARY_INV, etc )

Output:
- `dst` is the output array or image of the same size and type and the same number of channels as src.
- `retval`  is the threshold value if you use other thresholding types such as Otsu or Triangle 

Clearly, we can see that the vectorized code is much better than the code that uses loops. 

The OpenCV function is even faster than the one we wrote using numpy.

This is because they have written highly optimized code and made further optimizations based on the system architectre and Operating system you are using. It is highly recommeneded to use OpenCV functions instead of writing your own algorithm from scratch if it is already available in OpenCV. 

## <font style="color:rgb(50,120,229)">Morphological Operations</font>

We have already discussed about morphological operations like Erosion and Dilation in the previous video. Let's recap.

**Dilation** is used to merge or expand white regions which may be close to each other and 
## <font style="color:rgb(50,120,229)">Dilation</font>

### <font style="color:rgb(8,133,37)">Function Syntax</font>

```python
dst	=	cv.dilate(	src, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]]	)
```
**Parameters**

Both functions take the same set of arguments

- **`src`**	input image; the number of channels can be arbitrary, but the depth should be one of CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
- **`dst`**	output image of the same size and type as src.
- **`kernel`**	structuring element used for dilation; if elemenat=Mat(), a 3 x 3 rectangular structuring element is used.
- **`anchor`**	position of the anchor within the element; default value (-1, -1) means that the anchor is at the element center.
- **`iterations`**	number of times dilation is applied.
- **`borderType`**	pixel extrapolation method.
- **`borderValue`**	border value in case of a constant border

**Erosion** is used to separate or shrink white regions

In this section, we will discuss how to use dilate and erode operations available in OpenCV and in the next section, we will see what is going on under the hood of these operations and how to implement them on your own. You will also be asked to implement one version of these algorithms as a Homework.
## <font style="color:rgb(50,120,229)">Erosion</font>

### <font style="color:rgb(8,133,37)">Function Syntax</font>

```python
dst	=	cv.erode(	src, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]]	)
```


## <font style="color:rgb(50,120,229)">Opening and Closing</font>
As discussed in the video, we can combine erosion and dilation operations to perform some interesting operations on binary images. We can remove small black or white spots from a binary image. For example, we know that erosion removes white region and dilation adds white region. Thus, if we want to **remove small white spots**, we should perform **erosion followed by dilation** so that the smaller white spots are removed, whereas the bigger white blobs are unchanged. Similarly you can **remove black spots using dilation followed by erosion**.

Again, weird as it may sound, these operations are also given some names : **Opening and Closing**.

**Opening** refers Erosion followed by Dilation and these operations is used for clearing **white blobs** and 

**Closing** refers Dilation followed by Erosion and are used for clearing **black holes**

In this section, we will discuss how to use opening and closing operations on binary images. 

## <font style="color:rgb(50,120,229)">Opening and Closing using OpenCV</font>

In OpenCV, the opening and closing operations are implemented using **MorphologyEx** function.

To chose between the opening and closing operation to be performed we specify an argument in the function [**`MorphologyEx`**](https://docs.opencv.org/4.1.0/d4/d86/group__imgproc__filter.html?fbclid=IwAR1GtoDsIv4Fi8o7vrZ8SGb3bb1uiU_Nyt94fc9J2sHKF7FlbDNT1fq-kI0#ga67493776e3ad1a3df63883829375201f) definition. The argument for opening operation and closing operations are [**`MORPH_OPEN`**] and [**`MORPH_CLOSE`**] respectively.

### <font style="color:rgb(8,133,37)">Function Syntax</font>

### <font style="color:rgb(50,120,229)">Opening</font>

```python:
imageMorphOpened = cv2.morphologyEx( src, cv2.MORPH_OPEN, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]] )
```

### <font style="color:rgb(50,120,229)">Closing</font>

```python:
imageMorphClosed = cv2.morphologyEx( src, cv2.MORPH_CLOSE, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]] )
```

**Parameters**
- **`src`**	Source image. The number of channels can be arbitrary. The depth should be one of CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
- **`dst`**	Destination image of the same size and type as source image.
- **`op`**	Type of a morphological operation
- **`kernel`**	Structuring element. It can be created using getStructuringElement.
- **`anchor`**	Anchor position with the kernel. Negative values mean that the anchor is at the kernel center.
- **`iterations`**	Number of times erosion and dilation are applied.
- **`borderType`**	Pixel extrapolation method.
- **`borderValue`**	Border value in case of a constant border. The default value has a special meaning.

## Connected Component Analysis (CCA). 
```
th, imThresh = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY)
print(th, imThresh)
# Find connected components
_, imLabels = cv2.connectedComponents(imThresh)
```

It is a fancy name for labeling blobs in a binary image. So, it can also be used to count the number of blobs ( also called connected components ) in a binary image.
OpenCV defines 12 colormaps that can be applied to a grayscale image using the function `applyColorMap` to produce a pseudocolored image. We will use COLORMAP_JET for our example.  

```
# Normalize the image so that the min value is 0 and max value is 255.
imLabels = 255 * (imLabels - minVal)/(maxVal-minVal)

# Convert image to 8-bits unsigned type
imLabels = np.uint8(imLabels)

# Apply a color map
imColorMap = cv2.applyColorMap(imLabels, cv2.COLORMAP_JET)
```

## <font style="color:rgb(50,120,229)">Contours </font>
Contours are simply the boundaries of an object or part of object in an image. They are useful in shape analysis and object Detection/Recognition using traditional Computer Vision algorithms.

You can do a lot of analysis based on contours to arrive at a solution to many computer vision problems. 

In this Notebook, we will discuss many different aspects of Contour Analysis. Let's get started.

## <font style="color:rgb(50,120,229)">How to find Contours</font>
We have seen earlier that there are many algorithms for finding Contours. We will use the OpenCV function [**`findContours`**](https://docs.opencv.org/4.1.0/d3/dc0/group__imgproc__shape.html#gadf1ad6a0b82947fa1fe3c3d497f260e0) to get the boundaries of the objects displayed above.

### <font style="color:rgb(8,133,37)">Function Syntax </font>

```python
contours, hierarchy	=	cv.findContours(image, mode, method[, contours[, hierarchy[, offset]]])
```

Where,


- **`image`**	- input image (8-bit single-channel). Non-zero pixels are treated as 1's. Zero pixels remain 0's, so the image is treated as binary . You can use compare, inRange, threshold , adaptiveThreshold, Canny, and others to create a binary image out of a grayscale or color one. 
- **`contours`** - Detected contours. Each contour is stored as a vector of points.
- **`hierarchy`** - Optional output vector containing information about the image topology. It has been described in detail in the video above.
- **`mode`** - Contour retrieval mode, ( RETR_EXTERNAL, RETR_LIST, RETR_CCOMP, RETR_TREE )
- **`method`** - Contour approximation method. ( CHAIN_APPROX_NONE, CHAIN_APPROX_SIMPLE, CHAIN_APPROX_TC89_L1 etc )
- **`offset`** - Optional offset by which every contour point is shifted. This is useful if the contours are extracted from the image ROI and then they should be analyzed in the whole image context.

```
contours, hierarchy = cv2.findContours(imageGray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(image, contours, -1, (0,255,0), 3);

# center of mass
for cnt in contours:
    # We will use the contour moments
    # to find the centroid
    M = cv2.moments(cnt)
    x = int(round(M["m10"]/M["m00"]))
    y = int(round(M["m01"]/M["m00"]))
    
    # Mark the center
    cv2.circle(image, (x,y), 10, (255,0,0), -1);

# Area and Perimeter
for index,cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    print("Contour #{} has area = {} and perimeter = {}".format(index+1,area,perimeter))

# Bounding Box
There are 2 type of bounding boxes we can create around a contour:

1. A vertical rectangle
2. A rotated rectangle - This is the bounding box with the minimum area

image = imageCopy.copy()
for cnt in contours:
    # Vertical rectangle
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,255), 2)

image = imageCopy.copy()
for cnt in contours:
    # Rotated bounding box
    box = cv2.minAreaRect(cnt)
    boxPts = np.int0(cv2.boxPoints(box))
    # Use drawContours function to draw 
    # rotated bounding box
    cv2.drawContours(image, [boxPts], -1, (0,255,255), 2)

# Fit a circle or ellipse
image = imageCopy.copy()
for cnt in contours:
    # Fit a circle
    ((x,y),radius) = cv2.minEnclosingCircle(cnt)
    cv2.circle(image, (int(x),int(y)), int(round(radius)), (125,125,125), 2)

image = imageCopy.copy()
for cnt in contours:
    # Fit an ellipse
    # We can fit an ellipse only
    # when our contour has minimum
    # 5 points
    if len(cnt) < 5:
        continue
    ellipse = cv2.fitEllipse(cnt)
    cv2.ellipse(image, ellipse, (255,0,125), 2)
```


## <font style="color:rgb(50,120,229)">What is a Blob?</font>

<img src="https://www.learnopencv.com/wp-content/uploads/2015/02/blob_detection.jpg">

A Blob is a group of connected pixels in an image that share some common property ( E.g grayscale value ). In the image above, the dark connected regions are blobs, and the goal of blob detection is to identify and mark these regions.

## <font style="color:rgb(50,120,229)">Simple Blob Detector Example</font>
OpenCV provides a convenient way to detect blobs and filter them based on different characteristics. Let’s start with the simplest example.

# <font style="color:rgb(50,120,229)">How does Blob detection work ?</font>

SimpleBlobDetector, as the name implies, is based on a rather simple algorithm described below. The algorithm is controlled by parameters ( shown in bold below )  and has the following steps. 

1. **Thresholding** : Convert the source images to several binary images by thresholding the source image with thresholds starting at minThreshold. These thresholds are incremented  by thresholdStep until maxThreshold. So the first threshold is minThreshold, the second is minThreshold + thresholdStep, the third is minThreshold + 2 x thresholdStep, and so on.
2. **Grouping** : In each binary image,  connected white pixels are grouped together.  Let’s call these binary blobs.
3. **Merging**  : The centers of the binary blobs in the binary images are computed, and  blobs located closer than minDistBetweenBlobs are merged.
4. **Center & Radius Calculation** :  The centers and radii of the new merged blobs are computed and returned.

```
detector = cv2.SimpleBlobDetector_create()
keypoints = detector.detect(im)
```