## <font style="color:rgb(50,120,229)">HoughLine: How to Detect Lines using OpenCV</font>

In OpenCV, line detection using Hough Transform is implemented in the function [**`HoughLines`**](https://docs.opencv.org/4.1.0/dd/d1a/group__imgproc__feature.html#ga46b4e588934f6c8dfd509cc6e0e4545a) and [**`HoughLinesP`**](https://docs.opencv.org/4.1.0/dd/d1a/group__imgproc__feature.html#ga8618180a5948286384e3b7ca02f6feeb) [Probabilistic Hough Transform].

### <font style="color:rgb(8,133,37)">Function Syntax</font>

```python
lines = cv.HoughLinesP(image, rho, theta, threshold[, lines[, minLineLength[, maxLineGap]]])
```

```python
lines = cv.HoughLines(image, rho, theta, threshold[, lines[, srn[, stn[, min_theta[, max_theta]]]]])
```

**Parameters**


- **`image`** -	8-bit, single-channel binary source image. The image may be modified by the function.
- **`lines`** -	Output vector of lines. Each line is represented by a 2 or 3 element vector $(\rho,\theta)$ or $(\rho,\theta,\text{votes})$ . $\rho$ is the distance from the coordinate origin $(0,0)$ (top-left corner of the image). $\theta$ is the line rotation angle in radians $( 0∼\text{vertical line},\pi/2∼\text{horizontal line} )$. votes is the value of accumulator.
- **`rho`** -	Distance resolution of the accumulator in pixels.
- **`theta`** -	Angle resolution of the accumulator in radians.
- **`threshold`** -	Accumulator threshold parameter. Only those lines are returned that get enough votes $( >\text{threshold} )$.
- **`srn`** -	For the multi-scale Hough transform, it is a divisor for the distance resolution rho . The coarse accumulator distance resolution is rho and the accurate accumulator resolution is rho/srn . If both srn=0 and stn=0 , the classical Hough transform is used. Otherwise, both these parameters should be positive.
- **`stn`** -	For the multi-scale Hough transform, it is a divisor for the distance resolution theta.
- **`min_theta`** -	For standard and multi-scale Hough transform, minimum angle to check for lines. Must fall between 0 and max_theta.
- **`max_theta`** -	For standard and multi-scale Hough transform, maximum angle to check for lines. Must fall between min_theta and CV_PI.

## <font style="color:rgb(50,120,229)">HoughCircles : Detect circles in an image with OpenCV</font>

In the case of line Hough transform, we required two parameters, $(\theta, \rho)$ but to detect circles, we require three parameters:

- $(x, y)$ coordinates of the center of the circle.
- radius.

As you can imagine, a circle detector will require a **3D accumulator** — one for each parameter.

The equation of a circle is given by

$\begin{align*} (x - x_0)^2 + (y - y_0)^2 = r^2 \end{align*} \tag{2}$

The following steps are followed to detect circles in an image: –

1. Find the edges in the given image with the help of edge detectors (**Canny**).
2. For detecting circles in an image, we set a threshold for the maximum and minimum value of the radius.
3. Evidence is collected in a 3D accumulator array for the presence of circles with different centers and radii.

The function [**`HoughCircles`**](https://docs.opencv.org/4.1.0/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d) is used in OpenCV to detect the circles in an image.

### <font style="color:rgb(8,133,37)">Function Syntax</font>

```python
circles = cv.HoughCircles( image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]])
```

**Parameters**


- **`image`** -	8-bit, single-channel binary source image. The image may be modified by the function.
- **`circles`** -	Output vector of found circles. Each vector is encoded as 3 or 4 element floating-point vector (x,y,radius) or (x,y,radius,votes) .
- **`method`** -	Detection method. Currently, the only implemented method is **`HOUGH_GRADIENT`**
- **`dp`** -	Inverse ratio of the accumulator resolution to the image resolution. For example, if `dp=1` , the accumulator has the same resolution as the input image. If `dp=2` , the accumulator has half as big width and height.
- **`minDist`** -	Minimum distance between the centers of the detected circles. If the parameter is too small, multiple neighbor circles may be falsely detected in addition to a true one. If it is too large, some circles may be missed.
- **`param1`** -	First method-specific parameter. In case of **`HOUGH_GRADIENT`** , it is the higher threshold of the two passed to the Canny edge detector (the lower one is twice smaller).
- **`param2`** -	Second method-specific parameter. In case of **`HOUGH_GRADIENT`** , it is the accumulator threshold for the circle centers at the detection stage. The smaller it is, the more false circles may be detected. Circles, corresponding to the larger accumulator values, will be returned first.
- **`minRadius`** -	Minimum circle radius.
- **`maxRadius`** -	Maximum circle radius. If `<= 0`, uses the maximum image dimension. If `< 0`, returns centers without finding the radius.


## HDR
## <font style="color:rgb(50,120,229)">What is High Dynamic Range (HDR) imaging? </font>

Most digital cameras and displays capture or display color images as 24-bits matrices. There are 8-bits per color channel and the pixel values are therefore in the range **0 – 255** for each channel. In other words, a regular camera or a display has a **limited dynamic range**.

However, the world around us has a very large dynamic range. It can get pitch black inside a garage when the lights are turned off and it can get really bright if you are looking directly at the Sun. Even without considering those extremes, in everyday situations, 8-bits are barely enough to capture the scene. So, **the camera tries to estimate the lighting and automatically sets the exposure so that the most interesting aspect of the image has good dynamic range, and the parts that are too dark and too bright are clipped off to 0 and 255 respectively**.

In the Figure below, the image on the left is a normally exposed image. Notice the sky in the background is completely washed out because the camera decided to use a setting where the subject (my son) is properly photographed, but the bright sky is washed out. The image on the right is an HDR image produced by the iPhone.


<img src="https://www.learnopencv.com/wp-content/uploads/2017/09/high-dynamic-range-hdr.jpg" width="600" height="600"/>

How does an iPhone capture an HDR image? It actually **takes 3 images at three different exposures**. The images are taken in quick succession so there is almost no movement between the three shots. **The three images are then combined to produce the HDR image**. We will see the details in the next section.

## <font style="color:rgb(50,120,229)">Step 1: Capture multiple images with different exposures </font>

When we take a picture using a camera, we have only 8-bits per channel to represent the dynamic range ( brightness range ) of the scene. But **we can take multiple images of the scene at different exposures by changing the shutter speed**. Most SLR cameras have a feature called Auto Exposure Bracketing (AEB) that allows us to take multiple pictures at different exposures with just one press of a button. If you are using an iPhone, you can use this [AutoBracket HDR app](https://itunes.apple.com/us/app/autobracket-hdr/id923626339?mt=8&ign-mpt=uo%3D8) and if you are an android user you can try [A Better Camera app](https://play.google.com/store/apps/details?id=com.almalence.opencam&hl=en).

Using AEB on a camera or an auto bracketing app on the phone, we can take multiple pictures quickly one after the other so the scene does not change. When we use HDR mode in an iPhone, it takes three pictures.

1. An **underexposed** image: This image is darker than the properly exposed image. The goal is the capture parts of the image that are very bright.
2. A **properly exposed** image: This is the regular image the camera would have taken based on the illumination it has estimated.
3. An **overexposed** image: This image is brighter than the properly exposed image. The goal is the capture parts of the image that very dark and the camera needs more time to capture enough light to see what is there in the dark.

*However, if the dynamic range of the scene is very large, we can take more than three pictures to compose the HDR image.*

We will use 4 images taken with exposure time 1/30, 0.25, 2.5 and 15 seconds. The thumbnails are shown below.

<img src="https://www.learnopencv.com/wp-content/uploads/2017/10/hdr-image-sequence.jpg" />

## <font style="color:rgb(50,120,229)">Step 2: Align Images </font>

Misalignment of images used in composing the HDR image can result in severe artifacts. In the Figure below, **the image on the left is an HDR image composed using unaligned images and the image on the right is one using aligned images**. By zooming into a part of the image, shown using red circles, we see severe ghosting artifacts in the left image.

<img src="https://www.learnopencv.com/wp-content/uploads/2017/10/aligned-unaligned-hdr-comparison.jpg" />

## <font style="color:rgb(50,120,229)">Step 3: Recover the Camera Response Function </font>
    
The response of a typical camera is not linear to scene brightness. What does that mean? Suppose, two objects are photographed by a camera and one of them is twice as bright as the other in the real world. When you measure the pixel intensities of the two objects in the photograph, the pixel values of the brighter object will not be twice that of the darker object! Without estimating the Camera Response Function (CRF), we will not be able to merge the images into one HDR image.

What does it mean to merge multiple exposure images into an HDR image?

Consider just ONE pixel at some location (x,y) of the images. If the CRF was linear, the pixel value would be directly proportional to the exposure time unless the pixel is too dark ( i.e. nearly 0 ) or too bright ( i.e. nearly 255) in a particular image. We can filter out these bad pixels ( too dark or too bright ), and estimate the brightness at a pixel by dividing the pixel value by the exposure time and then averaging this brightness value across all images where the pixel is not bad ( too dark or too bright ). We can do this for all pixels and obtain a single image where all pixels are obtained by averaging “good” pixels.

But the CRF is not linear and we need to make the image intensities linear before we can merge/average them by first estimating the CRF.

The good news is that the CRF can be estimated from the images if we know the exposure times for each image. Like many problems in computer vision, the problem of finding the CRF is set up as an optimization problem where the goal is to minimize an objective function consisting of a data term and a smoothness term. These problems usually reduce to a linear least squares problem which are solved using Singular Value Decomposition (SVD) that is part of all linear algebra packages. The details of the CRF recovery algorithm are in the paper titled [Recovering High Dynamic Range Radiance Maps from Photographs](http://www.pauldebevec.com/Research/HDR/debevec-siggraph97.pdf).

## <font style="color:rgb(50,120,229)">Step 5: Tone mapping </font>
-> 다시 24bit (u8)로 톤매핑.

The process of converting a High Dynamic Range (HDR) image to an 8-bit per channel image while preserving as much detail as possible is called **Tone mapping**.

Now we have merged our exposure images into one HDR image. Can you guess the minimum and maximum pixel values for this image? The minimum value is obviously 0 for a pitch black condition. What is the theoretical maximum value? Infinite! In practice, the maximum value is different for different situations. If the scene contains a very bright light source, we will see a very large maximum value.

Even though we have recovered the relative brightness information using multiple images, we now have the challenge of saving this information as a 24-bit image for display purposes.

There are several tone mapping algorithms. OpenCV implements few of them. **The thing to keep in mind is that there is no right way to do tone mapping**. Usually, we want to see more detail in the tonemapped image than in any one of the exposure images. Sometimes the goal of tone mapping is to produce realistic images and often times the goal is to produce surreal images. The algorithms implemented in OpenCV tend to produce realistic and therefore less dramatic results.

Let’s look at the various options. Some of the common parameters of the different tone mapping algorithms are listed below.

- **`gamma`** : This parameter compresses the dynamic range by applying a gamma correction. When gamma is equal to 1, no correction is applied. A gamma of less than 1 darkens the image, while a gamma greater than 1 brightens the image.
- **`saturation`** : This parameter is used to increase or decrease the amount of saturation. When saturation is high, the colors are richer and more intense. Saturation value closer to zero, makes the colors fade away to grayscale.
- **`contrast`** : Controls the contrast ( i.e. log (maxPixelValue/minPixelValue) ) of the output image.
Let us explore the four tone mapping algorithms available in OpenCV.



## <font style = "color:rgb(50,120,229)">Seamless Cloning</font>

| <img src="https://www.learnopencv.com/wp-content/uploads/2015/01/seamless-cloning-example.jpg" alt=" Seamless Cloning Example" width="600" height="600"/> |
| --- |
| <center>Figure 1 : Seamless Cloning Example : An airplane cloned into the picture of an evening sky.</center> |

We had seen in a previous section, how you can use arithmetic and bitwise operations to blend one image on another. It is difficult to get good results when there is a huge color difference between the two images. There are more advanced methods which help perform this kind of blending which can be useful in many applications which require you to blend images and make them look natural and real.

The above image was created using a scene of a sky and that of an airplane.  If I had simply overlaid the airplane image on top of the sky image, the result would look ridiculous (See Figure 2).

| <img src="https://www.learnopencv.com/wp-content/uploads/2015/02/sky-with-plane.jpg" alt="Sky with plane overlaid." width="600" height="600"/> |
| --- |
| <center>Figure 2 : Sky with plane overlaid.</center> |

Now of course nobody in their right mind would do a composition like that. You would obviously mask the image out carefully, and perhaps after spending half a day in Photoshop get an image that looks like Figure 3.

| <img src="https://www.learnopencv.com/wp-content/uploads/2015/02/sky-with-plane-masked.jpg" alt="Sky with plane overlaid." width="600" height="600"/> |
| --- |
| <center>Figure 3 : Sky image with airplane overlaid with careful masking.</center> |

    
### <font style = "color:rgb(8,133,37)">Function Syntax</font>
    
A quick look at the usage first

```python
output = cv2.seamlessClone(src, dst, mask, center, flags)
```

Where,

- **`src`** - Source image that will be cloned into the destination image. In our example it is the airplane.
- **`dst`** - Destination image into which the source image will be cloned. In our example it is the sky image.
- **`mask`** - A rough mask around the object you want to clone. This should be the size of the source image. Set it to an all white image if you are lazy!
- **`center`** - Location of the center of the source image in the destination image.
- **`flags`** - The two flags that currently work are NORMAL_CLONE and MIXED_CLONE. I have included an example to show the difference.
- **`output`** - Output / result image.

```
src_mask = np.zeros(src.shape, src.dtype)
poly = np.array([ [4,80], [30,54], [151,63], [254,37], [298,90], [272,134], [43,122] ], np.int32)
src_mask = cv2.fillPoly(src_mask, [poly], (255, 255, 255))

output = cv2.seamlessClone(src, dst, src_mask, center, cv2.NORMAL_CLONE)
```

## <font style = "color:rgb(50,120,229)">Normal Cloning ( NORMAL_CLONE ) versus Mixed Cloning ( MIXED_CLONE )</font>
    
1. If we use Normal Cloning by using the `NORMAL_CLONE` flag, we did not use a good mask and you can see excessive smoothing between the words “I” and “Love”, and between “you” and “Paa”. Sure we were lazy. We could have created a rough mask and improved the result.  But if you are lazy and smart, you would use Mixed Cloning.
1. **In Normal Cloning the texture ( gradient ) of the source image is preserved in the cloned region.**
1. In Mixed Cloning, the texture ( gradient ) of the cloned region is determined by a combination of the source and the destination images. 
1. **Mixed Cloning does not produce smooth regions because it picks the dominant texture ( gradient ) between the source and destination images**. Notice the texture for mixed cloning is no longer smooth between “I” and “Love”, and “you” and “Paa”. Lazy people cheer!

## <font style = "color:rgb(50,120,229)">Inpainting Code in Python</font>

### <font style = "color:rgb(8,133,37)">Function Syntax</font>
    
In OpenCV inpainting is implemented using the function `inpaint`.

```python
dst = cv2.inpaint(
             src, 
             inpaintMask, 
             inpaintRadius, 
             flags)
```

Where,

- **`src`** = Source image
- **`inpaintMask`** = A binary mask indicating pixels to be inpainted.
- **`dst`** = Destination image
- **`inpaintRadius`** = Neighborhood around a pixel to inpaint. Typically, if the regions to be inpainted are thin, smaller values produce better results (less blurry).
- **`flags`** : `INPAINT_NS` (Navier-Stokes based method) or `INPAINT_TELEA` (Fast marching based method)