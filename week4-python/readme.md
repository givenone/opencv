## <font style="color:rgb(50,120,229)">The HSV color space</font>

This is one of the most popular color spaces used in image processing after the RGB color space. Its three components are :

- **Hue** - indicates the color / tint of the pixel

- **Saturation** - indicates the purity (or richness) of the color

- **Value** - indicates the amount of brightness of the pixel

The HSV color space converts the RGB color space from cartesian coordinates (x, y, z) to cylindrical coordinates (ρ, φ, z). It is **more intuitive than the RGB color space** because it separates the color and brightness into different axes. This makes it easier for us to describe any color directly.

Let us first see how to convert an image from BGR to HSV format and display the different channels to get some more insights about them. We will use OpenCV’s [**`cvtColor`**](https://docs.opencv.org/4.1.0/d8/d01/group__imgproc__color__conversions.html#ga397ae87e1288a81d2363b61574eb8cab) function for conversion.

### <font style="color:rgb(8,133,37)">Function Syntax</font>

```python
dst	=	cv2.cvtColor(	src, code[, dst[, dstCn]]	)
```

**Parameters**

- **`src`** - input image: 8-bit unsigned, 16-bit unsigned ( CV_16UC... ), or single-precision floating-point.
- **`dst`** - output image of the same size and depth as src.
- **`code`** - color space conversion code (see ColorConversionCodes).
- **`dstCn`** - number of channels in the destination image; if the parameter is 0, the number of the channels is derived automatically from src and code.


## <font style="color:rgb(50,120,229)">Image Histogram </font>
An image histogram is similar to what we discussed above. In case of image histogram,
- the x-axis represents the different intensity values or range of intensity values ( also called **bins** ), which lie between 0 and 255, and 
- the y-axis represents the **number of times a particular intensity value occurs in the image**.

<table style="width:100%">
  <tr>
    <th style = "width:55%">   
         marks = [9, 7, 10, 15, 18,
         21, 33, 35, 30, 31,
         41, 46, 43, 41, 41,
         42, 45, 41, 49, 45,
         50, 51, 53, 53, 56,
         60, 64, 62, 65, 61, 
         65, 71, 72, 74, 71,
         70, 71, 75, 72, 73,
         72, 70, 83, 85, 87,
         85, 82, 97, 100, 99 ]
         </th>
    <th><center> <img src = "https://www.learnopencv.com/wp-content/uploads/2017/12/opcv4face-w2-m1-histogramScore.png"/></center></th> 
  </tr>
  <tr>
      <td><center>Array containing scores of 50 students</center></td>
      <td><center>Histogram plot of scores obtained</center></td>
  </tr>
</table>

## <font style="color:rgb(50,120,229)">Histograms using Matplotlib</font>
We will use the function [**`plt.hist()`**](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.hist.html) available in the matplotlib library for drawing the histogram.

### <font style="color:rgb(8,133,37)">Function Syntax </font>
```python
hist, bins, patches	=	plt.hist( x, bins=None, range=None, density=None, weights=None, cumulative=False, bottom=None, histtype='bar', align='mid', orientation='vertical', rwidth=None, log=False, color=None, label=None, stacked=False, normed=None )
```

**Parameters**

There are many parameters in the function. Let us go over the most important and frequently used ones.

**Input**
- **`x`** - source image as an array
- **`bins`** - number of bins
- **`color`** - color for plotting the histogram

**Output**
- **`hist`** - histogram array
- **`bins`** - edges of bins

#### <font style="color:rgb(200,0,0)">NOTE </font>
The input to the function is an array ( not an image matrix ). Thus, we need to flatten the matrix into an array before passing it to the function.

## <font style="color:rgb(50,120,229)">Histogram Equalization</font>

Histogram Equalization is a non-linear method for enhancing contrast in an image. We have already seen the theory in the video. Now, let's see how to perform histogram equalization using OpenCV using [**`equalizeHist()`**](https://docs.opencv.org/4.1.0/d6/dc7/group__imgproc__hist.html#ga7e54091f0c937d49bf84152a16f76d6e). 

## <font style="color:rgb(50,120,229)">Histogram Equalization for Grayscale Images</font>

The function [**`equalizeHist()`**](https://docs.opencv.org/4.1.0/d6/dc7/group__imgproc__hist.html#ga7e54091f0c937d49bf84152a16f76d6e) performs histogram equalization on a grayscale image. The syntax is given below.


### <font style="color:rgb(8,133,37)">Function Syntax</font>

```python
	dst	=	cv2.equalizeHist(	src[, dst]	)
```

**Parameters**

- **`src`** -	Source 8-bit single channel image.
- **`dst`** -	Destination image of the same size and type as src .

### <font style="color:rgb(8,133,37)">Right Way</font>

We just saw that histogram equalization performed on the three channels separately leads to a bad results. The reason is that when each color channel is non-linearly transformed independently, you can get completely new and unrelated colors. 

The right way to perform histogram equalization on color images is to transform the images to a space like the **HSV** colorspace where colors/hue/tint is separated from the intensity. 

These are the steps involved

1. Tranform the image to HSV colorspace.
2. Perform histogram equalization only on the V channel. 
3. Transform the image back to RGB colorspace.

## <font style="color:rgb(50,120,229)">Contrast Limited Adaptive Histogram Equalization (CLAHE) </font>

Histogram equalization uses the pixels of the entire image to improve contrast. While this may look good in many cases, sometimes we may want to enhance the contrast locally so the image does not looks more natural and less dramatic. 

For such applications, we use [Contrast Limited Adaptive Histogram Equalization (CLAHE)](https://docs.opencv.org/4.1.0/d5/daf/tutorial_py_histogram_equalization.html) ) which improves the local contrast. More importantly it allows us to specify the size of the neighborhood that is considered "local".  

Let's pick a different image, where we may prefer CLAHE in place of regular histogram equalization. 

```
In this, image is divided into small blocks called "tiles" (tileSize is 8x8 by default in OpenCV). Then each of these blocks are histogram equalized as usual. So in a small area, histogram would confine to a small region (unless there is noise). If noise is there, it will be amplified. To avoid this, contrast limiting is applied. If any histogram bin is above the specified contrast limit (by default 40 in OpenCV), those pixels are clipped and distributed uniformly to other bins before applying histogram equalization. After equalization, to remove artifacts in tile borders, bilinear interpolation is applied.
```


## <font style = "color:rgb(50,120,229)">Color Tone Adjustment using Curves</font>

In image enhancement, manipulating color channels using curves is a very important technique. It maps the original colors and brightness of an image to values specified by the user. This is one of the mostly used feature in Photoshop. But Photoshop is manual work, why not write a piece of code for doing that? We will see how we can use this simple yet powerful way of adjusting colors by adding warming and cooling effects to an image.

Consider the curves for the warming filter shown below. The red curve is applied to the Red channel and the Blue curve to the blue channel of the original image. The dotted black line with a slope of 1, is the original mapping. We take some points on the black line and move them. The rest of the points between them are interpolated. If we move the points above the black line as in the case of the Red Channel, the resulting intensity values of red channel increase as given by the y-axis values. Similarly the values decrease for the Blue Channel. 

<center><img src = "https://www.learnopencv.com/wp-content/uploads/2017/12/opcv4face-w2-m2-warmingEffectResult.jpg"/></center>

<center>Figure : Curves for Warming filter. X-axis - Original Values of intensity, Y-axis - Modified values.</center>

dst	=	cv.LUT(	src, lut[, dst]	)

Performs a look-up table transform of an array.

The function LUT fills the output array with values from the look-up table. Indices of the entries are taken from the input array. That is, the function processes each element of src as follows

```
originalValue = np.array([0, 50, 100, 150, 200, 255])

# Changed points on Y-axis for each channel
rCurve = np.array([0, 80, 150, 190, 220, 255])
bCurve = np.array([0, 20,  40,  75, 150, 255])

# Create a LookUp Table
fullRange = np.arange(0,256)
rLUT = np.interp(fullRange, originalValue, rCurve )
bLUT = np.interp(fullRange, originalValue, bCurve )

bChannel = img[:,:,0]
bChannel = cv2.LUT(bChannel, bLUT)
img[:,:,0] = bChannel

# Get the red channel and apply the mapping
rChannel = img[:,:,2]
rChannel = cv2.LUT(rChannel, rLUT)
img[:,:,2] = rChannel

# show and save the ouput
combined = np.hstack([original,img])
```

| <center>**Curve**</center> | <center>**Result Image**</center> | <center>**Effects**</center> |
| :--------: | :--------: | -------- |
| <center><img src = "https://www.learnopencv.com/wp-content/uploads/2017/12/opcv4face-w2-m2-result1graph.jpg" width=330 /></center>    | <center> <img src = "https://www.learnopencv.com/wp-content/uploads/2017/12/opcv4face-w2-m2-result1.jpg" width = 330/></center>     | <div align='left'><ul><li>Keeps the bright areas same</li><li>Makes the dark areas brighter</li><li>Decreases contrast</li><li>Increases brightness</li></ul> </div>|
|<center><img src = "http://www.learnopencv.com/wp-content/uploads/2017/12/opcv4face-w2-m2-result2graph.jpg" width=330 /></center>|<center><img src = "https://www.learnopencv.com/wp-content/uploads/2017/12/opcv4face-w2-m2-result2.jpg" width=330 /></center>| <div align='left'><ul><li>Keeps dark areas same</li><li>Makes bright areas brighter</li><li>Increases contrast</li></ul> </div>|
|<center><img src = "https://www.learnopencv.com/wp-content/uploads/2017/12/opcv4face-w2-m2-result3Graph.jpg" width=330 /></center>|<center><img src = "https://www.learnopencv.com/wp-content/uploads/2017/12/opcv4face-w2-m2-result3.jpg" width=330 /></center>|<div align='left'><ul><li>Keeps dark areas same</li><li>Makes bright areas darker</li><li>Decreases the contrast and brightness</li></ul> </div> |
|<center><img src = "https://www.learnopencv.com/wp-content/uploads/2017/12/opcv4face-w2-m2-result4graph.jpg" width=330 /></center>|<center><img src = "https://www.learnopencv.com/wp-content/uploads/2017/12/opcv4face-w2-m2-result4.jpg" width=330 /></center>|<div align='left'><ul><li>Keeps bright areas same</li><li>Makes dark areas darker</li><li>Intensifies shadows</li></ul> </div>|
|<center><img src = "https://www.learnopencv.com/wp-content/uploads/2017/12/opcv4face-w2-m2-result5-graph.jpg" width=330 /></center>|<center><img src = "https://www.learnopencv.com/wp-content/uploads/2017/12/opcv4face-w2-m2-result5.jpg" width=330 /></center>|<div align='left'><ul><li>Makes dark areas darker and bright areas brighter</li><li>Intensifies shadows and highlights</li><li>Most commonly used</li></ul> </div>|



## <font style = "color:rgb(50,120,229)">Convolution in OpenCV</font>

In OpenCV, convolution is performed using the function [**`filter2D`**](https://docs.opencv.org/4.1.0/d4/d86/group__imgproc__filter.html#ga27c049795ce870216ddfb366086b5a04). The basic usage is given below.

### <font style = "color:rgb(8,133,37)">Function Syntax</font>

```python
dst	=	cv.filter2D(	src, ddepth, kernel[, dst[, anchor[, delta[, borderType]]]]
```

**Parameters**
- **`src`**	input image.
- **`dst`**	output image of the same size and the same number of channels as src.
- **`ddepth`**	desired depth of the destination image.
- **`kernel`**	convolution kernel (or rather a correlation kernel), a single-channel floating point matrix; if you want to apply different kernels to different channels, split the image into separate color planes using split and process them individually.
- **`anchor`**	anchor of the kernel that indicates the relative position of a filtered point within the kernel; the anchor should lie within the kernel; default value (-1,-1) means that the anchor is at the kernel center.
- **`delta`**	optional value added to the filtered pixels before storing them in dst.
- **`borderType`**	pixel extrapolation method.


```python
dst	=	cv2.GaussianBlur(	src, ksize, sigmaX[, dst[, sigmaY[, borderType]]]	)
```

**Parameters**

- **`src`**	input image; the image can have any number of channels, which are processed independently, but the depth should be CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
- **`dst`**	output image of the same size and type as src.
- **`ksize`**	Gaussian kernel size. ksize.width and ksize.height can differ but they both must be positive and odd. Or, they can be zero's and then they are computed from sigma.
- **`sigmaX`**	Gaussian kernel standard deviation in X direction.
- **`sigmaY`**	Gaussian kernel standard deviation in Y direction; if sigmaY is zero, it is set to be equal to sigmaX, if both sigmas are zeros, they are computed from ksize.width and ksize.height, respectively; to fully control the result regardless of possible future modifications of all this semantics, it is recommended to specify all of ksize, sigmaX, and sigmaY.
- **`borderType`**	pixel extrapolation method.


```python
dst	=	cv2.medianBlur(	src, ksize[, dst]	)
```

**Parameters**
- **`src`**	input 1-, 3-, or 4-channel image; when ksize is 3 or 5, the image depth should be CV_8U, CV_16U, or CV_32F, for larger aperture sizes, it can only be CV_8U.
- **`dst`**	destination array of the same size and type as src.
- **`ksize`**	aperture linear size; it must be odd and greater than 1, for example: 3, 5,7, ...

Time to look at some code in action!


## <font style="color:rgb(50,120,229)">Bilateral Filtering</font>

A **Bilateral Filter** is nonlinear, edge-preserving and noise-reducing smoothing filter. Most smoothing filters (e.g. a Gaussian or a Box filter) have a parameter called $\sigma_s$ (the s in the subscript stands for "spatial") that determines the amount of smoothing. Often this value is closely related to the kernel size. A typical smoothing filter replaces the intensity value of a pixel by the weighted sum of its neighbors. The bigger the neighborhood, the smoother the filtered image looks. The size of the neighborhood is directly proportional to the parameter $\sigma_s$.

In edge-preserving filters, there are two competing objectives :

1. Smooth the image.

2. Don’t smooth the edges / color boundaries. 

In other words, if we want to preserve edges, we cannot simply replace the color of a pixel by the weighted sum of its neighbors. 

Consider this 3x3 image patch, shown below.

<center><img src="https://www.dropbox.com/s/l4ugj52l5phid0y/bilateralMatrix1.png?dl=1" /> </center>

You can see the values in the left column are much lower than the values in the center and the right columns. In other words, this patch is part of a vertical edge. In a typical filtering operation, we replace the center pixel by some weighted combination of all 9 pixels in this 3x3 neighborhood. However, in this case, a better idea is to filter the center pixel based on only the center and right-hand side columns so that the edge is retained and not blurred-out. 

In bilateral filtering, while calculating the contribution of any pixel to the final output, we weigh the pixels that are close in terms of intensity to the center pixel higher as compared to the pixels whose intensities are very different from the center pixels. We want to find a weight that depends on the square of the intensity difference  ($I_p$ − $I_q$)$^2$  between the center pixel q and its neighbor p. But if you are a control freak, like most AI scientists, you will want more control over your definition of intensity difference. We can redefine intensity difference as the Gaussian function 

<center><img src="https://www.dropbox.com/s/wwmneobohkaphst/bilateralEquation2.png?dl=1" /></center>

and control the meaning of intensity differences using the parameter $\sigma_r$.
 
Additionally, just like Gaussian filtering, we also want to weight the pixels that are closer to the center pixel higher than the pixels that are farther away. So, the weights should depend on $|| p − q ||$. But again, you are likely a control freak and want to control the definition of distance. How do you do that? Well, you use a Gaussian $G_{\sigma_{s}} (|| p − q ||)$ and control the meaning of distance using the parameter $\sigma_s$. 

Combining the two, a bilateral filter will output the following at center pixel q. 

<center><img src="https://www.dropbox.com/s/f3egifbpk72ga7l/bilateralEquation1.png?dl=1" /> </center>

Where, 

$W_p$ = The normalization constant

$G_{\sigma_{s}}$= Spatial Gaussian kernel

$G_{\sigma_{r}}$ = Color / Range Gaussian kernel

q = Center pixel

p = Neighboring pixel

$I_p$  = Intensity at pixel p

$I_q$ = Intensity at pixel q



If the neighborhood pixels are edges, the difference in intensity $(I_p - I_q)$ will be higher. Since the Gaussian is a decreasing function, $G_{\sigma_{r}}(I_p - I_q)$ will have lower weights for higher values. Hence, the smoothing effect will be lower for such pixels, preserving the edges. 

To conclude, for bilateral filtering, we have two parameters : $\sigma_s$ and $\sigma_r$. Just like other smoothing filters $\sigma_s$ controls amount of spatial smoothing, and $\sigma_r$ (for sigma_range) controls how dissimilar colors within the neighborhood will be averaged. A higher $\sigma_r$ results in larger regions of constant color. Let’s have a look at the code. 

First, here is the [**`Bilateral filter`**](https://docs.opencv.org/4.1.0/d4/d86/group__imgproc__filter.html#ga9d7064d478c95d60003cf839430737ed) in OpenCV. 

### <font style="color:rgb(8,133,37)">Function Syntax</font>

```python
dst	=	cv2.bilateralFilter(	src, d, sigmaColor, sigmaSpace[, dst[, borderType]]	)
```

**Parameters**

- **`src`**	Source 8-bit or floating-point, 1-channel or 3-channel image.
- **`dst`**	Destination image of the same size and type as src .
- **`d`**	Diameter of each pixel neighborhood that is used during filtering. If it is non-positive, it is computed from sigmaSpace.
- **`sigmaColor`**	Filter sigma in the color space. A larger value of the parameter means that farther colors within the pixel neighborhood (see sigmaSpace) will be mixed together, resulting in larger areas of semi-equal color.
- **`sigmaSpace`**	Filter sigma in the coordinate space. A larger value of the parameter means that farther pixels will influence each other as long as their colors are close enough (see sigmaColor ). When d>0, it specifies the neighborhood size regardless of sigmaSpace. Otherwise, d is proportional to sigmaSpace.
- **`borderType`**	border mode used to extrapolate pixels outside of the image, see BorderTypes

The tutorial below shows an example usage and results for a bilateral filter.


## Sobel Filter

OpenCV provides a [**`Sobel`**](https://docs.opencv.org/4.1.0/d4/d86/group__imgproc__filter.html#gacea54f142e81b6758cb6f375ce782c8d) function for calculating the X and Y Gradients. Below, you can see the most common usage.

### <font style="color:rgb(8,133,37)">Function Syntax</font>

```python
dst	=	cv2.Sobel(	src, ddepth, dx, dy[, dst[, ksize[, scale[, delta[, borderType]]]]]	)
```

**Parameters**

- **`src`**	input image.
- **`dst`**	output image of the same size and the same number of channels as src .
- **`ddepth`**	output image depth,in the case of 8-bit input images it will result in truncated derivatives.
- **`dx`**	order of the derivative x.
- **`dy`**	order of the derivative y.
- **`ksize`**	size of the extended Sobel kernel; it must be 1, 3, 5, or 7.
- **`scale`**	optional scale factor for the computed derivative values; by default, no scaling is applied.
- **`delta`**	optional delta value that is added to the results prior to storing them in dst.
- **`borderType`**	pixel extrapolation method.

Let us go over the following tutorial and see the code in action.

```
# Apply sobel filter along x direction
sobelx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
# Apply sobel filter along y direction
sobely = cv2.Sobel(image,cv2.CV_32F,0,1)
```

# <font style="color:rgb(50,120,229)">Second Order Derivative Filters</font>

We know that the Sobel operator calculates the first derivative in the x- and y-directions. When the intensity changes abruptly, the Sobel filter response fires up, so the response of the Sobel filter can be used as an edge detector. 

However, first derivative operators (like the Sobel operators) are not the only way to detect edges. Second derivative operators have a zero crossing at the location of the edges.

![Second Order Derivative Filters](https://www.learnopencv.com/wp-content/uploads/2017/12/opcv4face-w2-m3.2-secondOrderDerivative.png)

In the image above, we see the effects of first and second derivatives in the 1-d case. A 1-d edge is shown in black. Notice, the first derivative curve, shown in green, has a maximum at the location of the edge. The second derivative, shown in blue, has a zero crossing at the location of the edge. This fact is used to recognize edges.

## <font style="color:rgb(50,120,229)">Laplacian Filter</font>

The [Laplacian](http://docs.opencv.org/3.0-beta/modules/imgproc/doc/filtering.html#laplacian) is a filter that is based on the second derivative. 

Mathematically, the Laplacian operator or filter is given by

$$
L(x,y) = \frac{\partial^2 I }{\partial x^2} +  \frac{\partial^2 I }{\partial y^2}
$$

I have to admit, that looks scary. But fortunately, after the mathematicians did their math, they told us the above expression reduces to the simple convolution kernel shown below. 

$$
\begin{bmatrix}
0 & 1 & 0\\
1 & -4 & 1\\
0 & 1 & 0
\end{bmatrix}
$$


The Laplacian filter is very sensitive to noise and therefore it is important to smooth the image before applying it. 

**<font style="color:rgb(255,0,0)">Note:</font>** As a rule of thumb, **summing and averaging operations are less affected by noise, and differencing operations are greatly affected by noise**. Here is a simple example. 

Let us consider two numbers. One is supposed to be 10 and the other is supposed to be 11, but because of a noise, the second number is recorded as 12. Now, the true difference between the numbers is 11 - 10 = 1. But because of noise, the difference is 12  - 10 = 2. So we have made a mistake of 100% while calculating the difference! 

On the other hand, the true sum of the two numbers is 11 + 10 = 21, but because of noise we estimate it to be 12 + 10 = 22. The error is ( 22 - 21 ) / 21 = 4.76%. 

So, while calculating derivatives we have to worry about noise, but while calculating integrals we don’t have to worry about noise. 

##<font style="color:rgb(50,129,229)">Image Sharpening</font>

In sharpening we want to enhance the edges and bring out more of the underlying texture. Although, sharpening can be performed using a single convolution filter, it is easy to understand it in terms an old photo enhancement technique called **unsharp** **masking**. This technique was developed back in the 1930s in Germany. It is fascinating to see the hacks people used back in those days to get a good photo. Wikipedia has the [story](https://en.wikipedia.org/wiki/Unsharp_masking).

Fortunately, it is much easier for us to achieve unsharp masking digitally. 

1. **Step 1**: Blur the image to smooth out texture. The blurred image contains low frequency information of the original image. Let $I$ be the original image and $I_b$ be the blurred image. 

2. **Step 2**: Obtain the high frequency information of the original image by subtracting the blurred image from the original image. 

3. **Step 3**: Now, put back the high frequency information back in the image and control the amount using a parameter. The final sharpened image is therefore, 

    $$I_s = I + \alpha  ( I − I_b) $$


## <font style="color:rgb(50,120,229)">Canny Edge Detection</font>

**Canny edge detection** is the most widely-used edge detector. For many of the applications that require edge detection, Canny edge detection is sufficient. 

Canny edge detection has the following three steps:

1. **Gradient calculations:**  Edges are pixels where intensity changes abruptly. From previous modules, we know that the magnitude of gradient is very high at edge pixels. Therefore, gradient calculation is the first step in Canny edge detection.

2. **Non-maxima suppression:** In the real world, the edges in an image are not sharp. The magnitude of gradient is very high not only at the real edge location, but also in a small neighborhood around it. Ideally, we want an edge to be represented by a single, pixel-thin contour. Simply thresholding the gradient leads to a fat contour that is several pixels thick.
      Fortunately, this problem can be eliminated by selecting the pixel with maximum gradient magnitude in a small neighborhood (say 3x3 neighborhood) of every pixel in the gradient image. The name non-maxima suppression comes from the fact that we eliminate (i.e. set to zero) all gradients except the maximum one in small 3x3 neighborhoods over the entire image.

3. **Hysteresis thresholding:** After non-maxima suppression, we could threshold the gradient image to obtain a new binary image which is black in all places except for pixels where the gradient is very high. This kind of thresholding would naively exclude a lot of edges because, in real world images, edges tend to fade in and out along their length. For example, an edge may be strong in the middle but fade out at the two ends.
      To fix this problem, Canny edge detection uses** **two thresholds. First, a higher threshold is used to select pixels with very high gradients. We say these pixels have a strong edge. Second, a lower threshold is used to obtain new pixels that are potential edge pixels. We say these pixels have a weak edge. A weak edge pixel can be re-classified as a strong edge if one of its neighbor is a strong edge. The weak edges that are not reclassified as strong are dropped from the final edge map. 

**<font style="color:rgb(255,0,0)">Note:</font>** According to Wikipedia, the word [hysteresis](https://en.wikipedia.org/wiki/Hysteresis) means "the dependence of the state of a system on its history." In thresholding, it is the dependence of the state ( edge / no edge ) of a pixel based on its neighbor. 

#### <font style="color:rgb(8,133,37)">Function Syntax</font>

```python
edges	=	cv.Canny(	dx, dy, threshold1, threshold2[, edges[, L2gradient]]	)
```

**Parameters**
- **`dx`**	16-bit x derivative of input image (CV_16SC1 or CV_16SC3).
- **`dy`**	16-bit y derivative of input image (same type as dx).
- **`edges`**	output edge map; single channels 8-bit image, which has the same size as image .
- **`threshold1`**	first threshold for the hysteresis procedure.
- **`threshold2`**	second threshold for the hysteresis procedure.
- **`L2gradient`**	a flag, indicating whether a more accurate L2 norm =√(dI/dx)2+(dI/dy)2 should be used to calculate the image gradient magnitude ( L2gradient=true ), or whether the default L1 norm =|dI/dx|+|dI/dy| is enough ( L2gradient=false ).

**If you want better accuracy at the expense of speed, you can set the L2gradient flag to true.**