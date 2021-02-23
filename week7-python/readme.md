## <font style="color:rgb(50,120,229)">Affine Transform</font>

In OpenCV, an Affine transform is stored in a 2 x 3 sized matrix. The first two columns of this matrix encode rotation, scale and shear, and the last column encodes translation ( i.e. shift ).

$$
A = \begin{bmatrix} a & b & t_x \\ c & d & t_y \end{bmatrix}
$$

Given a point (x, y) in the original image, the above affine transform, moves it to point (${x_t}$, ${y_t}$) using the equation given below

$$
\begin{bmatrix} x_t \\ y_t \end{bmatrix} = \begin{bmatrix} a & b \\ c & d \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} + \begin{bmatrix} t_x \\ t_y \end{bmatrix}
$$


Translation, Rotation and Euclidean transforms are special cases of the Affine transform. 

In Translation, the rotation and shear parameters are zero and scale is 1, the translation parameters are non-zero. 

In a Euclidean transform the rotation and translation parameters are non-zero, with scale 1 and shear 0. 

In OpenCV, if you want to apply an affine transform to the entire image you can use the function [**`warpAffine`**](https://docs.opencv.org/4.1.0/da/d54/group__imgproc__transform.html#ga0203d9ee5fcd28d40dbc4a1ea4451983).

```python
dst	=	cv2.warpAffine(	src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]]	)
```

**Parameters**

- **`src`**	input image.
- **`dst`**	output image that has the size dsize and the same type as src .
- **`M`**	2×3 transformation matrix.
- **`dsize`**	size of the output image.
- **`flags`**	combination of interpolation methods (see InterpolationFlags) and the optional flag WARP_INVERSE_MAP that means that M is the inverse transformation ( dst→src ).
- **`borderMode`**	pixel extrapolation method (see BorderTypes); when borderMode=BORDER_TRANSPARENT, it means that the pixels in the destination image corresponding to the "outliers" in the source image are not modified by the function.
- **`borderValue`**	value used in case of a constant border; by default, it is 0.

```
srcPoints = np.float32([[50, 50],[50, 149],[149, 50]])
dstPoints = np.float32([[68, 45],[76, 155],[176, 27]])
estimatedMat = cv2.estimateAffine2D(srcPoints, dstPoints)[0]
```

## Homography Transformation

## <font style="color:rgb(50,120,229)">What is Homography ?</font>

Consider two images of a plane (top of the book) shown in Figure below. The red dot represents the same physical point in the two images. In computer vision jargon we call these corresponding points. The figure shows four corresponding points in four different colors — red, green, yellow and orange. 

![Homography Example Annotated](https://www.learnopencv.com/wp-content/uploads/2016/01/homography-example-768x511.jpg)

A **Homography** is a transformation ( a 3×3 matrix ) that maps the points in one image to the corresponding points in the other image.

Now since a homography is a 3×3 matrix we can write it as

$$
H = \begin{bmatrix}h_{00} & h_{01} & h_{02}\\h_{10} & h_{11} & h_{12}\\h_{20} & h_{21} & h_{22}\end{bmatrix}
$$

Let us consider the first set of corresponding points — $(x_1, y_1)$ in the first image and $(x_2, y_2)$ in the second image. Then, the Homography H maps them in the following way
$$
\begin{bmatrix}x_1 \\ y_1 \\ 1\end{bmatrix} = H\begin{bmatrix}x_2 \\ y_2 \\ 1\end{bmatrix} = \begin{bmatrix}h_{00} & h_{01} & h_{02}\\h_{10} & h_{11} & h_{12}\\h_{20} & h_{21} & h_{22}\end{bmatrix}\begin{bmatrix}x_2 \\ y_2 \\ 1\end{bmatrix}
$$

```
# Calculate homography
h, status = cv2.findHomography(srcPoints, dstPoints)
print(h)

# Warp source image to destination based on homography
imH = cv2.warpPerspective(im, h, outDim)
```

### <font style="color:rgb(8,133,37)">Function Syntax</span>

**`pts_src`** and **`pts_dst`** are numpy arrays of points in source and destination images. We need at least 4 corresponding points.

```python
h, status = cv2.findHomography(pts_src, pts_dst)
```

The calculated homography can be used to warp the source image to destination. Size is the size (width,height) of **`im_dst`**.

```python
im_dst = cv2.warpPerspective(im_src, h, size)
```



## <font style="color:rgb(50,120,229)">ORB (Oriented FAST and Rotated BRIEF)</font>

ORB is basically a fusion of FAST keypoint detector and BRIEF descriptor with many modifications to enhance the performance. First it use FAST to find keypoints, then apply Harris corner measure to find top N points among them. It also use pyramid to produce multiscale-features. But one problem is that, FAST doesn't compute the orientation. So what about rotation invariance? Authors came up with following modification.

It computes the intensity weighted centroid of the patch with located corner at center. The direction of the vector from this corner point to centroid gives the orientation. To improve the rotation invariance, moments are computed with x and y which should be in a circular region of radius r, where r is the size of the patch.

Now for descriptors, ORB use BRIEF descriptors. But we have already seen that BRIEF performs poorly with rotation. So what ORB does is to "steer" BRIEF according to the orientation of keypoints. For any feature set of $n$ binary tests at location $(x_i,y_i)$, define a $2 \times n$ matrix, $S$ which contains the coordinates of these pixels. Then using the orientation of patch, $\theta$, its rotation matrix is found and rotates the $S$ to get steered(rotated) version $S_{\theta}$.

ORB discretize the angle to increments of $\frac{2\pi}{30}$ (12 degrees), and construct a lookup table of precomputed BRIEF patterns. As long as the keypoint orientation $\theta$ is consistent across views, the correct set of points $S_{\theta}$ will be used to compute its descriptor.

BRIEF has an important property that each bit feature has a large variance and a mean near 0.5. But once it is oriented along keypoint direction, it loses this property and become more distributed. High variance makes a feature more discriminative, since it responds differentially to inputs. Another desirable property is to have the tests uncorrelated, since then each test will contribute to the result. To resolve all these, ORB runs a greedy search among all possible binary tests to find the ones that have both high variance and means close to 0.5, as well as being uncorrelated. The result is called **rBRIEF**.

For descriptor matching, multi-probe LSH which improves on the traditional LSH, is used. The paper says ORB is much faster than SURF and SIFT and ORB descriptor works better than SURF. ORB is a good choice in low-power devices for panorama stitching etc.


## <font style="color:rgb(50,120,229)">ORB in OpenCV</font>

### <font style="color:rgb(8,133,37)">Function Syntax</font>

Let's have a look at the function syntax for [**`cv2.ORB_create()`**](https://docs.opencv.org/4.1.0/db/d95/classcv_1_1ORB.html#aeff0cbe668659b7ca14bb85ff1c4073b) function which is used to create an ORB detector.

```python
retval	=	cv2.ORB_create()
```
The above function has many arguments, but the default ones work pretty well. It creates an object with 500 features points.

The ORB belongs to the [**feature2D class**](https://docs.opencv.org/3.4/d0/d13/classcv_1_1Feature2D.html). It has a few important functions : [**`cv2.Feature2D.detect()`**](https://docs.opencv.org/4.1.0/d0/d13/classcv_1_1Feature2D.html#aa4e9a7082ec61ebc108806704fbd7887), [**`cv2.Feature2D.compute()`**](https://docs.opencv.org/4.1.0/d0/d13/classcv_1_1Feature2D.html#ab3cce8d56f4fc5e1d530b5931e1e8dc0) and [**`cv2.Feature2D.detectAndCompute()`**](https://docs.opencv.org/4.1.0/d0/d13/classcv_1_1Feature2D.html#a8be0d1c20b08eb867184b8d74c15a677) which can be used as `orb.detect()`, `orb.compute()` and `orb.detectAndCompute()` where, `orb = cv2.ORB_create()`.

Let's see the function syntax:



#### <font style="color:rgb(8,133,37)">1. cv2.Feature2D.detect()</font>
```python
keypoints	=	cv2.Feature2D.detect(	image[, mask]	)
```

Where,

- **`image`** - Image.
- **`keypoints`** - The detected keypoints. In the second variant of the method keypoints[i] is a set of keypoints detected in images[i] .
- **`mask`** - Mask specifying where to look for keypoints (optional). It must be a 8-bit integer matrix with non-zero values in the region of interest.


#### <font style="color:rgb(8,133,37)">2. cv2.Feature2D.compute()</font>


```python
keypoints, descriptors	=	cv2.Feature2D.compute(	image, keypoints[, descriptors]	)
```

Where,

- **`image`**- Image.
- **`keypoints`** - Input collection of keypoints. Keypoints for which a descriptor cannot be computed are removed. Sometimes new keypoints can be added, for example: SIFT duplicates keypoint with several dominant orientations (for each orientation).
- **`descriptors`** - Computed descriptors. In the second variant of the method descriptors[i] are descriptors computed for a keypoints[i]. Row j is the keypoints (or keypoints[i]) is the descriptor for keypoint j-th keypoint.



And finally:
#### <font style="color:rgb(8,133,37)">3. cv2.Feature2D.detectAndCompute()</font>

```python
keypoints, descriptors	=	cv2.Feature2D.detectAndCompute(	image, mask[, descriptors[, useProvidedKeypoints]]	)
```

We can also draw the detected keypoints using [**`cv2.drawKeypoints()`**](https://docs.opencv.org/4.1.0/d4/d5d/group__features2d__draw.html#ga5d2bafe8c1c45289bc3403a40fb88920)

#### <font style="color:rgb(8,133,37)">4. cv2.drawKeypoints()</font>
```python
outImage	=	cv2.drawKeypoints(	image, keypoints, outImage[, color[, flags]]	)
```

Where,

- **`image`** - Source image.
- **`keypoints`** - Keypoints from the source image.
- **`outImage`** - Output image. Its content depends on the flags value defining what is drawn in the output image. See possible flags bit values below.
- **`color`** - Color of keypoints.
- **`flags`** - Flags setting drawing features. Possible flags bit values are defined by DrawMatchesFlags. See details above in drawMatches


## Feature Matching

### Brute Force Matcher

Brute-Force matcher is simple. It takes the descriptor of one feature in first set and is matched with all other features in second set using some distance calculation. And the closest one is returned.

For BF matcher, first we have to create the BFMatcher object using [**`cv2.BFMatcher()`**](https://docs.opencv.org/4.1.0/d3/da1/classcv_1_1BFMatcher.html). 

### <font style="color:rgb(8,133,37)">Function Syntax </font>
#### <font style="color:rgb(8,133,37)">1. Create Matcher object </font>

```python
retval	=	cv2.BFMatcher_create(	[, normType[, crossCheck]]	)

or

retval	=	cv2.BFMatcher()
```
It takes two optional params. 
- `normType`. It specifies the distance measurement to be used. By default, it is [`cv2.NORM_L2`](https://docs.opencv.org/4.1.0/d2/de8/group__core__array.html#ggad12cefbcb5291cf958a85b4b67b6149fa7bacbe84d400336a8f26297d8e80e3a2) which is good for SIFT, SURF etc. For binary string based descriptors like ORB, BRIEF, BRISK etc, [`cv2.NORM_HAMMING`](https://docs.opencv.org/4.1.0/d2/de8/group__core__array.html#ggad12cefbcb5291cf958a85b4b67b6149fa4b063afd04aebb8dd07085a1207da727) should be used, which uses Hamming distance as measurement. 

- `crossCheck` which is `False` by default. If it is `True`, `Matcher` returns only those matches with value (i,j) such that i-th descriptor in set A has j-th descriptor in set B as the best match and vice-versa. That is, **the two features in both sets should match each other. It provides consistent result, and is a good alternative to ratio test proposed by D.Lowe in SIFT paper.**

#### <font style="color:rgb(8,133,37)">2. Match Features</font>
Once the matcher is created, two important methods that can be used for mathing are 
- **`BFMatcher.match()`** - returns the best match, or
- **`BFMatcher.knnMatch()`**. - returns k best matches where k is specified by the user. It may be useful when we need to do additional work on that.

#### <font style="color:rgb(8,133,37)">3. Drawing Matches </font>
Like we used [**`cv2.drawKeypoints()`**](https://docs.opencv.org/4.1.0/d4/d5d/group__features2d__draw.html#ga5d2bafe8c1c45289bc3403a40fb88920) to draw keypoints, [**`cv2.drawMatches()`**](https://docs.opencv.org/4.1.0/d4/d5d/group__features2d__draw.html#gad8f463ccaf0dc6f61083abd8717c261a) helps us to draw the matches. It stacks two images horizontally and lines are drawn from the first image to second showing best matches. 

There is also **`cv2.drawMatchesKnn`** which draws all the k best matches. If `k=2`, it will draw two match-lines for each keypoint. So we have to pass a mask if we want to selectively draw it.

 result of `matches = bf.match(des1,des2)` line is a list of `DMatch` objects. This `DMatch` object has following attributes:

- `DMatch.distance` - Distance between descriptors. The lower, the better it is.
- `DMatch.trainIdx` - Index of the descriptor in train descriptors
- `DMatch.queryIdx` - Index of the descriptor in query descriptors
- `DMatch.imgIdx` - Index of the train image.

### Descriptor Matcher 

### <font style="color:rgb(8,133,37)">Function Syntax </font>
#### <font style="color:rgb(8,133,37)">1. Create Matcher object </font>

```python
matcher	=	cv.DescriptorMatcher_create( matcherType	)
```
It takes 1 argument. 

- `matcherType` - Matcher type to be specified. Supported types - BruteForce (it uses L2 ) BruteForce-L1, BruteForce-Hamming, BruteForce-Hamming(2), FlannBased.

### <font style="color:rgb(50,120,229)">FLANN based Matcher </font>

FLANN stands for Fast Library for Approximate Nearest Neighbors. It contains a collection of algorithms optimized for fast nearest neighbor search in large datasets and for high dimensional features. It works faster than BFMatcher for large datasets. 

#### <font style="color:rgb(8,133,37)">Specify algorithm parameters</font>
For FLANN based matcher, we need to specify the algorithm to be used, its related parameters etc. The parameters are 

- `IndexParams`. - Specifies the algorithm to be used

    e.g. For algorithms like SIFT, SURF etc. you can pass following:

```python
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
```

    For ORB, you can pass the following :
    
```python
FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6,
                   key_size = 12,
                   multi_probe_level = 1)
```

- `SearchParams`. It specifies the number of times the trees in the index should be recursively traversed. Higher values gives better precision, but also takes more time. If you want to change the value, pass `search_params = dict(checks=100)`.
