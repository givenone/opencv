## <font style="color:rgb(50,120,229)">GrabCut</font>

GrabCut is implemented in OpenCV using a function named [**`grabCut`**](https://docs.opencv.org/4.1.0/d7/d1b/group__imgproc__misc.html#ga909c1dda50efcbeaa3ce126be862b37f).

### <font style="color:rgb(8,133,37)">Function Syntax</font>

```python
mask, bgdModel, fgdModel	=	cv.grabCut(	img, mask, rect, bgdModel, fgdModel, iterCount[, mode]	)
```

**Parameters**
- **`img`**:	Input 8-bit 3-channel image.
- **`mask`**:	Input/output 8-bit single-channel mask. The mask is initialized by the function when mode is set to GC_INIT_WITH_RECT. Its elements may have one of the GrabCutClasses.
- **`rect`**:	ROI containing a segmented object. The pixels outside of the ROI are marked as "obvious background". The parameter is only used when mode==GC_INIT_WITH_RECT .
- **`bgdModel`**:	Temporary array for the background model. Do not modify it while you are processing the same image.
- **`fgdModel`**:	Temporary arrays for the foreground model. Do not modify it while you are processing the same image.
- **`iterCount`**:	Number of iterations the algorithm should make before returning the result. Note that the result can be refined with further calls with mode==GC_INIT_WITH_MASK or mode==GC_EVAL .
- **`mode`**:	Operation mode that could be one of the GrabCutModes

## Histogram of Oriented Gradients

## <font style="color:rgb(50,120,229)">HOG Descriptor in OpenCV</font>

OpenCV implements HOG using the HOGDescriptor class.

**Python** = hog : **cv2.HOGDescriptor**(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma, histogramNormType, L2HysThreshold, gammaCorrection, nlevels, useSignedGradients)

<center><img src="https://www.learnopencv.com/wp-content/uploads/2018/01/opcv4face-w7-m2-blockNormalization2.png"/></center>

&nbsp;
The image above will be used as a reference for describing important parameters below. Please see the description of HOG in the previous section if needed.

- **winSize** = This parameter is set to the size of the window over which the descriptor is calculated. In classification problems, we set it to the size of the image. E.g. we set it to 64x128 for pedestrian detection.

- **blockSize** = The size of the blue box in the image. The notion of blocks exist to tackle illumination variation. A large block size makes local changes less significant while a smaller block size weights local changes more. Typically blockSize is set to 2 x cellSize.

- **blockStride** = The blockStride determines the overlap between neighboring blocks and controls the degree of contrast normalization. Typically a blockStride is set to 50% of blockSize.

- **cellSize** = The cellSize is the size of the green squares. It is chosen based on the scale of the features important to do the classification. A very small cellSize would blow up the size of the feature vector and a very large one may not capture relevant information.

- **nbins** = Sets the number of bins in the histogram of gradients. The authors of the HOG paper had recommended a value of 9 to capture gradients between 0 and 180 degrees in 20 degrees increments. 

- **derivAperture** = Size of the Sobel kernel used for derivative calculation.

- **winSigma** = According to the HOG paper, it is useful to “downweight pixels near the edges of the block by applying a Gaussian spatial window to each pixel before accumulating orientation votes into cells”. winSigma is the standard deviation of this Gaussian. In practice, it is best to leave this parameter to default ( -1 ). On doing so, winSigma is automatically calculated as shown below: winSigma = ( blockSize.width + blockSize.height ) / 8

- **histogramNormType** = In the HOG paper, the authors use four different kinds of normalization. OpenCV 3.2 implements only one of those types L2Hys. So, we simply use the default. L2Hys is simply L2 normalization followed by a threshold (L2HysThreshold)where all values above a threshold are clipped to that value.

- **L2HysThreshold** = Threshold used in L2Hys normalization. E.g. If the L2 norm of a vector is  [0.87, 0.43, 0.22], the L2Hys normalization with L2HysThreshold = 0.8 is [0.8, 0.43, 0.22].

- **gammaCorrection** = Boolean indicating whether or not Gamma correction should be done as a pre-processing step.

- **nlevels** = Number of pyramid levels used during detection. It has no effect when the HOG descriptor is used for classification.

- **signedGradient** = Typically gradients can have any orientation between 0 and 360 degrees. These gradients are referred to as “signed” gradients as opposed to “unsigned” gradients that drop the sign and take values between 0 and 180 degrees. In the original HOG paper, unsigned gradients were used for pedestrian detection.

## <font style="color:rgb(50,120,229)">Training a HOG based Classifier for Eye Glasses</font>
In this section, we share the code and walk you through the steps of building a HOG based classifier. The classifier looks at an image patch around the eyes and classifies it as wearing glasses or not wearing glasses.


## SVM

```
def svmInit(C, gamma):
  model = cv2.ml.SVM_create()
  model.setGamma(gamma)
  model.setC(C)
  model.setKernel(cv2.ml.SVM_RBF)
  model.setType(cv2.ml.SVM_C_SVC)

  return model

def svmTrain(model, samples, responses):
  model.train(samples, cv2.ml.ROW_SAMPLE, responses)
  return model
```

- Type of SVM. We choose here the type C_SVC that can be used for n-class classification (n ≥ 2). The important feature of this type is that it deals with imperfect separation of classes (i.e. when the training data is non-linearly separable). This feature is not important here since the data is linearly separable and we chose this SVM type only for being the most commonly used.  
- Type of SVM kernel. We have not talked about kernel functions since they are not interesting for the training data we are dealing with. Nevertheless, let's explain briefly now the main idea behind a kernel function. It is a mapping done to the training data to improve its resemblance to a linearly separable set of data. This mapping consists of increasing the dimensionality of the data and is done efficiently using a kernel function. We choose here the type LINEAR which means that no mapping is done. This parameter is defined using cv::ml::SVM::setKernel.  
- Termination criteria of the algorithm. The SVM training procedure is implemented solving a constrained quadratic optimization problem in an iterative fashion. Here we specify a maximum number of iterations and a tolerance error so we allow the algorithm to finish in less number of steps even if the optimal hyperplane has not been computed yet. This parameter is defined in a structure cv::TermCriteria .


## Object Detection using HOG
-> Multiscale에서 Sliding Window 기반으로 Object Detection.

## <font style="color:rgb(50,120,229)">Testing the model on a new image </font>
We want to load the SVM models trained by us and use it to find pedestrians in a new image. For this, we need to use the function detectMultiScale present in OpenCV. 

## <font style="color:rgb(50,120,229)">Usage of hog.**detectMultiScale**</font>

Once the classifier is loaded into the detector, the function **detectMultiScale** is used to detect objects in an image.


### <font style="color:rgb(8,133,37)">Function Syntax</font>

```python
(foundLocations, foundWeights) = hog.detectMultiScale(img, hitThreshold=hitThreshold, winStride=winStride, padding=padding, scale=scale,  finalThreshold=finalThreshold, useMeanshiftGrouping=useMeanshiftGrouping)
```

where

- **`img`** = Input image.

- **`foundLocations`** = A vector of rectangles corresponding to detected objects.

- **`foundWeights`** = SVM response (weights) associated with the above bounding boxes. You can treat this weight as a measure of confidence.

- **`hitThreshold`** = The SVM response threshold above which we say the rectangle is an object. Usually, the default value of 0 is good. Note elements of foundWeights will always be greater than hitThreshold.

- **`winStride`** = The stride of the object detection window as explained in the Location search subsection above.

- **`padding`** = Padding added to the entire image so objects near the boundary can be detected. Usually this is set to half the detection window width. E.g. in pedestrian detection where the detection window has size 64 x 128, the padding is set to 32 x 32.

- **`scale`** = To create the image pyramid, the image at a level is scaled down by this number. E.g. when scale is set to 1.05, it means that second level of of the pyramid is created by resizing the original image by a factor of 1/1.05. The third level is created by rescaling the image at level 2 by a factor of 1/1.05 and so on.

- **`finalThreshold`** = A better name for this parameter would have been groupThreshold. In the non-maximum suppression step, this parameter is used to prune out clusters that have fewer than finalTreshold number of rectangles.

- **`useMeanshiftGrouping`** = Use the non-standard grouping algorithm based on meanshift. Based on reading the code and the documentation, it is not clear when this is a good idea.


## HAAR Cascade object detection

Object Detection using Haar feature-based cascade classifiers is an effective object detection method proposed by Paul Viola and Michael Jones in their paper, "Rapid Object Detection using a Boosted Cascade of Simple Features" in 2001. It is a machine learning based approach where a cascade function is trained from a lot of positive and negative images. It is then used to detect objects in other images.

Here we will work with face detection. Initially, the algorithm needs a lot of positive images (images of faces) and negative images (images without faces) to train the classifier. Then we need to extract features from it. For this, Haar features shown in the below image are used. They are just like our convolutional kernel. Each feature is a single value obtained by subtracting sum of pixels under the white rectangle from sum of pixels under the black rectangle.

Now, all possible sizes and locations of each kernel are used to calculate lots of features. (Just imagine how much computation it needs? Even a 24x24 window results over 160000 features). For each feature calculation, we need to find the sum of the pixels under white and black rectangles. To solve this, they introduced the integral image. However large your image, it reduces the calculations for a given pixel to an operation involving just four pixels. Nice, isn't it? It makes things super-fast.

But among all these features we calculated, most of them are irrelevant. For example, consider the image below. The top row shows two good features. The first feature selected seems to focus on the property that the region of the eyes is often darker than the region of the nose and cheeks. The second feature selected relies on the property that the eyes are darker than the bridge of the nose. But the same windows applied to cheeks or any other place is irrelevant. So how do we select the best features out of 160000+ features? It is achieved by **Adaboost**.

For this, we apply each and every feature on all the training images. For each feature, it finds the best threshold which will classify the faces to positive and negative. Obviously, there will be errors or misclassifications. We select the features with minimum error rate, which means they are the features that most accurately classify the face and non-face images. (The process is not as simple as this. Each image is given an equal weight in the beginning. After each classification, weights of misclassified images are increased. Then the same process is done. New error rates are calculated. Also new weights. The process is continued until the required accuracy or error rate is achieved or the required number of features are found).

The final classifier is a weighted sum of these weak classifiers. It is called weak because it alone can't classify the image, but together with others forms a strong classifier. The paper says even 200 features provide detection with 95% accuracy. Their final setup had around 6000 features. (Imagine a reduction from 160000+ features to 6000 features. That is a big gain).

So now you take an image. Take each 24x24 window. Apply 6000 features to it. Check if it is face or not. Wow.. Isn't it a little inefficient and time consuming? Yes, it is. The authors have a good solution for that.

In an image, most of the image is non-face region. So it is a better idea to have a simple method to check if a window is not a face region. If it is not, discard it in a single shot, and don't process it again. Instead, focus on regions where there can be a face. This way, we spend more time checking possible face regions.

For this they introduced the concept of **Cascade of Classifiers**. Instead of applying all 6000 features on a window, the features are grouped into different stages of classifiers and applied one-by-one. (Normally the first few stages will contain very many fewer features). If a window fails the first stage, discard it. We don't consider the remaining features on it. If it passes, apply the second stage of features and continue the process. The window which passes all stages is a face region. How is that plan!

The authors' detector had 6000+ features with 38 stages with 1, 10, 25, 25 and 50 features in the first five stages. (The two features in the above image are actually obtained as the best two features from Adaboost). According to the authors, on average 10 features out of 6000+ are evaluated per sub-window.

The main function used in the demo is that of `detectMultiscale`

### <font style="color:rgb(8,133,37)">Function Syntax</font>

```python
cv2.CascadeClassifier.detectMultiScale(image[, scaleFactor[, minNeighbors]])
```

Where,

- **`image`** is the input grayscale image.
- **`objects`** is the rectangular region enclosing the objects detected
- **`scaleFactor`** is the parameter specifying how much the image size is reduced at each image scale. It is used to create the scale pyramid. 
- **`minNeighbors`** is a parameter specifying how many neighbors each candidate rectangle should have, to retain it. Higher number gives lower false positives.

```python
# Load the cascade classifier from the xml file.
faceCascade = cv2.CascadeClassifier(DATA_PATH + 'models/haarcascade_frontalface_default.xml')
faceNeighborsMax = 10
neighborStep = 1

# Perform multi scale detection of faces
plt.figure(figsize=(18,18))
count = 1
for neigh in range(1, faceNeighborsMax, neighborStep):
    faces = faceCascade.detectMultiScale(frameGray, 1.2, neigh)
    frameClone = np.copy(frame)
    for (x, y, w, h) in faces:
        cv2.rectangle(frameClone, (x, y), 
                      (x + w, y + h), 
                      (255, 0, 0),2)

    cv2.putText(frameClone, 
    "# Neighbors = {}".format(neigh), (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 6)
    
    plt.subplot(3,3,count)
    plt.imshow(frameClone[:,:,::-1])
    count += 1

plt.show()
```