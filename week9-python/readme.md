## <font style="color:rgb(50,120,229)">Detect Corners for tracking them </font>
We will use the Shi Tomasi corner detection algorithm to find some points which we will track over the video. It is implemented in OpenCV using the function `goodFeaturesToTrack`.

### <font style="color:rgb(8,133,37)">Function Syntax </font>

```python
cv2.goodFeaturesToTrack(image, maxCorners, qualityLevel, minDistance[, mask[, blockSize]])
```
where,

- `image` - Input image
- `maxCorners` - maximum Number of corners to be detected
- `qualityLevel` - Parameter characterizing the minimal accepted quality of image corners. The parameter value is multiplied by the best corner quality measure
- `minDistance` - Minimum possible Euclidean distance between the returned corners.
- `mask` - Optional region of interest
- `blockSize` - Size of an average block for computing a derivative covariation matrix over each pixel neighborhood

We are specifying the parameters in a separate dictionary as given below.

## <font style="color:rgb(50,120,229)">Lucas Kanade Tracker </font>

### <font style="color:rgb(8,133,37)">Function Syntax </font>

```python
nextPts, status, err = cv2.calcOpticalFlowPyrLK(prevImg, nextImg, prevPts[, winSize[, maxLevel[, criteria]]])
```

where,
- `prevImg` - previous image
- `nextImg` - next image
- `prevPts` - points in previous image
- `nextPts` - points in next image
- `winSize` - size of the search window at each pyramid level
- `maxLevel` - 0-based maximal pyramid level number; if set to 0, pyramids are not used (single level), if set to 1, two levels are used, and so on
- `criteria` - parameter, specifying the termination criteria of the iterative search algorithm (after the specified maximum number of iterations criteria.maxCount or when the search window moves by less than criteria.epsilon