

## <font style="color:rgb(50,120,229)">Read and display a Video in OpenCV</font>

Just like we used [**`cv2.imread`**](https://docs.opencv.org/4.1.0/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56) to read an image stored on our machine, we will use **`cv2.VideoCapture`** to create a [**VideoCapture**](https://docs.opencv.org/4.1.0/d8/dfe/classcv_1_1VideoCapture.html#ac4107fb146a762454a8a87715d9b7c96) object and read from input file (video).

### <font style="color:rgb(8,133,37)">Function Syntax </font>

``` python
<VideoCapture object>	=	cv.VideoCapture(		)
<VideoCapture object>	=	cv.VideoCapture(	filename[, apiPreference]	)
<VideoCapture object>	=	cv.VideoCapture(	index[, apiPreference]	)

```

**Parameters**

- **`filename`** it can be:
    - name of video file (eg. video.avi)
    - or image sequence (eg. img_%02d.jpg, which will read samples like img_00.jpg, img_01.jpg, img_02.jpg, ...)
    -or URL of video stream (eg. protocol://host:port/script_name?script_params|auth). Note that each video stream or IP camera feed has its own URL scheme. Please refer to the documentation of source stream to know the right URL.
- **`apiPreference`**:	preferred Capture API backends to use. Can be used to enforce a specific reader implementation if multiple are available: e.g. cv::CAP_FFMPEG or cv::CAP_IMAGES or cv::CAP_DSHOW.

## <font style="color:rgb(50,120,229)">Create a video reader object </font>
```python
cap = cv2.VideoCapture(args)
```

Three most popular ways of reading videos using the VideoCapture Object are :
1. Using Webcam ( Pass the argument as 0 )
2. From a video File ( Specify the filename as argument )
3. Image sequence ( e.g. image_%03d.jpg )

## <font style="color:rgb(50,120,229)">Write a Video in OpenCV</font>

After we are done with capturing and processing the video frame by frame, the next step we would want to do is to save the video.

For images, it is straightforward. We just need to use [**`cv2.imwrite()`**](https://docs.opencv.org/4.1.0/d4/da8/group__imgcodecs.html#gabbc7ef1aa2edfaa87772f1202d67e0ce) and specify an image format(jpg/png). But for videos, some more info is required. 

The steps are as follows:

__1__. Create a [**`VideoWriter`**](https://docs.opencv.org/4.1.0/dd/d9e/classcv_1_1VideoWriter.html#ac3478f6257454209fa99249cc03a5c59) object

### <font style="color:rgb(8,133,37)">Function Syntax </font>

```python
<VideoWriter object>	=	cv.VideoWriter(		)
<VideoWriter object>	=	cv.VideoWriter(	filename, fourcc, fps, frameSize[, isColor]	)
<VideoWriter object>	=	cv.VideoWriter(	filename, apiPreference, fourcc, fps, frameSize[, isColor]	)
```

**Parameters**
- **`filename`**: Name of the output video file.
- **`fourcc`**:	4-character code of codec used to compress the frames. For example, VideoWriter::fourcc('P','I','M','1') is a MPEG-1 codec, VideoWriter::fourcc('M','J','P','G') is a motion-jpeg codec etc. List of codes can be obtained at Video Codecs by FOURCC page. FFMPEG backend with MP4 container natively uses other values as fourcc code: see ObjectType, so you may receive a warning message from OpenCV about fourcc code conversion.
- **`fps`**:	Framerate of the created video stream.
- **`frameSize`**:	Size of the video frames.
- **`isColor`**:	If it is not zero, the encoder will expect and encode color frames, otherwise it will work with grayscale frames (the flag is currently supported on Windows only).

**2**. Write frames to the object in a loop.

**3**. Close and release the object.


### <font style="color:rgb(50,120,229)">FourCC Code</font>

[FourCC](https://en.wikipedia.org/wiki/FourCC) is a 4-byte code used to specify the video codec. The list of available codes can be found at [fourcc.org](http://fourcc.org/). There are many FOURCC codes available, but in this lecture we would work only with MJPG.

#### <font style="color:rgb(200,0,0)">Note</font>
Only a few of the FourCC codes listed above will work on your system based on the availability of the codecs on your system. Sometimes, even when the specific codec is available, OpenCV may not be able to use it. **MJPG is a safe choice.**


## <font style="color:rgb(50,120,229)"> How to use the Keyboard in OpenCV </font>

Getting the input from the keyboard is done using the [**`waitKey()`**](https://docs.opencv.org/4.1.0/d7/dfc/group__highgui.html#ga5628525ad33f52eab17feebcfba38bd7) function.

### <font style="color:rgb(8,133,37)">Function Syntax </font>

``` python
retval	=	cv.waitKey(	[, delay]	)
```

**Parameters**
- **`delay`** : Delay in milliseconds. 0 is the special value that means "forever".