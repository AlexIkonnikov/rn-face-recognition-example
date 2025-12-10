import {StyleSheet, Text, TouchableOpacity} from 'react-native';
import {useEffect, useRef} from 'react';
import {
  Camera,
  useCameraDevice,
  useCameraFormat,
  useCameraPermission,
  useFrameProcessor,
} from 'react-native-vision-camera';
import {
  FaceDetectionOptions,
  useFaceDetector,
} from 'react-native-vision-camera-face-detector';
import {useSharedValue, Worklets} from 'react-native-worklets-core';
import {useResizePlugin} from 'vision-camera-resize-plugin';
import Canvas, {ImageData} from 'react-native-canvas';
import {
  ColorConversionCodes,
  DataTypes,
  InterpolationFlags,
  ObjectType,
  OpenCV,
} from 'react-native-fast-opencv';
import useFaceNet from './useFaceNet.ts';
import {l2Normalize} from './l2Normalize.ts';
import {cosineSimilarity} from './cosineSimilarity.ts';

const FaceDetectorVisionCamera = () => {
  const canvasRef = useRef<Canvas>(null);
  const canvasRef2 = useRef<Canvas>(null);
  const {hasPermission, requestPermission} = useCameraPermission();
  const isActive = useSharedValue(false);
  const {resize} = useResizePlugin();

  const embedding = useRef<Float32Array | null>(null);

  const model = useFaceNet();

  const faceDetectionOptions = useRef<FaceDetectionOptions>({
    cameraFacing: 'back',
    performanceMode: 'accurate',
    landmarkMode: 'all',
  }).current;

  useEffect(() => {
    if (!hasPermission) {
      requestPermission();
    }
  }, [hasPermission, requestPermission]);

  const device = useCameraDevice('front');
  const format = useCameraFormat(device, []);
  const {detectFaces} = useFaceDetector(faceDetectionOptions);

  const draw = Worklets.createRunOnJS(
    async (array: number[], width: number, height: number) => {
      const canvas = canvasRef.current;
      if (!canvas) {
        return;
      }
      const ctx = canvas.getContext('2d');
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      canvas.width = width;
      canvas.height = height;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const imageData = new ImageData(canvas, array, height, width);
      ctx.putImageData(imageData, 0, 0);
    },
  );

  const frameProcessor = useFrameProcessor(
    frame => {
      'worklet';
      if (!isActive.value) {
        return;
      }
      isActive.value = false;
      const desiredFaceWidth = 160;
      const desiredLeftEye = {x: 0.31, y: 0.31};

      const faces = detectFaces(frame);
      if (faces.length === 0) {
        isActive.value = false;
        return;
      }

      const face = faces[0];
      if (!face.landmarks) {
        return;
      }

      const resizedFrame = resize(frame, {
        scale: {
          width: frame.width,
          height: frame.height,
        },
        pixelFormat: 'bgr',
        dataType: 'uint8',
      });

      const srcMat = OpenCV.bufferToMat(
        'uint8',
        frame.height,
        frame.width,
        3,
        resizedFrame,
      );

      const leftEye = face.landmarks.LEFT_EYE;
      const rightEye = face.landmarks.RIGHT_EYE;

      const centerX = (leftEye.x + rightEye.x) / 2;
      const centerY = (leftEye.y + rightEye.y) / 2;

      const eyesCenter = OpenCV.createObject(
        ObjectType.Point2f,
        centerX,
        centerY,
      );

      const dy = rightEye.y - leftEye.y;
      const dx = rightEye.x - leftEye.x;
      let angle = Math.atan2(dy, dx) * (180.0 / Math.PI);

      let desiredRightEyeX = 1 - desiredLeftEye.x;

      const currentDist = Math.hypot(dx, dy);
      const desiredDist =
        (desiredRightEyeX - desiredLeftEye.x) * desiredFaceWidth;

      const scale = desiredDist / currentDist;

      const rotMat = OpenCV.createObject(
        ObjectType.Mat,
        2,
        3,
        DataTypes.CV_64F,
      );
      OpenCV.invoke('getRotationMatrix2D', eyesCenter, angle, scale, rotMat);

      const transformMat = OpenCV.matToBuffer(rotMat, 'float64');
      const tX = desiredFaceWidth * 0.5;
      const tY = desiredFaceWidth * desiredLeftEye.y;

      transformMat.buffer[2] += tX - centerX;
      transformMat.buffer[5] += tY - centerY;

      const updatedRotateMat = OpenCV.bufferToMat(
        'float64',
        2,
        3,
        1,
        transformMat.buffer,
      );

      const output = OpenCV.createObject(
        ObjectType.Mat,
        frame.height,
        frame.height,
        DataTypes.CV_64F,
      );

      OpenCV.invoke(
        'warpAffine',
        srcMat,
        output,
        updatedRotateMat,
        OpenCV.createObject(ObjectType.Size, 160, 160),
      );

      const resizedMat = OpenCV.createObject(
        ObjectType.Mat,
        160,
        160,
        DataTypes.CV_8UC4,
      );

      OpenCV.invoke(
        'resize',
        output,
        resizedMat,
        OpenCV.createObject(ObjectType.Size, 160, 160),
        1,
        1,
        InterpolationFlags.INTER_LINEAR,
      );

      const rgbaMat = OpenCV.createObject(
        ObjectType.Mat,
        160,
        160,
        DataTypes.CV_8UC4,
      );

      OpenCV.invoke(
        'cvtColor',
        resizedMat,
        rgbaMat,
        ColorConversionCodes.COLOR_BGR2RGBA,
      );

      const uint8 = OpenCV.matToBuffer(rgbaMat, 'uint8');
      draw(Array.from(uint8.buffer), uint8.cols, uint8.rows).finally(
        OpenCV.clearBuffers,
      );
    },
    [isActive.value, OpenCV, draw, model, embedding.current],
  );

  return (
    <>
      {device && format && (
        <>
          <Camera
            style={[StyleSheet.absoluteFill]}
            device={device}
            format={format}
            isActive={true}
            frameProcessor={frameProcessor}
            androidPreviewViewType={'texture-view'}
          />
          <TouchableOpacity
            style={styles.container}
            onPress={() => {
              isActive.value = !isActive.value;
            }}>
            <Text>Распознать</Text>
          </TouchableOpacity>
          <Canvas ref={canvasRef} style={styles.canvas} />
          <Canvas ref={canvasRef2} style={styles.canvas2} />
        </>
      )}
    </>
  );
};

const styles = StyleSheet.create({
  container: {
    position: 'absolute',
    width: '100%',
    backgroundColor: 'green',
    bottom: 0,
    padding: 20,
    alignItems: 'center',
  },

  canvas: {
    position: 'absolute',
    bottom: 100,
    width: 160,
    height: 160,
  },
  canvas2: {
    position: 'absolute',
    bottom: 100,
    left: 120,
    width: 160,
    height: 160,
  },
});

export default FaceDetectorVisionCamera;
