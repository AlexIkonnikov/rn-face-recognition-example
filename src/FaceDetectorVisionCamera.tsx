import {StyleSheet, Text, TouchableOpacity} from 'react-native';
import {useEffect, useRef} from 'react';
import {
  Camera,
  Frame,
  useCameraDevice,
  useCameraFormat,
  useCameraPermission,
  useFrameProcessor,
} from 'react-native-vision-camera';
import {
  FaceDetectionOptions,
  Landmarks,
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
import {cosineSimilarity} from './cosineSimilarity.ts';
import {l2Normalize} from './l2Normalize.ts';

const FaceDetectorVisionCamera = () => {
  const canvasRef = useRef<Canvas>(null);
  const canvasRef2 = useRef<Canvas>(null);
  const {hasPermission, requestPermission} = useCameraPermission();
  const isActive = useSharedValue(false);
  const {resize} = useResizePlugin();

  const embedding = useRef<Float32Array | null>(null);

  const model = useFaceNet();

  const faceDetectionOptions = useRef<FaceDetectionOptions>({
    cameraFacing: 'front',
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

  const drawOnCanvas = Worklets.createRunOnJS(
    async (
      floatRGBArray: Float32Array,
      width: number,
      height: number,
      isFirst: boolean,
    ) => {
      const canvas = isFirst ? canvasRef.current : canvasRef2.current;
      if (!canvas || !floatRGBArray) {
        return;
      }
      const ctx = canvas.getContext('2d');
      canvas.width = width;
      canvas.height = height;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const imageDataArray = new Uint8ClampedArray(width * height * 4);

      for (let i = 0, j = 0; i < imageDataArray.length; i += 4, j += 3) {
        imageDataArray[i] = Math.round(floatRGBArray[j] * 255); // R
        imageDataArray[i + 1] = Math.round(floatRGBArray[j + 1] * 255); // G
        imageDataArray[i + 2] = Math.round(floatRGBArray[j + 2] * 255); // B
        imageDataArray[i + 3] = 255; // Alpha (полностью непрозрачное)
      }
      const imageData = new ImageData(
        canvas,
        Array.from(imageDataArray),
        height,
        width,
      );
      ctx.putImageData(imageData, 0, 0);
    },
  );

  const frameProcessor = useFrameProcessor(
    frame => {
      'worklet';
      if (!isActive.value) {
        return;
      }

      const faces = detectFaces(frame);
      if (faces.length === 0) {
        isActive.value = false;
        return;
      }

      const face = faces[0];
      if (!face?.landmarks?.LEFT_EYE || !face?.landmarks?.RIGHT_EYE) {
        isActive.value = false;
        return;
      }

      isActive.value = false;

      const {x, y, height, width} = face.bounds;

      const resizedFrame = resize(frame, {
        scale: {
          width: frame.width,
          height: frame.height,
        },
        pixelFormat: 'bgr',
        dataType: 'uint8',
      });

      const srcMat = OpenCV.frameBufferToMat(
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

      const currentDist = Math.hypot(dx, dy); // текущее расстояние между глазами
      const desiredDist = 80.0; // целевое расстояние между глазами
      const scale = desiredDist / currentDist;

      const rotMat = OpenCV.createObject(
        ObjectType.Mat,
        2,
        3,
        DataTypes.CV_64F,
      );
      OpenCV.invoke('getRotationMatrix2D', eyesCenter, angle, scale, rotMat);

      const {cols, rows} = OpenCV.toJSValue(srcMat);

      const dstSize = OpenCV.createObject(ObjectType.Size, cols, rows);

      const alignedMat = OpenCV.createObject(
        ObjectType.Mat,
        0,
        0,
        DataTypes.CV_8UC3,
      );
      OpenCV.invoke('warpAffine', srcMat, alignedMat, rotMat, dstSize);

      // Задаём квадратный ROI 160x160, центрированный на точке eyesCenter (середине глаз).
      const cx = Math.round(centerX); // координаты eyesCenter были float, округляем до целого
      const cy = Math.round(centerY);
      const half = 80; // половина стороны квадрата 160

      // Координаты левого верхнего угла ROI (с учётом границ изображения)
      let roiX = cx - half;
      let roiY = cy - half;
      if (roiX < 0) {
        roiX = 0;
      }
      if (roiY < 0) {
        roiY = 0;
      }

      const alignedMatInfo = OpenCV.toJSValue(alignedMat);

      if (roiX + 160 > alignedMatInfo.cols) {
        roiX = alignedMatInfo.cols - 160;
      }
      if (roiY + 160 > alignedMatInfo.rows) {
        roiY = alignedMatInfo.rows - 160;
      }

      const faceRect = OpenCV.createObject(
        ObjectType.Rect,
        roiX,
        roiY,
        160,
        160,
      );

      // Вырезаем ROI из повернутого изображения
      const faceMat = OpenCV.createObject(
        ObjectType.Mat,
        0,
        0,
        DataTypes.CV_8UC3,
      );
      OpenCV.invoke('crop', alignedMat, faceMat, faceRect);

      // Масштабируем faceMat к точному размеру 160x160 (если нужно)
      const size160 = OpenCV.createObject(ObjectType.Size, 160, 160);
      const faceResized = OpenCV.createObject(
        ObjectType.Mat,
        0,
        0,
        DataTypes.CV_8UC3,
      );
      OpenCV.invoke(
        'resize',
        faceMat,
        faceResized,
        size160,
        1,
        1,
        InterpolationFlags.INTER_LINEAR,
      );

      const faceRGB = OpenCV.createObject(
        ObjectType.Mat,
        0,
        0,
        DataTypes.CV_8UC3,
      );
      OpenCV.invoke(
        'cvtColor',
        faceResized,
        faceRGB,
        ColorConversionCodes.COLOR_BGR2RGB,
      );

      const faceFloat = OpenCV.createObject(
        ObjectType.Mat,
        0,
        0,
        DataTypes.CV_32FC3,
      );
      OpenCV.invoke(
        'convertTo',
        faceRGB,
        faceFloat,
        DataTypes.CV_32FC3,
        1 / 255,
        0,
      );

      const result = OpenCV.matToBuffer(faceFloat, 'float32');

      drawOnCanvas(result.buffer, result.rows, result.cols, !embedding.current);

      const faceEmbeddings = model?.runSync([result.buffer]) as Float32Array[];

      if (!faceEmbeddings) {
        OpenCV.clearBuffers();
        return;
      }

      if (!embedding.current) {
        embedding.current = l2Normalize(faceEmbeddings[0].slice());
      } else {
        console.log(
          cosineSimilarity(embedding.current, l2Normalize(faceEmbeddings[0])),
        );
      }

      OpenCV.clearBuffers();
    },
    [isActive.value, OpenCV, drawOnCanvas, model, embedding.current],
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
            onPress={() => {
              isActive.value = !isActive.value;
            }}
            style={styles.btn}>
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
  btn: {
    position: 'absolute',
    width: '100%',
    bottom: 0,
    backgroundColor: 'green',
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
