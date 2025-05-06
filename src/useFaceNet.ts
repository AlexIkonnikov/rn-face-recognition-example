import {useTensorflowModel} from 'react-native-fast-tflite';

const useFaceNet = () => {
  const plugin = useTensorflowModel(require('./facenet.tflite'));
  return plugin.state === 'loaded' ? plugin.model : undefined;
};

export default useFaceNet;
