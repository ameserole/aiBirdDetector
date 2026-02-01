import tensorflow as tf
import numpy as np
from PIL import Image
import os


class BirdIdTFLiteModel:
    """
    TensorFlow Lite model wrapper for bird species classification using a Bird ID model.
    Loads a trained TFLite model and performs inference for bird identification.
    """
    
    def __init__(self, model_path, labels_path=None, image_size=224):
        """
        Initialize the TFLite model.
        
        Args:
            model_path: Path to the .tflite model file
            labels_path: Path to labels file (if None, uses default)
            image_size: Input image size (default 224x224)
        
        Raises:
            FileNotFoundError: If model or labels file not found
            RuntimeError: If model initialization fails
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
        except Exception as e:
            raise RuntimeError(f"Failed to load TFLite model: {e}")
        
        self._input_details = self.interpreter.get_input_details()
        self._output_details = self.interpreter.get_output_details()
        self.image_size = image_size
        
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels file not found: {labels_path}")
        
        try:
            with open(labels_path, 'r') as f:
                self.labels = f.read().strip().split('\n')
        except Exception as e:
            raise RuntimeError(f"Failed to load labels: {e}")
        
        print(f"Bird ID model loaded successfully")
        print(f"  Input shape: {self._input_details[0]['shape']}")
        print(f"  Number of species: {len(self.labels)}")

    def _prepare_image_array(self, image_data):
        """
        Prepare image array for inference.
        
        Args:
            image_data: Either a file path (str) or numpy array (BGR format from cv2)
        
        Returns:
            Prepared numpy array ready for model inference
        
        Raises:
            TypeError: If image_data format is not supported
            RuntimeError: If image preparation fails
        """
        try:
            if isinstance(image_data, str):
                # Load from file path
                if not os.path.exists(image_data):
                    raise FileNotFoundError(f"Image file not found: {image_data}")
                image = Image.open(image_data).resize((self.image_size, self.image_size))
                input_data_type = self._input_details[0]["dtype"]
                image_array = np.array(image, dtype=input_data_type)
            elif isinstance(image_data, np.ndarray):
                # Handle numpy array (expected to be BGR from cv2)
                # Convert BGR to RGB if needed
                if len(image_data.shape) == 3 and image_data.shape[2] == 3:
                    # Resize the image
                    import cv2
                    image = Image.fromarray(cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB))
                    image = image.resize((self.image_size, self.image_size))
                    input_data_type = self._input_details[0]["dtype"]
                    image_array = np.array(image, dtype=input_data_type)
                else:
                    raise ValueError(f"Unexpected array shape: {image_data.shape}. Expected (H, W, 3)")
            else:
                raise TypeError(f"Unsupported image_data type: {type(image_data)}. Expected str (file path) or numpy.ndarray")
            
            # Reshape to model input format
            input_shape = self._input_details[0]['shape']
            image_array = np.array(image_array).reshape(input_shape)
            
            return image_array
        except (FileNotFoundError, ValueError, TypeError):
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to prepare image: {e}")

    def run_from_filepath(self, image_path):
        """
        Run inference on an image file.
        
        Args:
            image_path: Path to image file
        
        Returns:
            List of [species_name, probability] sorted by probability (descending)
        
        Raises:
            FileNotFoundError: If image not found
            RuntimeError: If inference fails
        """
        try:
            image_array = self._prepare_image_array(image_path)
            return self.run(image_array)
        except Exception as e:
            raise RuntimeError(f"Failed to process image {image_path}: {e}")

    def run(self, image):
        """
        Run inference on a numpy array image.
        
        Args:
            image: Input image as (1, image_size, image_size, 3) numpy array
        
        Returns:
            List of [species_name, probability] sorted by probability (descending)
        """
        try:
            self.interpreter.set_tensor(self._input_details[0]["index"], image)
            self.interpreter.invoke()
            tflite_output = self.interpreter.get_tensor(self._output_details[0]["index"])
            probabilities = np.array(tflite_output[0])
            
            # Create list of [label, probability] sorted by probability
            results = []
            for i, prob in enumerate(probabilities):
                results.append([self.labels[i], float(prob)])
            
            return sorted(results, key=lambda x: x[1], reverse=True)
        except Exception as e:
            raise RuntimeError(f"Failed to run inference: {e}")
    
    def classify_from_array(self, image_array, top_k=None):
        """
        Classify bird species from a numpy array without disk I/O.
        
        Args:
            image_array: Image as numpy array (BGR format from cv2, or any format)
            top_k: Number of top predictions to return (None = all)
        
        Returns:
            List of [species_name, probability] sorted by probability (descending)
            If top_k specified, returns top K predictions only
        
        Raises:
            RuntimeError: If inference fails
        """
        try:
            prepared_array = self._prepare_image_array(image_array)
            results = self.run(prepared_array)
            return results[:top_k] if top_k else results
        except Exception as e:
            raise RuntimeError(f"Failed to classify image array: {e}")
    
    def get_top_predictions(self, image, top_k=5):
        """
        Get top K predictions from image.
        
        Args:
            image: Input image as numpy array or filepath
            top_k: Number of top predictions to return
        
        Returns:
            List of top K [species_name, probability] predictions
        """
        if isinstance(image, str):
            results = self.run_from_filepath(image)
        else:
            results = self.run(image)
        
        return results[:top_k]


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    try:
        model = BirdIdTFLiteModel("assets/model.tflite", "assets/labels.txt")
        results = model.get_top_predictions(image_path, top_k=10)
        
        print(f"\nTop predictions for: {image_path}")
        print(f"{'Rank':<6} {'Species':<30} {'Confidence':<12}")
        print("-" * 50)
        
        for rank, (species, confidence) in enumerate(results, 1):
            print(f"{rank:<6} {species:<30} {confidence:>10.1%}")
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)