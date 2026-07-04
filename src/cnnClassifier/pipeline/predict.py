import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os


# ✅ Alphabetical order — matches flow_from_directory subfolder indexing:
#    0 → glioma
#    1 → meningioma
#    2 → notumor
#    3 → pituitary
CLASS_NAMES = [
    'glioma',
    'meningioma',
    'notumor',
    'pituitary'
]

CLASS_LABELS = {
    'glioma':      'Glioma',
    'meningioma':  'Meningioma',
    'notumor':     'No Tumor',
    'pituitary':   'Pituitary Tumor'
}

CLASS_INFO = {
    'glioma':     'Glioma is a tumor that occurs in the brain and spinal cord. It begins in glial cells. Can be aggressive and requires urgent medical attention.',
    'meningioma': 'Meningioma is a tumor that arises from the meninges. Usually benign and slow-growing, but location can cause serious symptoms.',
    'notumor':    'No tumor detected in this MRI scan. Brain tissue appears normal.',
    'pituitary':  'Pituitary tumor forms in the pituitary gland at the base of the brain. Most are benign but can affect hormone production.'
}


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        # Load trained model
        model = load_model(os.path.join("artifacts", "training", "model.h5"))

        # Preprocess image — same rescale as training generator
        test_image = image.load_img(self.filename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = test_image / 255.0
        test_image = np.expand_dims(test_image, axis=0)

        # Predict
        predictions = model.predict(test_image)
        class_idx = int(np.argmax(predictions, axis=1)[0])
        confidence = float(np.max(predictions))

        class_key = CLASS_NAMES[class_idx]
        label = CLASS_LABELS[class_key]
        info = CLASS_INFO[class_key]

        print(f"Predicted: {class_key} | Label: {label} | Confidence: {confidence:.2%}")

        return [{
            "image":      label,
            "class_key":  class_key,
            "confidence": f"{confidence * 100:.2f}",
            "info":       info
        }]