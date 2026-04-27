from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


class ImageValidationError(ValueError):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


class ImagePreprocessor:
    def __init__(self, image_size: int, max_image_bytes: int) -> None:
        self.image_size = image_size
        self.max_image_bytes = max_image_bytes

    def preprocess(self, payload: bytes) -> np.ndarray:
        if not payload:
            raise ImageValidationError("Uploaded image is empty.")
        if len(payload) > self.max_image_bytes:
            raise ImageValidationError("Uploaded image exceeds the maximum allowed size.", status_code=413)

        try:
            with Image.open(io.BytesIO(payload)) as image:
                image = ImageOps.exif_transpose(image)
                image = image.convert("RGB")
                image = ImageOps.fit(image, (self.image_size, self.image_size), method=Image.Resampling.BILINEAR)
                array = np.asarray(image, dtype=np.float32) / 255.0
        except (UnidentifiedImageError, OSError) as exc:
            raise ImageValidationError("Uploaded file is not a readable image.") from exc

        imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (array - imagenet_mean) / imagenet_std
        chw = np.transpose(normalized, (2, 0, 1))
        return np.expand_dims(chw, axis=0).astype(np.float32)

