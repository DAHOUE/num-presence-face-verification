"""A representative excerpt of Num Presence face-verification logic.

The original Num Presence source code is maintained in a private repository.
This public sample presents a cleaned, self-contained excerpt of the same
face-verification workflow, with application-specific code and data removed.

Pipeline: decode image -> detect one face -> extract embedding -> normalize ->
compare against a stored reference embedding using Euclidean distance.
"""

from dataclasses import dataclass
from typing import Optional



@dataclass
class VerificationResult:
    """Result returned by a face-embedding comparison."""
    verified: bool
    distance: Optional[float]
    message: str



# Threshold used for this illustrative sample. It requires calibration and
# validation for the model and data conditions in which it is deployed.
DEFAULT_DISTANCE_THRESHOLD = 0.7



# Initialize InsightFace with CPU inference for a broadly reproducible example.
# The model is pretrained; this sample does not train a face-recognition model.
from insightface.app import FaceAnalysis
import cv2
import numpy as np


FACE_ANALYSIS = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
FACE_ANALYSIS.prepare(ctx_id=-1, det_size=(640, 640))



def decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode uploaded image bytes into an OpenCV BGR image.


    Args:
        image_bytes: Encoded image bytes (e.g., JPEG or PNG).

    Returns:
        Image array in BGR color order.

    Raises:
        ValueError: If bytes are empty or the image cannot be decoded.
    """
    if not image_bytes:
        raise ValueError("Uploaded image is empty.")

    # Convert the byte buffer into a NumPy array suitable for OpenCV decoding.
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)

    # Decode to a 3-channel color image. None indicates a decoding failure.
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid or unsupported image file.")
    return image



def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """Normalize a face feature vector using its L2 (Euclidean) norm."""
    vector = np.asarray(embedding, dtype=np.float32)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("Embedding must be a non-empty 1D vector.")

    # L2 normalization scales the vector to unit length (when its norm is nonzero).
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Embedding has an invalid or zero norm.")
    return vector / norm



def extract_face_embedding(image_bytes: bytes) -> np.ndarray:
    """Detect exactly one face and return its normalized embedding."""
    image = decode_image(image_bytes)



    # Run pretrained face analysis: detection and embedding extraction.
    detected_faces = FACE_ANALYSIS.get(image)

    # Reject ambiguous inputs: no face or more than one face detected.
    if not detected_faces:
        raise ValueError("No face detected in the image.")
    if len(detected_faces) > 1:
        raise ValueError("Multiple faces detected; submit an image with one face.")

    # InsightFace provides the feature vector for the detected face.
    raw_embedding = detected_faces[0].embedding
    return normalize_embedding(raw_embedding)



def euclidean_distance(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
    """Compute L2 distance between two vectors of the same dimension."""
    a = np.asarray(embedding_a, dtype=np.float32)
    b = np.asarray(embedding_b, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError("Embedding dimensions do not match.")

    # Euclidean distance: sqrt(sum((a_i - b_i)^2)).
    return float(np.linalg.norm(a - b))



def verify_embeddings(
    query_embedding: np.ndarray,
    stored_embedding: np.ndarray,
    threshold: float = DEFAULT_DISTANCE_THRESHOLD,
) -> VerificationResult:
    """Compare two embeddings and make a threshold-based match decision."""
    if threshold < 0:
        raise ValueError("Threshold must be non-negative.")

    distance = euclidean_distance(query_embedding, stored_embedding)
    match = distance <= threshold

    return VerificationResult(
        verified=match,
        distance=distance,
        message=(
            "Face embedding matched the reference."
            if match
            else "Face embedding did not match the reference."
        ),
    )



def find_matching_face(db_session, user_face_model, query_embedding: np.ndarray,
                      threshold: float = DEFAULT_DISTANCE_THRESHOLD):
    """Find the nearest stored embedding using a SQLAlchemy/pgvector model.



    This is an illustrative integration pattern. Adapt it to the actual model,
    schema, and pgvector version used by your application.
    """
    from sqlalchemy import select

    vector = np.asarray(query_embedding, dtype=np.float32).tolist()
    distance_expr = user_face_model.embedding.l2_distance(vector)

    # Order by ascending distance to retrieve the nearest candidate.
    statement = (
        select(user_face_model, distance_expr.label("distance"))
        .where(user_face_model.embedding.is_not(None))
        .order_by(distance_expr.asc())
        .limit(1)
    )
    row = db_session.execute(statement).first()
    if row is None:
        return None



    record, distance = row
    distance = float(distance)
    if distance <= threshold:
        return record, distance
    return None

