# CLIP Gallery-Based Owner Recognition System

## Overview

This system uses **OpenAI CLIP** (Contrastive Language-Image Pre-training) to generate robust visual embeddings for owner recognition. Unlike FaceNet which requires visible faces, CLIP works with any body part and is robust to pose/orientation changes (front, back, side views).

## Key Improvements Over FaceNet

| Feature | FaceNet | CLIP |
|---------|---------|------|
| **View Dependency** | Faces only | Any body part |
| **Pose Robustness** | Medium (front-facing) | High (all poses) |
| **Occlusion Handling** | Poor (face hidden) | Good (uses torso/limbs) |
| **Gallery System** | Single mean embedding | Multiple reference views |
| **Orientation** | Front/back limited | Works at any angle |

## Installation

### Install CLIP

```bash
pip install git+https://github.com/openai/CLIP.git
```

### Verify Installation

```bash
python test_clip_owner.py
```

If successful, you'll see:
```
✓ CLIP model loaded
✓ Gallery size: 0 embeddings
```

## Usage

### 1. Enroll Owner (One-Time Setup)

Capture 3-5 different views of the owner (front, back, side):

```bash
python enroll_owner_clip.py --camera 0 --views 3
```

**Steps:**
1. Camera feed opens
2. Position yourself for a front-facing view
3. Press SPACE to capture
4. Turn around for a back view, press SPACE
5. Turn sideways for a side view, press SPACE

Output:
- Gallery saved to `face_db/clip_owner_gallery.npy`
- Contains 512-D embeddings for each view

### 2. Test Recognition

```bash
python test_clip_owner.py
```

**Steps:**
1. Camera feed opens
2. Stand in different poses/distances
3. Press SPACE to test recognition
4. See the similarity score

### 3. Run Main System

The threat detection system now automatically:
- Loads CLIP gallery if available
- Falls back to FaceNet if CLIP not available
- Uses gallery-based matching (cosine similarity ≥ 0.82)

```bash
python threat_detection_test.py
```

## Gallery System

### How It Works

1. **Enrollment**: Capture multiple views (front, back, side)
   - Each generates a 512-D CLIP embedding
   - Stored in `face_db/clip_owner_gallery.npy`

2. **Recognition**: For each frame
   - Extract full-body crop from person detection
   - Generate CLIP embedding
   - Compare against all gallery embeddings using cosine similarity
   - If `max_similarity ≥ 0.82` → Owner detected

3. **Matching Logic**

```python
# For each detected person
embedding = clip_recognizer.embed_image(crop)
similarities = [cosine_sim(embedding, ref) for ref in owner_gallery]
is_owner = max(similarities) >= 0.82
```

## Configuration

### Threshold Adjustment

Decrease threshold to be more permissive (more false positives):
```python
clip_recognizer = CLIPOwnerRecognizer(threshold=0.75)  # More forgiving
```

Increase threshold to be more strict (more false negatives):
```python
clip_recognizer = CLIPOwnerRecognizer(threshold=0.85)  # Stricter
```

### Gallery Size

- **Minimum**: 1 view (single reference)
- **Recommended**: 3-5 views (front, back, side, different distances)
- **Maximum**: 10+ views (diminishing returns)

## Performance Considerations

### Inference Speed

- **CLIP ViT-B/32**: ~200ms per crop on CPU
- **FaceNet**: ~50ms per face on CPU
- On GPU: Both are <50ms

### Memory Usage

- **CLIP model**: ~600MB
- **Gallery**: ~2MB per 1000 embeddings
- **FaceNet model**: ~100MB

### Optimization Tips

1. **Use smaller crops**: Resizing to 224×224 is CLIP's native size
2. **Batch processing**: Process multiple crops together
3. **Cache embeddings**: Store person embeddings in memory
4. **GPU acceleration**: Use CUDA if available

## Development Notes

### File Structure

```
face_db/
├── clip_owner_gallery.npy          # Gallery embeddings
├── clip_owner_metadata.txt         # Optional metadata
├── owner_embeddings.npy            # Legacy FaceNet embeddings (if using FaceNet)
└── owner_mean_embedding.npy        # Legacy FaceNet mean (if using FaceNet)
```

### Class Integration

**CLIPOwnerRecognizer** in `clip_owner_recognizer.py`:
```python
recognizer = CLIPOwnerRecognizer(threshold=0.82)
recognizer.add_to_gallery(embedding, label="front")
result = recognizer.recognize(image)
```

### Fallback Strategy

If CLIP is not available:
1. System falls back to FaceNet automatically
2. No code changes required
3. Same interface, different backend

## Troubleshooting

### CLIP Installation Issues

```bash
# If pip install fails, try updating pip first
pip install --upgrade pip
pip install git+https://github.com/openai/CLIP.git
```

### Gallery Not Found

```
⚠ Gallery is empty! Run enroll_owner_clip.py first to capture owner views.
```

Solution: Run the enrollment script to capture owner views.

### Poor Recognition Performance

**Causes and Solutions:**
- **Insufficient gallery data**: Enroll 5+ views from multiple distances
- **Threshold too high**: Lower to 0.75-0.80
- **Poor lighting**: Ensure good lighting during enrollment and recognition
- **Pose changes**: Include more pose variations during enrollment

## Future Improvements

1. **Hard negative mining**: Train on false positives
2. **Multi-person retrieval**: Track multiple people simultaneously
3. **Online learning**: Update gallery with new views during runtime
4. **Video-level features**: Use temporal consistency across frames
5. **Attribute-based matching**: Color, clothing, height estimation

## References

- [CLIP Paper](https://arxiv.org/abs/2103.14030)
- [CLIP GitHub](https://github.com/openai/CLIP)
- [Gallery matching research](https://arxiv.org/abs/2005.08973)
