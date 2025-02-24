import pytest
from PIL import Image
import os
from src.image_processing.segmentation import segment_image

@pytest.fixture
def test_image():
    img = Image.new('RGB', (800, 2000), color='white')
    # Add some non-white content
    for y in range(0, 2000, 200):
        img.paste((0, 0, 0), (0, y, 800, y + 100))
    return img

def test_segment_image(tmp_path, test_image):
    image_path = tmp_path / "test.png"
    test_image.save(image_path)
    
    segments = segment_image(
        str(image_path),
        segment_height=500,
        overlap=50,
        output_folder=str(tmp_path)
    )
    
    assert len(segments) > 0
    assert all(os.path.exists(s) for s in segments)