from PIL import Image
import os

def segment_image(image_path: str, segment_height: int, overlap: int, output_folder: str, output_prefix: str = "segment_") -> list:
    """
    Splits an image into vertical segments with overlap.
    
    Args:
        image_path: Path to input image
        segment_height: Height of each segment in pixels
        overlap: Overlap between segments in pixels 
        output_folder: Folder to save segments
        output_prefix: Prefix for segment filenames
    
    Returns:
        List of paths to saved valid segments
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    img = Image.open(image_path)
    width, height = img.size
    y = 0
    segment_index = 1
    valid_segments = []
    
    while y < height:
        bottom = min(y + segment_height, height)
        segment = img.crop((0, y, width, bottom))
        
        # Skip uniform segments
        seg_rgb = segment.convert("RGB")
        colors = seg_rgb.getcolors(maxcolors=1000000)
        if colors and len(colors) == 1 and colors[0][1] in [(255, 255, 255), (0, 0, 0)]:
            y = y + segment_height - overlap
            segment_index += 1
            continue

        segment_path = os.path.join(output_folder, f"{output_prefix}{segment_index}.png")
        segment.save(segment_path)
        valid_segments.append(segment_path)
        
        if bottom == height:
            break
            
        y = y + segment_height - overlap
        segment_index += 1

    return valid_segments