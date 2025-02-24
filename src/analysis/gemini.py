import os
import PIL.Image
from google import genai
from ..config.setting import GEMINI_API_KEY
import datetime

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

def analyze_image(image_path: str) -> str:
    """
    Analyzes an image using Gemini Vision API with detailed UX analysis prompt.
    
    Args:
        image_path: Path to image file
    Returns:
        Analysis text from Gemini
    """
    image = PIL.Image.open(image_path)
    segment_num = int(os.path.basename(image_path).split('_')[1].split('.')[0])
    
    prompt = f"""
You are a user experience expert. Analyze the following image segment of a website thoroughly.

Part 1: Detailed Analysis
- Translate the image content into text.
- Describe all the text content in detail.
- Describe all visible elements in detail (for example: hero section, layout, textual content, imagery, color scheme, fonts, and design patterns).

Part 2: Professional Critique
- Evaluate this segment from a marketing and user experience perspective.
- Focus on design aesthetics, content clarity, and ease of navigation.
- Provide constructive feedback and suggest specific improvements.

Segment Identifier: Segment {segment_num}

Please structure your response as follows:

Segment Analysis:
[Your detailed analysis here]

Critique:
[Your professional critique here]
    """.strip()

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt, image]
    )
    return response.text

def process_folder(folder_path: str) -> str:
    """
    Process all images in folder and compile results into a structured text file.
    
    Args:
        folder_path: Path to folder containing image segments
    Returns:
        Combined analysis text
    """
    # Sort files by segment number
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    files_sorted = sorted(files, key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    results = []
    folder_name = os.path.basename(os.path.normpath(folder_path))
    
    # Add header metadata
    results.append(f"Folder: {folder_name}")
    results.append(f"Number of segments: {len(files_sorted)}")
    results.append("=" * 80 + "\n")
    
    # Process each image
    for filename in files_sorted:
        image_path = os.path.join(folder_path, filename)
        try:
            analysis = analyze_image(image_path)
            # Get current timestamp
            processed_at = datetime.datetime.utcnow().isoformat() + "Z"
            
            # Add a metadata block for each segment
            metadata_block = (
                f"Segment Identifier: {filename}\n"
                f"Segment ID: {int(filename.split('_')[1].split('.')[0])}\n"
                f"Filename: {filename}\n"
                f"Folder: {folder_name}\n"
                f"Processed At: {processed_at}\n"
                + "-" * 60
            )
            results.append(metadata_block)
            results.append(analysis)
            results.append("-" * 60 + "\n")
            print(f"Processed {filename}")
        except Exception as e:
            results.append(f"Failed to process {filename}: {str(e)}\n")
            print(f"Failed to process {filename}: {e}")
    
    # Write results to file
    result_text = "\n".join(results)
    result_file = os.path.join(folder_path, "results.txt")
    with open(result_file, "w", encoding="utf-8") as f:
        f.write(result_text)
    
    return result_text