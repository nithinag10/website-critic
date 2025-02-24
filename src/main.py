import asyncio
import os
import logging
import colorlog
from urllib.parse import urlparse
from typing import Dict, List
import tiktoken
from datetime import datetime

# Import existing modules
from src.screenshot.capture import capture_screenshot
from src.image_processing.segmentation import segment_image
from src.analysis.gemini import process_folder
from src.analysis.vector_store import create_vector_store, get_all_analyses
from src.analysis.chat import create_chat_chain
from src.config.setting import SEGMENT_HEIGHT, SEGMENT_OVERLAP

# Configure logging
def setup_logging():
    """Configure logging with custom format and both file and console handlers."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('website_critic')
    logger.setLevel(logging.DEBUG)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Colored console formatter
    console_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s | %(blue)s%(message)s%(reset)s",
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'red,bg_white',
        },
        secondary_log_colors={},
        style='%'
    )
    
    # File handler (detailed logging)
    file_handler = logging.FileHandler(f'{log_dir}/website_critic_{timestamp}.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler (colored output)
    console_handler = colorlog.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Set to DEBUG to see more details
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Add divider for better readability
    print("\n" + "="*80)
    print(" Website Critic Starting ".center(80, "="))
    print("="*80 + "\n")
    
    return logger

# Initialize logger
logger = setup_logging()

TOKEN_LIMIT = 8191  # OpenAI's max token limit for embeddings

# Initialize the tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

async def process_website(url: str, output_base: str) -> None:
    """Process a single website end-to-end."""
    logger.info(f"Starting processing for website: {url}")
    
    domain = urlparse(url).netloc.replace("www.", "")
    output_dir = os.path.join(output_base, domain)
    os.makedirs(output_dir, exist_ok=True)
    logger.debug(f"Created output directory: {output_dir}")
    
    try:
        logger.info("Capturing screenshot...")
        screenshot = await capture_screenshot(url)
        logger.debug(f"Screenshot captured successfully: {len(screenshot)} bytes")
        
        temp_path = os.path.join(output_dir, "temp.png")
        with open(temp_path, "wb") as f:
            f.write(screenshot)
        logger.debug(f"Temporary screenshot saved to: {temp_path}")
        
        logger.info("Segmenting screenshot...")
        segments = segment_image(temp_path, SEGMENT_HEIGHT, SEGMENT_OVERLAP, output_dir)
        logger.info(f"Created {len(segments)} segments")
        logger.debug(f"Segment paths: {segments}")
        
        os.remove(temp_path)
        logger.debug("Temporary screenshot removed")
        
        logger.info("Analyzing segments with Gemini Vision...")
        process_folder(output_dir)
        logger.info("Segment analysis complete")
        
    except Exception as e:
        logger.error(f"Error processing website {url}: {str(e)}", exc_info=True)
        raise

def count_tokens(text: str) -> int:
    """Returns the number of tokens in the given text using OpenAI's tokenizer."""
    return len(tokenizer.encode(text))

async def process_all_websites(websites: Dict[str, List[str]]) -> None:
    """Process all websites and create combined vector store."""
    logger.info("Starting batch processing of websites")
    logger.debug(f"Websites to process: {websites}")
    
    tasks = []
    for category, urls in websites.items():
        base_dir = f"{category}_websites"
        os.makedirs(base_dir, exist_ok=True)
        logger.debug(f"Created directory for {category}: {base_dir}")
        tasks.extend([process_website(url, base_dir) for url in urls])
    
    logger.info("Processing all websites concurrently...")
    await asyncio.gather(*tasks)
    logger.info("Website processing complete")
    
    base_dirs = {
        "target": "target_websites",
        "competitors": "competitors_websites"
    }
    
    logger.info("Collecting analyses from all websites...")
    all_documents = get_all_analyses(base_dirs)
    logger.info(f"Collected {len(all_documents)} document segments")
    
    logger.info("Checking token limits for each segment...")
    for doc in all_documents:
        segment_text = doc.page_content
        segment_tokens = count_tokens(segment_text)
        
        logger.debug(
            f"Segment {doc.metadata['segment_index']} "
            f"(Website: {doc.metadata['website']}): {segment_tokens} tokens"
        )
        
        if segment_tokens > TOKEN_LIMIT:
            excess_tokens = segment_tokens - TOKEN_LIMIT
            logger.warning(
                f"Segment {doc.metadata['segment_index']} exceeds token limit "
                f"by {excess_tokens} tokens"
            )
    
    logger.info("Creating vector store...")
    create_vector_store(all_documents, "combined_vectorstore")
    logger.info("Vector store created successfully")