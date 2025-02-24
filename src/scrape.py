import asyncio
import os
from typing import Dict, List
from .main import setup_logging, process_all_websites

logger = setup_logging()

async def main():
    logger.info("Starting Website Data Collection")
    
    websites = {
        "target": ["https://www.mygreatlearning.com/pg-program-artificial-intelligence-course"],
        "competitors": [
            "https://www.simplilearn.com/pgp-ai-machine-learning-certification-training-course",
            "https://talentsprint.com/course/ai-machine-learning-iiit-hyderabad"
        ]
    }
    
    try:
        await process_all_websites(websites)
        logger.info("Website data collection completed successfully")
    except Exception as e:
        logger.error("Fatal error in data collection", exc_info=True)
        raise
    finally:
        logger.info("Website Data Collection completed")

if __name__ == "__main__":
    asyncio.run(main())