import asyncio
from typing import Optional, List
from langchain_community.vectorstores import FAISS
from .analysis.chat import create_chat_chain
from .main import setup_logging
from langchain_openai import OpenAIEmbeddings
from .config.setting import OPENAI_API_KEY, GROQ_API_KEY

from .analysis.vector_store import get_all_analyses

# Additional imports for our analysis chains
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document

# Update imports
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq


# New imports
import os
import re
from datetime import datetime

logger = setup_logging()
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

def load_vector_store(store_path: str = "combined_vectorstore") -> Optional[FAISS]:
    """Load the existing vector store."""
    try:
        logger.info(f"Loading vector store from {store_path}")
        vector_store = FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)
        logger.info("Vector store loaded successfully")
        return vector_store
    except Exception as e:
        logger.error(f"Error loading vector store: {e}")
        return None

# Update the generate_comprehensive_report function
async def generate_comprehensive_report():
    """
    Generate a comprehensive report by:
    1. Running an individual segment analysis on each segment (map step)
    2. Summarizing the combined analyses using a refine-style summarization chain (reduce step)
    3. Printing the final comprehensive report
    """
    # Instead of using vector store retrieval, read directly from results file
    base_dirs = {
        "target": "target_websites"
    }
    unFiltered_mygl_segments = get_all_analyses(base_dirs)

    mygl_segments = []

    for doc in unFiltered_mygl_segments:
        if len(doc.page_content) > 100 :
            mygl_segments.append(doc)

    print("PRinting all mggl segments")
    print(len(mygl_segments))
    
    if not mygl_segments:
        logger.error("No segments found in results file")
        return

    # Create raw_segments directory with parents if it doesn't exist
    segments_dir = os.path.join(os.getcwd(), "raw_segments")
    os.makedirs(segments_dir, exist_ok=True)
    logger.debug(f"Created or verified raw_segments directory at: {segments_dir}")

    # Generate filename with timestamp for raw segments
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    segments_filename = f"raw_segments_{timestamp}.txt"
    segments_path = os.path.join(segments_dir, segments_filename)
    logger.debug(f"Writing segments to: {segments_path}")

    # Write raw segments to file with better error handling
    try:
        with open(segments_path, 'w', encoding='utf-8') as f:
            f.write("=== RAW SEGMENTS FROM RESULTS FILE ===\n\n")
            for idx, doc in enumerate(mygl_segments, start=1):
                f.write(f"--- Segment {idx} ---\n")
                f.write(f"Metadata: {doc.metadata}\n")
                f.write("Content:\n")
                f.write(doc.page_content)
                f.write("\n\n" + "="*80 + "\n\n")
        logger.info(f"Successfully saved {len(mygl_segments)} segments to: {segments_path}")
        print(f"\nRaw segments saved to: {segments_path}")
    except Exception as e:
        logger.error(f"Error saving raw segments: {str(e)}", exc_info=True)
        print(f"Error saving raw segments: {str(e)}")
        return

    # -----------------------------
    # (A) Individual Segment Analysis (Map Step)
    # -----------------------------
    segment_analysis_prompt = PromptTemplate.from_template(
        """You are a world renowed UX and marketing expert, you have helped many companies improve their website's user experience. You have been asked to provide a comprehensive critique of a website. The website is a business website that offers educational products like course. Your critique should focus on the following aspects:
1. Content & Messaging:
   - Clarity of headlines and subheadlines
   - Readability (sentence length, jargon, SEO keyword density)
   - Clarity of the value proposition

2. User Experience (UX):
   - CTA placement (above/below the fold)
   - Form field complexity and carousel usability

4. Trust & Credibility:
   - Presence and authenticity of trust badges and testimonials
   - Security indicators (SSL, privacy policies)

5. Conversion Elements:
   - Use of urgency tactics (deadlines, scholarships)
   - Risk-reduction offers (free trials, refunds)
   - Button contrast versus background

Provide actionable suggestions for improvement. The user may ask for the source of your critique. So only critizes if you are sure. 
Point out the exact piece of content you are critizing so that user can easily find and optimise it as per your improvement suggestions. 


Segment Text:
{text}
It has two section, 
**Segment Analysis:** Contains text details of the segment
**Critique:** Contains only vision critic by a vision model. Just belive and use the same. 

<output_example>
Critics_points : 
1. The "Download Brocure" button is too big , takes away full user attention rather than other useful CTAs. 
2. You have put average salary hike as 50 percent under the heading "Our learners transformed their careers", it has become scam nowdays, put exact number so that people believe you more.

Improvement_suggestions :
1. There is a college photo in the first hero section beside "Build Job-Ready Skills with a Top AI & Machine Learning Course", as it is online, it doesn't make sense.  
</output_example>
"""
    )

    # Update the chain creation
    llm = ChatOpenAI(
        temperature=0.5,
        model_name="gpt-3.5-turbo",
        api_key=OPENAI_API_KEY
    )

    reasoning_llm = ChatGroq(
        model_name='deepseek-r1-distill-qwen-32b',
        temperature=0.3,  # Lower temperature for more focused reasoning
        api_key=GROQ_API_KEY
    )


    # Create the chain using modern syntax
    segment_analysis_chain = (
        {"text": RunnablePassthrough()} 
        | segment_analysis_prompt 
        | llm
    )

    segment_analyses = []
    print("Analyzing each segment individually...")
    for idx, doc in enumerate(mygl_segments, start=1):
        print(f"Analyzing segment {idx}...")
        # Update deprecated run method
        response = await segment_analysis_chain.ainvoke({"text": doc.page_content})
        analysis_text = f"--- Analysis for Segment {idx} ---\n{response.content}"
        segment_analyses.append(analysis_text)

    # -----------------------------
    # (B) Summarization Chain (Reduce Step)
    # -----------------------------
    # Define the initial prompt template for overall report summarization.
    prompt_template = """You are a world renowed UX expert, you have helped many companies improve their website's user experience. You have been asked to provide a comprehensive critique of a website. The website is a business website that offers a variety of services and products. Your critique should focus on the following aspects:

1. Navigation Flow:
   - Logical journey from awareness to conversion
   - Internal linking strategy and breadcrumb consistency

2. Brand Consistency:
   - Color/font consistency and tone of voice
   - Logo placement and overall brand presentation

3. Conversion Funnel:
   - Friction points across segments
   - Presence of lead magnets and clarity of CTAs

4. Technical SEO:
   - Meta descriptions, schema markup, and canonical issues

5. Emotional Journey:
   - Use of emotional triggers (fear, FOMO, aspiration)
   - Storytelling and cognitive load

Given the following analysis text: Note the user may ask for source for your critic. So, only criticise if you are sure. Point out the exact piece of content you are criticising so that user can easily find and optimise it as per your improvement suggestions. 
{text}

CONCISE CRITIQUE REPORT:
Example:
Navigation Flow: The website’s navigation is cluttered and non-intuitive. The absence of a clear breadcrumb trail hinders users from understanding their path from initial awareness to conversion. Consider adding a structured breadcrumb navigation to guide users effectively.
Brand Consistency: In the hero section, the headline “Elevate your career with advanced AI skills” is too vague. It does not specify which AI skills are being taught or the tangible benefits. Revise it to something like “Elevate your career by mastering cutting-edge AI techniques in NLP, Machine Learning, and Deep Learning for real-world impact.” Additionally, inconsistencies in color and font usage across the site dilute the brand identity.
Conversion Funnel: Critical CTAs are either buried below the fold or lack visual prominence, resulting in missed opportunities for engagement. Enhance the conversion funnel by positioning clear, standout call-to-action buttons and incorporating compelling lead magnets.
Technical SEO: Several key pages are missing meta descriptions and proper schema markup, which can adversely affect search engine rankings. Ensure that all pages include optimized metadata and structured data.
Emotional Journey: The content fails to evoke strong emotional responses. The generic headline in the hero section does not instill a sense of urgency or aspiration. Strengthen the narrative by incorporating persuasive language, customer testimonials, and dynamic visuals that tap into emotions like FOMO.
"""

    prompt = PromptTemplate.from_template(prompt_template)

    refine_template = (
        "You are a world renowed UX expert, you have helped many companies improve their website's user experience. You are tasked with refining a preliminary comprehensive critique report. "
        "Below is an existing summary which addresses various aspects of a website's user experience. "
        "Your goal is to produce a final summary that preserves and clearly presents all the following details:\n\n"
        "1. Navigation Flow:\n"
        "   - A logical journey from awareness to conversion\n"
        "   - Internal linking strategy and breadcrumb consistency\n\n"
        "2. Brand Consistency:\n"
        "   - Color/font consistency and tone of voice\n"
        "   - Logo placement and overall brand presentation\n\n"
        "3. Conversion Funnel:\n"
        "   - Identification of friction points across segments\n"
        "   - Presence of lead magnets and clarity of CTAs\n\n"
        "4. Technical SEO:\n"
        "   - Meta descriptions, schema markup, and canonical issues\n\n"
        "5. Emotional Journey:\n"
        "   - Use of emotional triggers (fear, FOMO, aspiration)\n"
        "   - Storytelling and cognitive load\n\n"
        "Existing summary:\n"
        "{existing_answer}\n\n"
        "Additional context:\n"
        "{text}\n\n"
        "Based on the additional context provided, refine the original summary accordingly. "
        "Ensure that your final summary thoroughly covers all the items listed above. "
        "If the additional context does not add value, simply return the original summary."
    )
    refine_prompt = PromptTemplate.from_template(refine_template)

    # Update the summarize chain creation
    # use resoning LLM for this. 
    summarize_chain = load_summarize_chain(
        llm=llm,
        chain_type="refine",
        question_prompt=prompt,
        refine_prompt=refine_prompt,
        return_intermediate_steps=True,
        input_key="input_documents",
        output_key="output_text",
        verbose=True
    )

    # Convert individual segment analyses to Document objects for summarization
    split_docs = [Document(page_content=analysis) for analysis in segment_analyses]

    # # Update to use ainvoke
    summary_result = await summarize_chain.ainvoke(
        {"input_documents": split_docs}
    )
    overall_report = summary_result["output_text"]

    # Combine individual analyses and overall summary into one comprehensive report
    final_report = "\n\n".join([
        "=== FINAL SEGMENT ANALYSES ===",
        "\n\n".join(segment_analyses),
        "=== FINAL OVERALL CRITIQUE REPORT ===",
        overall_report
    ])
    
    # Create reports directory if it doesn't exist
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"website_analysis_{timestamp}.txt"
    report_path = os.path.join(reports_dir, filename)
    
    # Write report to file
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(final_report)
        logger.info(f"Report saved to: {report_path}")
        print(f"\nReport saved to: {report_path}")
    except Exception as e:
        logger.error(f"Error saving report: {e}")
        print("Error saving report to file.")
    
    print("\n--- Comprehensive Report ---\n")
    print(final_report)
    return final_report

# Update the chat_loop function
async def chat_loop():
    """Interactive chat loop."""
    vector_store = load_vector_store()
    if not vector_store:
        logger.error("Failed to load vector store. Please run scrape.py first.")
        return

    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    chat_chain = create_chat_chain(retriever)
    
    logger.info("Starting chat session. Type 'quit' to exit or 'report' for comprehensive report.")
    print("\nChat session started. Type 'quit' to exit or 'report' for comprehensive report.")
    
    while True:
        question = input("\nYour question: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            break
        if question.lower() == 'report':
            await generate_comprehensive_report()
            continue
            
        try:
            logger.info(f"Processing question: {question}")
            # Update to use ainvoke
            response = await chat_chain.ainvoke(question)
            print("response_metadata")
            print(response.response_metadata)
            print(f"\nAnswer: {response}")
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            print("Sorry, there was an error processing your question.")

if __name__ == "__main__":
    asyncio.run(chat_loop())
