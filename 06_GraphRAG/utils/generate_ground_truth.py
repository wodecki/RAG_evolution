#!/usr/bin/env python3
"""
Ground Truth Generator for GraphRAG vs Naive RAG Comparison
==========================================================

Uses GPT-5 with full CV context to generate authoritative answers
for test questions. This provides an independent, unbiased ground truth
for evaluating both GraphRAG and Naive RAG systems.
"""

from dotenv import load_dotenv
load_dotenv(override=True)

import os
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any
import logging
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GroundTruthGenerator:
    """Generate ground truth answers using GPT-5 with full CV context."""

    def __init__(self):
        """Initialize the ground truth generator."""
        # Use GPT-5 for superior reasoning with large context
        self.llm = ChatOpenAI(
            model="gpt-5",
            max_tokens=4096,
            api_key=os.getenv("OPENAI_API_KEY")
        )

        # Directories
        self.data_dir = Path("data/programmers")
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)

        logger.info("✓ GPT-5 Ground Truth Generator initialized")

    def load_all_cvs(self) -> List[str]:
        """Load all CV PDFs and extract text content."""
        cv_texts = []
        cv_files = list(self.data_dir.glob("*.pdf"))

        if not cv_files:
            raise FileNotFoundError(f"No PDF files found in {self.data_dir}")

        logger.info(f"Loading {len(cv_files)} CV files...")

        for cv_file in sorted(cv_files):
            try:
                loader = PyPDFLoader(str(cv_file))
                documents = loader.load()

                # Combine all pages into single text
                cv_text = "\n".join([doc.page_content for doc in documents])
                cv_texts.append(f"=== CV: {cv_file.stem} ===\n{cv_text}")

            except Exception as e:
                logger.warning(f"Could not load {cv_file}: {e}")
                continue

        logger.info(f"✓ Successfully loaded {len(cv_texts)} CVs")
        return cv_texts

    def create_ground_truth_prompt(self) -> PromptTemplate:
        """Create the prompt template for ground truth generation."""
        template = """You are a senior HR manager with exceptional analytical skills and perfect memory.
You have been given ALL {num_cvs} CVs from our candidate database to review comprehensively.

Your task is to answer the following question with ABSOLUTE PRECISION and COMPLETENESS based on the CVs provided.

CRITICAL INSTRUCTIONS:
- For counting questions: Provide the exact number, count carefully
- For listing questions: Include ALL matches, not just examples or samples
- For aggregation questions: Calculate the actual numerical result (average, sum, max, etc.)
- For filtering questions: Apply all criteria strictly and list every match
- For ranking questions: Sort properly and provide the exact ranking
- Be specific with names, skills, companies, and other details
- If something cannot be determined from the CVs, state that clearly
- Show your reasoning step-by-step before giving the final answer

CVs Database ({num_cvs} total CVs):
{context}

Question: {question}

Step-by-step analysis:
1. [Identify relevant information from CVs]
2. [Apply the query logic/calculation]
3. [Verify completeness and accuracy]

Final Answer:"""

        return PromptTemplate(
            input_variables=["num_cvs", "context", "question"],
            template=template
        )

    async def generate_ground_truth_for_question(self, question: str, all_cv_texts: List[str]) -> Dict[str, Any]:
        """Generate ground truth answer for a single question."""
        try:
            # Combine all CVs into context
            full_context = "\n\n" + "="*80 + "\n\n".join(all_cv_texts)

            # Check context size (rough estimate)
            context_tokens = len(full_context.split()) * 1.3  # Rough token estimate
            logger.info(f"Context size: ~{context_tokens:,.0f} tokens for question: {question[:50]}...")

            if context_tokens > 300000:  # Conservative limit for GPT-5
                logger.warning(f"Context size may be large: {context_tokens:,.0f} tokens")

            # Create prompt
            prompt_template = self.create_ground_truth_prompt()
            prompt = prompt_template.format(
                num_cvs=len(all_cv_texts),
                context=full_context,
                question=question
            )

            # Generate answer with GPT-5
            logger.info(f"Generating ground truth for: {question}")
            response = await self.llm.ainvoke(prompt)

            ground_truth_answer = response.content.strip()

            logger.info(f"✓ Generated ground truth ({len(ground_truth_answer)} chars)")

            return {
                "question": question,
                "ground_truth_answer": ground_truth_answer,
                "context_tokens_estimate": int(context_tokens),
                "num_cvs_used": len(all_cv_texts),
                "model_used": "gpt-5",
                "status": "success"
            }

        except Exception as e:
            logger.error(f"Error generating ground truth for '{question}': {e}")
            return {
                "question": question,
                "ground_truth_answer": f"ERROR: {str(e)}",
                "context_tokens_estimate": 0,
                "num_cvs_used": len(all_cv_texts),
                "model_used": "gpt-5",
                "status": "error",
                "error": str(e)
            }

    def load_test_questions(self) -> Dict[str, Any]:
        """Load test questions from JSON file."""
        questions_file = Path(__file__).parent / "test_questions.json"
        if not questions_file.exists():
            raise FileNotFoundError(f"Test questions file not found: {questions_file}")

        with open(questions_file, 'r') as f:
            return json.load(f)

    async def generate_all_ground_truths(self) -> Dict[str, Any]:
        """Generate ground truth answers for all test questions."""
        # Load CVs and questions
        all_cv_texts = self.load_all_cvs()
        test_data = self.load_test_questions()

        # Extract all questions from all categories
        all_questions = []
        for category_name, category_data in test_data["test_suite"]["categories"].items():
            for question in category_data["questions"]:
                all_questions.append({
                    "question": question,
                    "category": category_name,
                    "description": category_data["description"]
                })

        logger.info(f"Generating ground truth for {len(all_questions)} questions...")

        # Generate ground truth for each question
        results = []
        for i, question_data in enumerate(all_questions):
            question = question_data["question"]
            category = question_data["category"]

            logger.info(f"\n[{i+1}/{len(all_questions)}] Processing {category}: {question}")

            # Generate ground truth
            result = await self.generate_ground_truth_for_question(question, all_cv_texts)
            result["category"] = category
            result["category_description"] = question_data["description"]
            result["question_index"] = i + 1

            results.append(result)

            # Small delay to respect rate limits
            await asyncio.sleep(1)

        # Compile final results
        ground_truth_data = {
            "metadata": {
                "generated_by": "GroundTruthGenerator",
                "model": "gpt-5",
                "num_questions": len(results),
                "num_cvs": len(all_cv_texts),
                "cv_source_dir": str(self.data_dir),
                "total_successful": len([r for r in results if r["status"] == "success"]),
                "total_errors": len([r for r in results if r["status"] == "error"])
            },
            "ground_truth_answers": results,
            "original_test_questions": test_data
        }

        return ground_truth_data

    def save_ground_truth(self, ground_truth_data: Dict[str, Any]) -> Path:
        """Save ground truth data to JSON file."""
        output_file = self.results_dir / "ground_truth_answers.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(ground_truth_data, f, indent=2, ensure_ascii=False)

        logger.info(f"✓ Ground truth saved to: {output_file}")
        return output_file

    def display_summary(self, ground_truth_data: Dict[str, Any]) -> None:
        """Display a summary of the ground truth generation."""
        metadata = ground_truth_data["metadata"]
        results = ground_truth_data["ground_truth_answers"]

        print("\n" + "="*60)
        print("Ground Truth Generation Summary")
        print("="*60)
        print(f"Model Used: {metadata['model']}")
        print(f"Total Questions: {metadata['num_questions']}")
        print(f"CVs Analyzed: {metadata['num_cvs']}")
        print(f"Successful: {metadata['total_successful']}")
        print(f"Errors: {metadata['total_errors']}")

        # Show category breakdown
        print(f"\nQuestions by Category:")
        categories = {}
        for result in results:
            cat = result["category"]
            categories[cat] = categories.get(cat, 0) + 1

        for category, count in categories.items():
            print(f"  • {category}: {count} questions")

        # Show sample ground truth
        print(f"\nSample Ground Truth Answers:")
        for result in results[:3]:
            if result["status"] == "success":
                answer = result["ground_truth_answer"]
                truncated = answer[:100] + "..." if len(answer) > 100 else answer
                print(f"\nQ: {result['question']}")
                print(f"A: {truncated}")

        print("\n" + "="*60)


async def main():
    """Main function to generate ground truth answers."""
    print("Ground Truth Generator for GraphRAG vs Naive RAG Comparison")
    print("=" * 65)

    try:
        generator = GroundTruthGenerator()

        # Generate all ground truths
        ground_truth_data = await generator.generate_all_ground_truths()

        # Save results
        output_file = generator.save_ground_truth(ground_truth_data)

        # Display summary
        generator.display_summary(ground_truth_data)

        print(f"\n🎉 Ground truth generation complete!")
        print(f"📁 Results saved to: {output_file}")
        print(f"\nNext steps:")
        print(f"1. Create naive RAG baseline: uv run python 4_naive_rag_cv.py")
        print(f"2. Run comparison: uv run python 5_compare_systems.py")

    except Exception as e:
        logger.error(f"Ground truth generation failed: {e}")
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())