"""Utility script to generate a sample university syllabus PDF for Athena AI Lite."""

from pathlib import Path
import pymupdf as fitz


def create_sample_syllabus(output_path: Path):
    doc = fitz.open()

    # Page 1: Course Overview & Grading
    p1 = doc.new_page()
    p1.insert_text((50, 70), "ATHENA UNIVERSITY - DEPARTMENT OF COMPUTER SCIENCE", fontsize=14, fontname="helv")
    p1.insert_text((50, 95), "CS101: Introduction to Artificial Intelligence (Fall 2026)", fontsize=12, fontname="helv")
    p1.insert_text((50, 130), "Instructor: Prof. Ada Lovelace | Email: lovelace@athena.edu", fontsize=10)
    p1.insert_text((50, 150), "Office: Turing Hall, Room 302 | Office Hours: Tuesdays & Thursdays 2:00 - 4:00 PM", fontsize=10)
    p1.insert_text((50, 180), "Course Description: This course covers foundations of state-space search, heuristics,", fontsize=10)
    p1.insert_text((50, 195), "vector embeddings, neural representations, and Retrieval-Augmented Generation (RAG).", fontsize=10)
    p1.insert_text((50, 230), "GRADING POLICY AND ASSESSMENT BREAKDOWN:", fontsize=11, fontname="helv")
    p1.insert_text((50, 250), "  - Final Capstone Project: 40% (Due December 15th)", fontsize=10)
    p1.insert_text((50, 270), "  - Midterm Examination: 30% (In-class on October 24th)", fontsize=10)
    p1.insert_text((50, 290), "  - Weekly Programming Labs: 20%", fontsize=10)
    p1.insert_text((50, 310), "  - Seminar Participation & Quizzes: 10%", fontsize=10)
    p1.insert_text((50, 350), "Late Submission Policy: Assignments submitted after the deadline incur a 10% deduction", fontsize=10)
    p1.insert_text((50, 365), "per day up to a maximum of 3 days. Submissions beyond 3 days receive zero credit.", fontsize=10)

    # Page 2: Weekly Schedule & Academic Integrity
    p2 = doc.new_page()
    p2.insert_text((50, 70), "WEEKLY LECTURE SCHEDULE:", fontsize=12, fontname="helv")
    p2.insert_text((50, 100), "Week 1: History of AI, Agents, and Search Algorithms (A* and Minimax)", fontsize=10)
    p2.insert_text((50, 120), "Week 2: Vector Embeddings and Dense Representations (Cosine Similarity & FAISS)", fontsize=10)
    p2.insert_text((50, 140), "Week 3: Transformer Architectures and Self-Attention Mechanisms", fontsize=10)
    p2.insert_text((50, 160), "Week 4: Retrieval-Augmented Generation (RAG) Architecture and Provenance", fontsize=10)
    p2.insert_text((50, 180), "Week 5: Multi-Agent Systems: Supervisors, Specialists, and Tool Calling", fontsize=10)
    p2.insert_text((50, 220), "ACADEMIC HONESTY & CODE OF CONDUCT:", fontsize=11, fontname="helv")
    p2.insert_text((50, 240), "All work must represent the student's own effort. Collaboration on conceptual ideas", fontsize=10)
    p2.insert_text((50, 255), "is encouraged, but sharing code or lab solutions constitutes academic misconduct.", fontsize=10)
    p2.insert_text((50, 280), "Required Textbook: Artificial Intelligence: A Modern Approach (Russell & Norvig, 4th ed).", fontsize=10)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))
    doc.close()
    print(f"Sample syllabus generated successfully at: {output_path}")


if __name__ == "__main__":
    target = Path("data/uploads/sample_syllabus.pdf")
    create_sample_syllabus(target)
