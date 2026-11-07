import streamlit as st
from typing import TypedDict
from langgraph.graph import StateGraph, END


st.set_page_config(page_title="AI Lesson Generator", page_icon="📚")


class LessonState(TypedDict):
    topic: str
    lesson: str
    attempts: int
    checks: list
    rejection_log: list
    passed: bool
    feedback: str


def generate_lesson(state):
    """Generate a free sample lesson locally; no paid API key is needed."""
    topic = state["topic"]

    lesson = f"""
# Introduction

This lesson introduces **{topic}** in a beginner-friendly way. You will learn
what the topic means, why it is useful, and how it can be applied in a
real-world situation.

## Learning objectives

By the end of this lesson, you should be able to:
- Explain the basic meaning of {topic}.
- Describe the important concepts and steps.
- Identify a practical use case.
- Answer a few review questions.

## Beginner-friendly explanation

{topic} can be understood by starting with its basic purpose and then looking
at how its parts work together. First, identify the problem that needs to be
solved. Next, understand the information or resources required. Then follow
the relevant steps in the correct order and check the result.

For a Retrieval-Augmented Generation (RAG) example, **retrieval** means finding
useful information from a collection of documents. **Generation** means
creating a response using the retrieved information. A RAG workflow commonly
loads documents, splits them into smaller chunks, converts the chunks into
embeddings, stores them in a vector database, retrieves relevant chunks for a
question, and uses a language model to generate an answer.

## RAG architecture and workflow

1. Document loading: collect the source documents.
2. Text splitting: divide long documents into smaller chunks.
3. Embedding: convert each chunk into a numerical representation.
4. Vector storage: save the embeddings for searching.
5. Retrieval: find the chunks most relevant to the user's question.
6. Generation: create an answer using the retrieved context.

## Real-world example

Imagine a company has hundreds of policy documents. An employee asks,
"What is the leave policy?" A RAG application searches the documents,
retrieves the relevant leave-policy section, and generates a clear answer
based on that information. This is a real-world example of using retrieval
and generation together. The answer can also refer to the source document
so that the employee can verify the information.

## Quiz with answers

1. What is retrieval?
   **Answer:** Finding relevant information from available documents.

2. What is generation?
   **Answer:** Producing a response using the available context.

3. Why are documents split into chunks?
   **Answer:** Smaller chunks make relevant information easier to retrieve.

4. Give one real-world example.
   **Answer:** Searching company policy documents to answer employee questions.
"""

    return {
        "lesson": lesson,
        "attempts": state["attempts"] + 1,
    }


def evaluate_lesson(state):
    lesson = state["lesson"].lower()

    checks = [
        ("Introduction", "introduction" in lesson),
        ("Learning objectives", "objective" in lesson),
        ("Beginner-friendly explanation", len(lesson) > 500),
        ("RAG concepts", "retrieval" in lesson and "generation" in lesson),
        ("Real-world example", "example" in lesson),
        ("Quiz and answers", "quiz" in lesson and "answer" in lesson),
    ]

    failed = [name for name, passed in checks if not passed]
    log = list(state["rejection_log"])

    if failed:
        log.append({
            "attempt": state["attempts"],
            "failed_checks": failed,
        })

    return {
        "checks": checks,
        "passed": len(failed) == 0,
        "feedback": "Please fix: " + ", ".join(failed) if failed else "",
        "rejection_log": log,
    }


def route_after_evaluation(state):
    if state["passed"] or state["attempts"] >= 3:
        return END
    return "generate"


def build_workflow():
    graph = StateGraph(LessonState)

    graph.add_node("generate", generate_lesson)
    graph.add_node("evaluate", evaluate_lesson)

    graph.set_entry_point("generate")
    graph.add_edge("generate", "evaluate")
    graph.add_conditional_edges(
        "evaluate",
        route_after_evaluation,
        {"generate": "generate", END: END},
    )

    return graph.compile()


st.title("📚 Self-Evaluating Lesson Content Generator")
st.write(
    "Generate a sample lesson, evaluate it against quality checks, "
    "and view the final lesson. This free demo does not use an OpenAI API."
)

topic = st.text_input("Enter lesson topic", "Introduction to RAG")

if st.button("Generate Lesson"):
    if not topic.strip():
        st.error("Please enter a topic.")
    else:
        initial_state = {
            "topic": topic.strip(),
            "lesson": "",
            "attempts": 0,
            "checks": [],
            "rejection_log": [],
            "passed": False,
            "feedback": "",
        }

        try:
            with st.spinner("Generating and evaluating lesson..."):
                workflow = build_workflow()
                result = workflow.invoke(initial_state)

            st.subheader("Quality Check Results")
            for name, passed in result["checks"]:
                if passed:
                    st.success(f"PASS: {name}")
                else:
                    st.error(f"FAIL: {name}")

            st.write("Attempts:", result["attempts"])

            st.subheader("Rejection Log")
            if result["rejection_log"]:
                st.json(result["rejection_log"])
            else:
                st.success("No rejected attempts.")

            if result["passed"]:
                st.success("Lesson successfully passed all checks!")
            else:
                st.warning("Maximum retries reached: Review the lesson.")

            st.subheader("Final Lesson")
            st.markdown(result["lesson"])

        except Exception as error:
            st.error(f"Error: {error}")
            st.info(
                "If LangGraph is missing, run: "
                "pip install langgraph"
            )
