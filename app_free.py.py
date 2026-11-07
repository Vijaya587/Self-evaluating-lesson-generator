import streamlit as st
from openai import OpenAI
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

def generate_lesson(state, client):
   prompt = f"""
   Create a beginner-friendly lesson on {state['topic']}.
   Include:
   1. Introduction
   2. Learning objectives
   3. Simple explanation
   4. RAG architecture and workflow
   5. Real-world example
   6. Quiz with answers

   Previous feedback: {state['feedback']}
   Make sure to correct previous issues.
"""
   response = client.chat.completions.create(
       model="gpt-4o-mini",
       messages=[{"role": "user", "content": prompt}]
   )
   return {
       "lesson": response.choices[0].message.content,
        "attempts": state["attempts"] + 1
   }

def evaluate_lesson(state):
   lesson = state["lesson"].lower()

   checks = [
       ("Introduction", "introduction" in lesson),
       ("Learning objectives", "objective" in lesson),
       ("Beginner-friendly explanation", len(lesson) > 500),
       ("RAG concepts", "retrieval" in lesson and "generation" in lesson),
       ("Real-world example", "example" in lesson),
       ("Quiz and answers", "quiz" in lesson and "answer" in lesson)
   ]

   failed = [name for name, passed in checks if not passed]
   log = list(state["rejection_log"])

   if failed:
       log.append({
           "attempt": state["attempts"],
           "failed_checks": failed
       })

   return {
       "checks": checks,
       "passed": len(failed) == 0,
       "feedback": "Please fix: " + ", ".join(failed) if failed else "",
       "rejection_log": log
   }

  
def route_after_evaluation(state):
    if state["passed"] or state["attempts"] >= 3:
        return END
    return "generate"

def build_workflow(client):
    graph = StateGraph(LessonState)

    graph.add_node(
       "generate",
       lambda state: generate_lesson(state, client)
    )
    graph.add_node("evaluate", evaluate_lesson)

    graph.set_entry_point("generate")
    graph.add_edge("generate", "evaluate")
    graph.add_conditional_edges(
       "evaluate",
        route_after_evaluation,
        {"generate": "generate", END: END}
    )

    return graph.compile()

st.title("📚 Self-Evaluating Lesson Content Generator")
st.write("AI generates, evaluates and improves lesson content.")

api_key = st.text_input("OpenAI API Key", type="password")
topic = st.text_input("Enter lesson topic", "Introduction to RAG")

if st.button("Generate Lesson"):
    if not api_key:
        st.error("Please enter your OpenAI API key.")
    elif not topic.strip():
        st.error("Please enter a topic.")
    else:
           try:
               client = OpenAI(api_key=api_key)
               workflow = build_workflow(client)

               initial_state = {
                 "topic": topic,
                 "lesson": "",
                 "attempts": 0,
                 "checks": [],
                 "rejection_log": [],
                 "passed": False,
                 "feedback": ""
               }

               with st.spinner("Generating and evaluating lesson..."):
                   result = workflow.invoke(initial_state)

               st.subheader("Quality Check Results")

               for name, passed in result["checks"]:
                   if passed:
                       st.success(f"PASS: {name}")
                   else:
                       st.error(f"FAIL: {name}")

               with st.spinner("Generating and evaluating lesson..."):
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

