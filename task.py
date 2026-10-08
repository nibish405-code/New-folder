import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool


# -----------------------------------------
# LOAD GROQ API KEY
# -----------------------------------------

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("ERROR: GROQ_API_KEY not found.")
    print("Please create a .env file and add your Groq API key.")
    exit()


# -----------------------------------------
# GROQ LLM
# -----------------------------------------

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=api_key
)


# -----------------------------------------
# STUDENT DATA
# -----------------------------------------

students = {
    "Arun": {
        "department": "CSE",
        "attendance": 82
    },

    "Priya": {
        "department": "AI",
        "attendance": 91
    }
}


# -----------------------------------------
# TOOL 1: CALCULATOR
# -----------------------------------------

@tool
def calculator(a: float, b: float, operation: str) -> str:
    """Perform basic mathematical calculations."""

    if operation == "add":
        return str(a + b)

    elif operation == "subtract":
        return str(a - b)

    elif operation == "multiply":
        return str(a * b)

    elif operation == "divide":
        if b == 0:
            return "Cannot divide by zero"
        return str(a / b)

    elif operation == "percentage":
        if b == 0:
            return "Cannot calculate percentage"
        return str((a / b) * 100)

    else:
        return "Invalid operation"


# -----------------------------------------
# TOOL 2: STUDENT INFORMATION
# -----------------------------------------

@tool
def student_info(name: str) -> str:
    """Get the department of a student."""

    for student in students:

        if student.lower() == name.lower():

            return (
                f"{student} is from the "
                f"{students[student]['department']} department."
            )

    return f"I couldn't find {name} in the student database."


# -----------------------------------------
# TOOL 3: ATTENDANCE
# -----------------------------------------

@tool
def get_attendance(name: str) -> str:
    """Get a student's attendance percentage."""

    for student in students:

        if student.lower() == name.lower():

            return (
                f"{student}'s attendance is "
                f"{students[student]['attendance']}%."
            )

    return f"I couldn't find {name} in the student database."


# -----------------------------------------
# BIND TOOLS
# -----------------------------------------

tools = [
    calculator,
    student_info,
    get_attendance
]

llm_with_tools = llm.bind_tools(tools)


# -----------------------------------------
# CHAT FUNCTION
# -----------------------------------------

def ask_agent(question):

    messages = [

        {
            "role": "system",
            "content": """
You are a Student Assistant AI.

Choose the correct tool when necessary.

Use calculator for mathematical questions.

Use student_info for questions about
a student's department.

Use get_attendance for attendance questions.

For general questions, answer directly.

Give short and simple answers.
"""
        },

        {
            "role": "user",
            "content": question
        }
    ]

    # First LLM response
    response = llm_with_tools.invoke(messages)

    # -------------------------------------
    # No tool required
    # -------------------------------------

    if not response.tool_calls:
        return response.content

    # -------------------------------------
    # Execute tools
    # -------------------------------------

    messages.append(response)

    for call in response.tool_calls:

        tool_name = call["name"]
        arguments = call["args"]

        if tool_name == "calculator":

            result = calculator.invoke(arguments)

        elif tool_name == "student_info":

            result = student_info.invoke(arguments)

        elif tool_name == "get_attendance":

            result = get_attendance.invoke(arguments)

        else:

            result = "Tool not found"

        messages.append(
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "content": str(result)
            }
        )

    # -------------------------------------
    # Final LLM response
    # -------------------------------------

    final_answer = llm_with_tools.invoke(messages)

    return final_answer.content


# -----------------------------------------
# MAIN PROGRAM
# -----------------------------------------

print("------------------------------------")
print("       STUDENT ASSISTANT AI")
print("------------------------------------")
print("Type 'exit' to stop.")

while True:

    question = input("\nYou: ")

    if question.lower() == "exit":
        print("Program ended.")
        break

    try:

        answer = ask_agent(question)

        print("AI:", answer)

    except Exception as e:

        print("Error:", e)