import os
from dotenv import load_dotenv
load_dotenv(dotenv_path = os.path.join(os.path.dirname(__file__), '.env'))

from my_agent.tools import send_email, psychoeducation, session_prep, calendar_input

from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, SystemMessage

# Get the API key
api_key = os.getenv('OPENAI_API_KEY')

# Check if the API key loaded
if api_key:
    print(f'API key successfully loaded. API key: {api_key[:10]}')
else:
    print(f'API key was not successfully loaded. Please check the .env file.')
    exit()

# Create LLM
model = 'gpt-4o'
llm = ChatOpenAI(
    model = model,
    temperature = 0.3,
    api_key = os.getenv('OPENAI_API_KEY')
)

# Bind tools to LLM
llm_with_tools = llm.bind_tools([send_email, psychoeducation, session_prep, calendar_input])


# Create a prompt for mental health coordination
system_message = SystemMessage(content = 
         """
        ROLE: You are part of an existing mental health chatbot called HopeBot. You are a useful mental health coordinator speaking directly to HopeBot users.
         
        RULES:
        - Do not re-classify or question the severity.
        - Call content tools before send-email.
        - Use only tool-provided content.
        - Ensure you use the tools assigned based on the triage. 
        - Let users know that a calendar function is available and offer to add appointments to their calendar if they request it. 
        - If a user requests a tool not assigned by their triage, use your judgement — provide it if it supports their wellbeing, but do not offer clinical-level tools to users triaged as minimal without explaining why a professional referral may not be indicated at this stage

        TOOL USAGE:
        - Always call content tools (psychoeducation, session preparation) BEFORE send_email
        - Use only the exact content returned by tools — do not generate your own self-care tips, interventions, or resources
        - You may re-word tool content to match the user's context.
        - When presenting self-care tips, display each category as a heading and list each tip as a separate bullet point underneath. Do not combine multiple tips into one paragraph.
        - If a tool was not called or returned no content, do not include placeholder text for it.

        EMAIL FORMAT:
        - Format the email body as HTML using <h3> for headings, <ul> and <li> for lists, <a href='...'> for links, and <p> for paragraphs
        - The email must include all of the following from tool outputs:
            1. Assessment Summary (name, raw score, severity)
            2. What This Means
            3. Self-Care Tips
            4. Recommended Interventions
            5. Psychoeducational resources with clickable links
            6. Session Preparation (if applicable)
            7. Disclaimer
        - The email should serve as a complete summary the user can refer back to.
        - The user's email address is provided in the initial triage input. Use it for all email communications without asking again.
        - If the user's name is not provided, introduce yourself and ask for their name at the start of the interaction. 
        - If the user's email is not provided, ask for it only when preparing to send an email.
        - After presenting the care coordination response in chat, automatically send the email summary to the user without waiting to be asked. If the user's email is available, call send_email immediately after calling the content tools.

        CHAT RESPONSE:
        - The chat response must contain the same complete information as the email
        - Follow the same content structure as the email format above
        - Format the chat response in plain text with clear headings, not HTML.

        TONE:  Warm, supportive, professional. Address user by name. Sign off as HopeBot.

        NOTE: Send only one email per conversation. 
        - Gather all content from tools first, and send a single comprehensive email that includes everything — assessment results, psychoeducation, session preparation, and any calendar attachments.
         """)

# ====== NODE FUNCTIONS ======
def agent_node(state: MessagesState):
    messages = [system_message] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode([send_email, psychoeducation, session_prep, calendar_input])

# Routing function: check if agent called a tool
def should_continue(state: MessagesState):
    last_message = state['messages'][-1]
    if last_message.tool_calls:
        return "tools"
    return END

# ===== BUILD THE GRAPH ======
graph = StateGraph(MessagesState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile()

# ======= TRIAGE CHAIN =======
TRIAGE_MAP = {
    'Minimal': 'Minimal',
    'Mild': 'Mild',
    'Moderate': 'Moderate-to-severe',
    'Moderately Severe': 'Moderate-to-severe',
    'Severe': 'Moderate-to-severe'
}

CUTOFFS = {
    "PHQ-9": [(5, "Minimal"), (10, "Mild"), (15, "Moderate"), (20, "Moderately Severe"), (28, "Severe")],
    "GAD-7": [(5, "Minimal"), (10, "Mild"), (15, "Moderate"), (22, "Severe")],
}

def triage_function(score, assessment):
    if assessment in CUTOFFS:
        for threshold, label in CUTOFFS[assessment]:
            if score < threshold:
                severity = label
                break
        triage_category = TRIAGE_MAP[severity]
    else:
        severity = score
        triage_category = severity
    return {'severity': severity, 'triage_category': triage_category}

# ====== ROUTING LOGIC ======
def route_by_severity(assessment: str, triage_category: str, q9: int) -> dict:
    # Safety override — non-zero Q9 always escalates regardless of score
    # Return emergency pathway with crisis-specific tools
    if q9 > 0:
        return {"pathway": 'emergency', "tools": ["send_email", "psychoeducation"]}

    if assessment == 'PHQ-9' or assessment == 'GAD-7':
    # For score-based assessments (PHQ-9, GAD-7):
        if triage_category == "Minimal":
            return {"pathway": "minimal", "tools": ["send_email"]}
        elif triage_category == 'Mild':
            return {"pathway": "mild", "tools": ["send_email", "psychoeducation"]}
        else:
            return {"pathway": "clinical", "tools": ["send_email", "psychoeducation", "session_prep"]}
    else:
    # For binary assessments (MDQ):
        if triage_category == 'Positive':
            return {"pathway": "clinical", "tools": ["send_email", "psychoeducation", "session_prep"]}
        if triage_category == 'Negative':
            return {"pathway": "minimal", "tools": ["send_email"]}

# ====== RESPONSE CHAIN PATHWAY ======
def run_pipeline(screening_data: dict):
    # Step 1: Extract what we need from the screening data
    # score, assessment type, q9, patient info etc.
    score = screening_data['score']
    assessment = screening_data['assessment_type']
    q9 = screening_data.get('question_9', 0)

    # Step 2: Classify — call triage_function to get severity and triage_category
    triage_dict = triage_function(score, assessment)
    severity = triage_dict['severity']
    triage_category = triage_dict['triage_category']

    # Step 3: Route — call route_by_severity to get pathway and tool list
    routing_dict = route_by_severity(assessment, triage_category, q9)
    
    # Step 4: Build the enriched input for the agent
    # Include: patient name, assessment type, score, severity, triage category, pathway, and which tools are available
    # This is a formatted string that gives the LLM all the context it needs without it having to figure any of the clinical logic out itself
    TOOL_DISPLAY_NAMES = {
    "send_email": "Email",
    "psychoeducation": "Psychoeducation",
    "session_prep": "Session Preparation",
    "calendar_input": "Calendar" 
    }
    
    enriched_input = f"""
    Email: {screening_data.get('email', 'Not provided')}
    Assessment: {assessment}
    Score: {score}
    Severity: {severity}
    Triage Category: {triage_category}
    Pathway: {routing_dict['pathway']}
    Available Tools: {', '.join(TOOL_DISPLAY_NAMES[t] for t in routing_dict['tools'])}

    IMPORTANT: When calling the psychoeducation tool, use assessment_type="{assessment}" 
    and severity="{severity}" (NOT the triage category).

    Based on the above triage, respond to the patient and use the available tools to coordinate their care.
    """

    # Step 5: Invoke the executor with the enriched input
    result = app.invoke({"messages": [HumanMessage(content = enriched_input)]})
    
    # Step 6: Return the agent's response
    return result
