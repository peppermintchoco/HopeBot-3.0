import os
from dotenv import load_dotenv
load_dotenv(dotenv_path = os.path.join(os.path.dirname(__file__), '.env'))

from my_agent.tools import send_email, psychoeducation, session_prep, calendar_input

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, SystemMessage

# Function to load for the neccessary
def load_config():
    try:
        import streamlit as st
        return {
            'OPENAI_API_KEY': st.secrets.get('OPENAI_API_KEY'),
            'LANGCHAIN_TRACING_V2': st.secrets.get('LANGCHAIN_TRACING_V2', 'false'),
            'LANGCHAIN_API_KEY': st.secrets.get('LANGCHAIN_API_KEY'),
            'LANGCHAIN_PROJECT': st.secrets.get('LANGCHAIN_PROJECT'),
            'GMAIL_ADDRESS': st.secrets.get('GMAIL_ADDRESS'),
            'GMAIL_APP_PASSWORD': st.secrets.get('GMAIL_APP_PASSWORD'),
        }
    except Exception:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))
        return {
            'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
            'LANGCHAIN_TRACING_V2': os.getenv('LANGCHAIN_TRACING_V2', 'false'),
            'LANGCHAIN_API_KEY': os.getenv('LANGCHAIN_API_KEY'),
            'LANGCHAIN_PROJECT': os.getenv('LANGCHAIN_PROJECT'),
            'GMAIL_ADDRESS': os.getenv('GMAIL_ADDRESS'),
            'GMAIL_APP_PASSWORD': os.getenv('GMAIL_APP_PASSWORD'),
        }

config = load_config()
os.environ['OPENAI_API_KEY'] = config['OPENAI_API_KEY'] or ''
os.environ['LANGCHAIN_TRACING_V2'] = config['LANGCHAIN_TRACING_V2']
os.environ['LANGCHAIN_API_KEY'] = config['LANGCHAIN_API_KEY'] or ''
os.environ['LANGCHAIN_PROJECT'] = config['LANGCHAIN_PROJECT'] or ''

# Get API key
api_key = config['OPENAI_API_KEY']

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
    api_key = api_key
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
        - At the end of your first response to the user, always offer to send them a calendar reminder for a follow-up appointment. Ask if they would like one and if so, what date and time works for them.
        - If a user requests a tool not assigned by their triage, use your judgement — provide it if it supports their wellbeing, but do not offer clinical-level tools to users triaged as minimal without explaining why a professional referral may not be indicated at this stage

        SEQUENCE (follow this exact order):
        1. Before calling any tools, provide a short thank you and supportive message to the user and ask the user: 
        "If you'd like session preparation material, could you let me know what stage of your care journey you're in — for example, are you waiting for your first appointment, already attending sessions, or in between sessions?"
        Classify the user's answer into one of these therapy_stage categories:
        - "PRE" — the user has not yet had their first appointment (e.g. "I'm waiting for my first session", "I haven't started yet")
        - "ONGOING" — the user is currently attending regular therapy sessions (e.g. "I'm in therapy","I have sessions regularly")
        - "BETWEEN" — the user has started treatment but is specifically asking about the gap between sessions (e.g. "I'm waiting for my next session," "it's been a while since my last appointment")
        - "GENERAL" — use this if the user does not specify their stage, gives an unclear answer, or declines to answer
        2. Call psychoeducation tool first
        3. Call session_prep tool (if in pathway)
        4. Ask the user if they would like a calendar reminder for any self-booked mental health care appointment
        5. If yes, call calendar_input tool to generate the .ics file
        6. Call send_email LAST — only after ALL content tools and calendar (if requested) have been called
        7. ONLY THEN present the full summary to the user in chat

        TOOL USAGE:
        - You MUST call psychoeducation and session_prep tools BEFORE generating any chat response
        - Always call content tools (psychoeducation, session preparation) BEFORE send_email    
        - NEVER generate self-care tips, interventions, or psychoeducational resources yourself
        - ALL content in your response must come directly from tool outputs
        - If you have not called the tools yet, call them first before responding to the user
        - Do not present placeholder text like "I'll gather resources" — gather them first, then respond    
        - You may re-word tool content to match the user's context.
        - When presenting self-care tips, display each category as a heading and list each tip as a separate bullet point underneath. Do not combine multiple tips into one paragraph.

        EMAIL FORMAT:
        - Format the email body as HTML using <h3> for headings, <ul> and <li> for lists, <a href='...'> for links, and <p> for paragraphs
        - The email must include all of the following:
            1. Assessment Summary (name, raw score, severity)
            2. What This Means
            3. Crisis/safety resources (if Pathway is "emergency" — e.g. Samaritans: 116 123, NHS 111), presented prominently near the top
            4. Self-Care Tips
            5. Recommended Interventions
            6. Psychoeducational resources with clickable links
            7. Session Preparation (if applicable)
            8. Disclaimer
        - The email should serve as a complete summary the user can refer back to.
        - Address the user warmly without requiring their name. 
        - The user's email is not provided, offer to send them an email summary — make clear it is optional. For example: 'If you'd like, I can send you a copy of this summary by email — would that be helpful?' Do not ask for their email address again if they decline.
        - After presenting the care coordination response in chat, automatically send the email summary to the user without waiting to be asked. If the user's email is available, call send_email immediately after calling the content tools.

        CHAT RESPONSE:
        - The chat response must contain the same complete information as the email
        - Follow the same content structure as the email format above
        - Format the chat response in plain text with clear headings, not HTML.

        TONE:  Warm, supportive, professional. Address user by name. 
        Sign off with "Warm regards, HopeBot" ONLY on:
        - The final comprehensive summary message
        - The email
        Do NOT sign off on intermediate messages such as confirmations, follow-up questions, or brief responses like "I've sent the email". 
        These should end naturally without a sign-off.

        CRISIS/EMERGENCY PATHWAY:
        - If the Pathway is "emergency", this indicates the user may be experiencing thoughts of self-harm or suicide.
        - Prioritise safety: acknowledge the user's distress with care and without judgement.
        - Provide immediate UK crisis resources (e.g. Samaritans: 116 123, or NHS 111) prominently in your response, before any other content.
        - Still ask the therapy_stage question (Step 1) to ensure session preparation content is appropriately tailored, 
        but do NOT proceed with the calendar reminder question — this routine scheduling step should be skipped in this pathway to avoid adding logistics-related burden during a moment of distress.
        - Still call all tools assigned based on the underlying severity/triage routing (e.g. psychoeducation, session preparation, email) to ensure the user receives complete, relevant content, 
        but frame the response around immediate safety and support first.
        - Strongly encourage the user to seek immediate professional help or contact emergency services if in immediate danger.

        NOTE: Send only one email per conversation. 
        - Gather all content from tools first, and send a single comprehensive email that includes everything — assessment results, psychoeducation, session preparation, and any calendar attachments.
        - Do NOT reintroduce yourself as HopeBot but introduce yourself as the mental health care coordinator
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
    # For PHQ-9/GAD-7: score is numeric, mapped to severity via CUTOFFS
    if assessment in CUTOFFS:
        for threshold, label in CUTOFFS[assessment]:
            if score < threshold:
                severity = label
                break
        triage_category = TRIAGE_MAP[severity]
    else:
        # For MDQ: score is a pre-classified string ("Positive"/"Negative") passed 
        severity = score
        triage_category = severity
    return {'severity': severity, 'triage_category': triage_category}

# ====== ROUTING LOGIC ======
def route_by_severity(assessment: str, triage_category: str, q9: int) -> dict:
    if assessment == 'PHQ-9' or assessment == 'GAD-7':
    # For score-based assessments (PHQ-9, GAD-7):
        if triage_category == "Minimal":
            base_pathway, base_tools = "minimal", ["send_email"]
        elif triage_category == 'Mild':
            base_pathway, base_tools = "mild", ["send_email", "psychoeducation"]
        else:
            base_pathway, base_tools = "clinical", ["send_email", "psychoeducation", "session_prep"]
    else:
    # For binary assessments (MDQ):
        if triage_category == 'Positive':
            base_pathway, base_tools = "clinical", ["send_email", "psychoeducation", "session_prep"]
        else:
            base_pathway, base_tools = "minimal", ["send_email"]
    
    # Safety override — non-zero Q9 always escalates regardless of score
    # Return emergency pathway with crisis-specific tools
    if q9 > 0:
        return {"pathway": 'emergency', "tools": list(set(base_tools + ["send_email", "psychoeducation"]))}
    
    return {"pathway": base_pathway, "tools": base_tools}

# ====== RESPONSE CHAIN PATHWAY ======
def run_pipeline(screening_data: dict, participant_id: str):
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

    {"NOTE: The user indicated some level of thoughts of self-harm on question 9 of the PHQ-9 (Pathway: emergency). This overrides the general severity/triage category shown above — follow the CRISIS/EMERGENCY PATHWAY instructions in your system prompt, prioritising safety resources and support, regardless of the overall severity score." if routing_dict['pathway'] == 'emergency' else ""}

    Based on the above triage, respond to the patient and use the available tools to coordinate their care.
    """

    # Step 5: Invoke the executor with the enriched input
    result = app.invoke({"messages": [HumanMessage(content = enriched_input)]},
                        config = {
                            "metadata": {"participant_id": participant_id},
                            "tags": [f"participant-{participant_id}"]
                        })
    
    # Step 6: Return the agent's response
    return result
