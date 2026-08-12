import os

from my_agent.tools import send_email, psychoeducation, session_prep, calendar_input

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, SystemMessage

# Function to load the neccessary congigurations
def load_config():
    from dotenv import load_dotenv
    load_dotenv(dotenv_path = os.path.join(os.path.dirname(__file__), '.env'))

    try:
        import streamlit as st
        for key in ['OPENAI_API_KEY', 'LANGCHAIN_TRACING_V2', 'LANGCHAIN_API_KEY', 'LANGCHAIN_PROJECT', 'GMAIL_ADDRESS', 'GMAIL_APP_PASSWORD']:
            val = st.secrets.get(key)
            if val:
                os.environ[key] = str(val)
    except Exception:
        pass

config = load_config()

# Get API key
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
    api_key = api_key
)

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
        - Ask the user whether they have an upcoming mental health appointment they would like to add to their calendar.
        - Call the calendar_input ONLY if the user provides a date and a time. NEVER infer, assume, or invent an appointment date.
        - If the user says they have an appointment but does not give the date, ask for it. If they decline or have no appointment, do not call calendar_input and do not mention a calendar file.

        SEQUENCE (follow this exact order):
        1. Check the Available Tools list. If "Session Preparation" is NOT listed, skip this step entirely and proceed to Step 2. Do not ask the user about their care journey stage.
        If session_prep IS included in the Available Tools list, before calling any tools, provide a short thank you and supportive message to the user and ask: "If you'd like session preparation material, could you let me know what stage of your care journey you're in — for example, are you waiting for your first appointment, already attending sessions, or in between sessions?"
        Classify the user's answer into one of these therapy_stage categories:
        - "PRE" — the user has not yet had their first appointment (e.g. "I'm waiting for my first session", "I haven't started yet")
        - "ONGOING" — the user is currently attending regular therapy sessions (e.g. "I'm in therapy","I have sessions regularly")
        - "BETWEEN" — the user has started treatment but is specifically asking about the gap between sessions (e.g. "I'm waiting for my next session," "it's been a while since my last appointment")
        - "GENERAL" — use this if the user does not specify their stage, gives an unclear answer, or declines to answer
        2. Call psychoeducation tool first
        3. 3. Call session_prep tool ONLY if "Session Preparation" appears in the Available Tools list provided in your context. 
        Do not call this tool otherwise, even if you have already asked about the user's therapy stage.
        4. Ask the user if they would like a calendar reminder for any self-booked mental health care appointment
        5. If yes, ask for the specific date and time of the appointment. Do NOT call calendar_input until the user has provided both a date and a time in their own words.
        6. Once a specific date and time have been given, call calendar_input to generate the .ics file using exactly what the user provided. Never infer, estimate, or invent a date or time.
        7. Call send_email LAST — only after ALL content tools and calendar (if requested) have been called and if the user's provide their email address.
        8. ONLY THEN present the full summary to the user in chat. This applies whether or not the user has provided an email address — the chat summary must always contain the complete content, not a recap of what was emailed.

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
            3. Crisis/safety resources: if Pathway is "emergency" (e.g. Samaritans: 116 123, NHS 111), presented prominently near the top
            4. Self-Care Tips
            5. Recommended Interventions
            6. Psychoeducational resources with clickable links
            7. Session Preparation - include this section only if the session_prep tool was calledd and returned content.
                - If session_prep was not called, omit this section entriely. NEVER write the session prep content yourself.
            8. Disclaimer
        - The email should serve as a complete summary the user can refer back to.
        - Address the user warmly without requiring their name. 
        - The user's email is not provided, offer to send them an email summary — make clear it is optional. For example: 'If you'd like, I can send you a copy of this summary by email — would that be helpful?' Do not ask for their email address again if they decline.
        - After presenting the care coordination response in chat, automatically send the email summary to the user without waiting to be asked. If the user's email is available, call send_email immediately after calling the content tools.

        CHAT RESPONSE:
        - The chat response must contain the same complete information as the email
        - Follow the same content structure as the email format above
        - Format the chat response in plain text with clear headings, not HTML.

        ENDING THE CONVERSATION:
        - After presenting the summary and handling the email (sent or declined), end your message by letting the user know they can type "end" to close 
        the session and receive the closing information, or continue chatting if they wish.
        - Example: "If you'd like to end our conversation, just type 'end' and I'll share the final steps. Otherwise, I'm here if you'd like to keep talking."

        TONE:  Warm, supportive, professional. Address user by name. 
        Sign off with "Warm regards, HopeBot" ONLY in:
        - The email
        Do NOT sign off on intermediate messages such as confirmations, follow-up questions, or brief responses like "I've sent the email". 
        These should end naturally without a sign-off.

        CRISIS/EMERGENCY PATHWAY:
        - If the Pathway is "emergency", this indicates the user may be experiencing thoughts of self-harm or suicide.
        - Prioritise safety: acknowledge the user's distress with care and without judgement.
        - Crisis resources (e.g. Samaritans: 116 123, or NHS 111) must appear FIRST in both the chat response and the email summary, before the assessment summary and any interpretation.
        - Do NOT describe symptoms as mild, manageable, or normal in this pathway, regardless of total score. Omit reassuring framing about overall severity.
        - Still ask the therapy_stage question (Step 1) to ensure session preparation content is appropriately tailored, 
        but do NOT proceed with the calendar reminder question — this routine scheduling step should be skipped in this pathway to avoid adding logistics-related burden during a moment of distress.
        - Still call all tools assigned based on the underlying severity/triage routing (e.g. psychoeducation, session preparation, email) to ensure the user receives complete, relevant content, 
        but frame the response around immediate safety and support first.
        - Strongly encourage the user to seek immediate professional help or contact emergency services if in immediate danger.

        NOTE: Send only one email per conversation. 
        - Gather all content from tools first, and send a single comprehensive email that includes everything — assessment results, psychoeducation, session preparation, and any calendar attachments.
        - Do NOT reintroduce yourself as HopeBot but introduce yourself as the mental health care coordinator
        """)

TOOL_DISPLAY_NAMES= {
    "send_email": "Email",
    "psychoeducation": "Psychoeducation",
    "session_prep": "Session Preparation",
    "calendar_input": "Calendar" 
    }

# ====== NODE FUNCTIONS ======
# Routing function: check if agent called a tool
def should_continue(state: MessagesState):
    last_message = state['messages'][-1]
    if last_message.tool_calls:
        return "tools"
    return END

# ===== BUILD THE GRAPH ======
def build_agent(allowed_tool_names):
    TOOL_REGISTRY = {
    "send_email": send_email,
    "psychoeducation": psychoeducation,
    "session_prep": session_prep,
    "calendar_input": calendar_input,
    }

    tools = [TOOL_REGISTRY[name] for name in allowed_tool_names]
    llm_scoped = llm.bind_tools(tools)

    def agent_node(state: MessagesState):
        messages = [system_message] + state["messages"]
        return {"messages": [llm_scoped.invoke(messages)]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    
    return graph.compile()

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
            base_pathway, base_tools = "minimal", ["send_email", "calendar_input"]
        elif triage_category == 'Mild':
            base_pathway, base_tools = "mild", ["send_email", "psychoeducation", "calendar_input"]
        else:
            base_pathway, base_tools = "clinical", ["send_email", "psychoeducation", "session_prep", "calendar_input"]
    else:
    # For binary assessments (MDQ):
        if triage_category == 'Positive':
            base_pathway, base_tools = "clinical", ["send_email", "psychoeducation", "session_prep", "calendar_input"]
        else:
            base_pathway, base_tools = "minimal", ["send_email", "calendar_input"]
    
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
    agent_app = build_agent(routing_dict['tools'])
    
    # Step 4: Build the enriched input for the agent
    # Include: patient name, assessment type, score, severity, triage category, pathway, and which tools are available
    # This is a formatted string that gives the LLM all the context it needs without it having to figure any of the clinical logic out itself
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
    result = agent_app.invoke(
        {"messages": [HumanMessage(content = enriched_input)]},
        config = {
            "metadata": {"participant_id": participant_id},
            "tags": [f"participant-{participant_id}"]})
    
    # Step 6: Return the agent's response
    return result, agent_app
