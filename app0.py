import streamlit as st
import os
from audio_recorder_streamlit import audio_recorder
from streamlit_float import float_init
import base64
import sys
import json
import chardet

import openai
import os
from dotenv import load_dotenv
import base64
import streamlit as st
import openai
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, AIMessage
# import pysqlite3 as sqlite3

from my_agent.agent import run_pipeline

# --------------------------------------------------------------------------------------------------------------------------logic2END

st.set_page_config(page_title="HopeBot: Your Mental Health Assistant", layout="wide")
# sys.modules["sqlite3"] = sqlite3
openai.api_key = st.secrets.get("OPENAI_API_KEY")

# Define system prompt
SYSTEM_PROMPT = """
    You are HopeBot, a professional psychotherapist specialising in Cognitive Behavioural Therapy. Your role is to focus on your clients' words and emotions, guiding them to reflect on their thoughts and behaviours through open-ended questions and guiding them through the PHQ-9 test. 
    Always show empathy and understanding of their feelings and help them to recognise how their behaviour affects their emotions. 
    Please keep your own responses concise and avoid giving long, repetitive answers, while using open questions and reflections to encourage the client to elaborate.
    You need to focus on listening, encourage clients to express themselves, and help them sort out and explore their emotions and thoughts. 
    If a customer comes to you for advice, give up to 2 at a time. 
    You need to provide helpful advice and assistance to users when they are experiencing extreme emotions, and start by adding encouraging sentences such as "You don't have to face this alone." 

    You must complete three tasks in turn:
    Task 1: Begin by warmly greeting the client and creating a comfortable space for conversation.
    Use the following techniques to encourage engagement:
    - Ask open questions that invite the client to express themselves freely and explore their thoughts and feelings, rather than questions answerable in one word.
    - Offer affirmations that acknowledge the client's strengths, efforts, and coping attempts.
    - Use reflections — brief empathetic restatements of what the client has said — to demonstrate understanding and invite them to elaborate.
    - Before introducing the PHQ-9, provide a short summary of the key points discussed, acknowledging what the client has shared and emphasizing how valuable their input has been.
    If the client indicates they have nothing further to share, or the dialogue reaches about 20 exchanges, smoothly transition to introducing the PHQ-9 questionnaire and 
    ask whether they would like to take it.
    
    PHQ-9 ITEMS (Use this exact wording):
    {phq9_items}

    Task 2: 
    - After the user agrees to use the PHQ-9, let them know that they can ask you to clarify any question if they are unsure what it means. 
    - Then ask each question in turn.
    - Always ensure to include the question and the possible responses (Not at all, Several days, More than half the days, Nearly every day). 
    - Accurately categorise the user's answers as options A, B, C or D using record_phq9_answer. If the user's answer is not precise enough, ambiguous or cannot be accurately categorised, ask the user to provide a clearer 
    answer. You must call record_phq9_answer immediately after classifying each answer, one question at a time, before moving to the next question.
    - If the user's answer requires interpretation to map onto an option — including approximate or non-standard phrasings such as "a few times", "maybe 3/4 times", or "once or twice" — 
    treat this as an inferred classification and call record_phq9_answer with inferred=true.
    - If the user asks you to infer or choose their answer based on the conversation, do the same: make your best classification from what they've shared and record it via record_phq9_answer with inferred=true.
    - In both cases, you MUST call record_phq9_answer FIRST (with inferred=true), in the same turn — before or alongside asking them to confirm.  
    - If the user then corrects you, call record_phq9_answer again with the corrected answer; it will update the record.
    - Confirmation is required whenever the answer was inferred especially for Question 9.
    - Record the classification, then confirm before proceeding: "Based on what you've shared, I'd classify this as [option]. Does that feel right?" 
    - Do not advance to the next question or to Task 3 until the user has confirmed.
    - If the user asks a clarifying question about any PHQ-9 question before giving their actual answer (e.g. "what does this mean", "any examples", "what kind of thoughts"), 
    provide clarification and wait for their actual answer. Do NOT proceed to the next question, Task 3, or any language suggesting a handoff to the care coordinator until record_phq9_answer has been successfully called for the current question.
    - Under no circumstances should the conversation advance to Task 3 or reference connecting the user to the care coordinator without record_phq9_answer having been called for Question 9 specifically.

    Task 3: Once all 9 questions have been classified, simply acknowledge that the assessment is complete and let the user know you are connecting them with HopeBot's care coordinator, who will share their full results and next steps.
    Do not list scores, categories, or totals yourself — this is handled separately.

    IMPORTANT: Do NOT use bullet points, numbered lists, headers, or any markdown formatting in your responses — write in natural spoken prose only.

    Please maintain the demeanour of a professional psychologist at all times and show empathy in your interactions.
    Here is some additional background information to help guide your responses:\n\n{context}
"""

with open("phq9_items.txt", r) as f:
    PHQ9_ITEMS = f.read()

# Define function calling for recording PHQ9 scores
tools = [{
    'type': 'function',
    'function': {
    'name': 'record_phq9_answer',
    'description': """ 
    Call this function ONLY when you are confident that you can classify the user's PHQ-9 answer — whether they chose an option explicitly or you inferred 
    their answer from natural language. Do not call during clarification turns or when still explaining options.
    
    If the user asks you to choose or infer the answer for them based on what they've shared (e.g. "you pick", "based on what I told you"), make your best classification 
    from the conversation context, set inferred=true, and call this function — do NOT leave the question unrecorded. Confirm your inference with the user in your response.
    Approximate or non-standard phrasings (e.g. 'a few times', 'maybe 3/4 times') should also be classified with inferred=true.
    You MUST call record_phq9_answer FIRST (with inferred=true), in the same turn — before or alongside asking them to confirm. Never ask "does that feel right?" without 
    having already recorded your classification. If the user then corrects you, call record_phq9_answer again with the corrected answer — it will update 
    the record.
    
    For Question 9 specifically: responses like "I don't think so", "not really", "I haven't had those", "no" should be classified as A (Not at all, score 0). 
    Do not leave Q9 unrecorded — it must be classified before Task 3 begins.
    """,
    'parameters': {
        'type': 'object',
        'properties':{
            'question_answer': {'type': 'integer'},
            'answer_category': {
                'type': 'string',
                'enum': ['A', 'B', 'C', 'D'],
                'description': 'A=Not at all, B=Several days, C=More than half the days, D=Nearly every day'
            },
            'score': {
                'type': 'integer',
                'enum': [0, 1, 2, 3]
            },
            'inferred': {
                'type': 'boolean',
                'description' : 'True if answer is inferred from natural language rather than explicit choice.'
                },
            'skipped': {'type': 'boolean'}
        },
        'required': ['question_answer', 'answer_category', 'score', 'inferred']
    }}
}]

# Function to initialize resources
@st.cache_resource
def initialize_resources():
    # Chat model
    chat = ChatOpenAI(
        model="gpt-4o",
        temperature=0.4
    )

    # Detect file encoding
    with open(r'cleaned_data.txt', 'rb') as f:
        result = chardet.detect(f.read())
        encoding = result['encoding']

    # Embedding model
    embed_model = OpenAIEmbeddings()

    # Vector stores
    vectorstore1 = Chroma(
        embedding_function=embed_model, 
        persist_directory="cleaned_data"
    )
    vectorstore2 = Chroma(
        embedding_function=embed_model, 
        persist_directory="mental_health"
    )
    vectorstore3 = Chroma(
        embedding_function=embed_model, 
        persist_directory="econ"
    )

    # Retrievers
    retriever1 = vectorstore1.as_retriever(k=2)
    retriever2 = vectorstore2.as_retriever(k=2)
    retriever3 = vectorstore3.as_retriever(k=2)

    # Return all initialized resources
    return retriever1, retriever2, retriever3

retriever1, retriever2, retriever3 = initialize_resources()

# Function to process input and return the chatbot's response
def get_assistant_response(messages):
    # Extract the user's last message (the latest user input)
    user_input = messages[-1]["content"]

    # Retrieve documents based on user input
    retrieved_docs1 = retriever1.invoke(user_input)
    retrieved_docs2 = retriever2.invoke(user_input)
    retrieved_docs3 = retriever3.invoke(user_input)
    combined_context = "\n".join([
        doc.page_content for doc in 
        retrieved_docs1 + retrieved_docs2 + retrieved_docs3
    ])

    # Build progress context
    recorded = st.session_state.recorded_question_numbers
    pending = [q for q in range(1, 10) if q not in recorded]

    if pending:
        next_required = min(pending)
        progress_note = f"""
        PHQ-9 PROGRESS TRACKER (do not share with user): 
        - Questions recorded so far: {recorded}. 
        - PENDING (must still be recorded): {pending}. 
        - NEXT question to ask: Question {next_required}
        - CRITICAL: Ask ONLY question {next_required} next. Do NOT skip ahead. Do not ask any other PHQ-9 question until Question {next_required} 
        has been recorded via the record_phq9_answer function.
        - Do NOT proceed to Task 3 until all pending questions are recorded via record_phq9_answer.
        """
    else:
        progress_note = f"\n\nPHQ-9 PROGRESS TRACKER: All 9 questions recorded. Proceed to Task 3."

    # Manually inject context into system prompt
    system_prompt = SYSTEM_PROMPT.replace("{context}", combined_context).replace("{phq9_items}", PHQ9_ITEMS) + progress_note

    openai_messages = [{'role': 'system', 'content': system_prompt}]

    for m in messages:
        openai_messages.append({'role': m['role'], 'content': m['content']})

    response = openai.chat.completions.create(
        model="gpt-4o",
        temperature=0.4,
        messages = openai_messages,
        tools = tools,
        tool_choice = 'auto'
    )

    # Return the assistant's response
    return response, openai_messages

def extract_agent_responses(agent_results):
    messages = agent_results['messages']

    for message in reversed(messages):
        if hasattr(message, 'content') and message.content:
            return message.content
    return ""

def display_text(content):
    lines = content.split("\n")
    formatted = "<br>".join(f"<span style='font-size: 24px;'>{line}</span>" for line in lines)
    st.markdown(f"<div style='font-size: 24px; margin: 0;'>{formatted}</div>", unsafe_allow_html=True)

# 语音识别功能
def speech_to_text(audio_path):
    with open(audio_path, "rb") as audio_file:
        transcript = openai.audio.transcriptions.create(
            model="whisper-1", response_format="text", file=audio_file
        )
    return transcript.strip()

# 语音合成功能
def text_to_speech(text):
    response = openai.audio.speech.create(model="tts-1", voice="nova", input=text)
    audio_path = "response_audio.mp3"
    with open(audio_path, "wb") as f:
        f.write(response.content)
    return audio_path

# 音频播放功能
def autoplay_audio(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    b64_audio = base64.b64encode(data).decode("utf-8")
    st.markdown(
        f"""
        <audio autoplay>
        <source src="data:audio/mp3;base64,{b64_audio}" type="audio/mp3">
        </audio>
        """,
        unsafe_allow_html=True,
    )

# Function to build assessment score summary
def build_score_summary():
    categories = {"A": 0, "B": 1, "C": 2, "D": 3}
    category_labels = {
        "A": "Not at all", "B": "Several days", 
        "C": "More than half the days", "D": "Nearly every day"
    }

    lines = ["Here's how each answer was interpreted:"]
    for i, cat in enumerate(st.session_state.answers_record, 1):
        score = categories[cat]
        lines.append(f"Question {i}: {category_labels[cat]} ({score} point)")
    
    lines.append(f"\nYou scored {st.session_state.total_phq9_score} points on the PHQ-9.")

    # Q9 safety message
    q9_category = st.session_state.answers_record[8]

    if q9_category in ["B", "C", "D"]:
        lines.append(
            "\nYour response to Question 9 suggests you may be experiencing thoughts of self-harm or suicide. "
            "You don't have to face this alone — please reach out for support. "
            "You can contact the Samaritans at any time on 116 123 (free, 24/7) or by email at jo@samaritans.org."
        )
    
    return "<br>".join(lines)

# Function to trigger when phq-9 assessment is completed
PHQ9_TOTAL_QUESTIONS = 9

def phq9_complete():
    return len(st.session_state.answers_record) == PHQ9_TOTAL_QUESTIONS

# 初始化会话状态
def initialize_session_state():
    if "participant_id" not in st.session_state:
        st.session_state.participant_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Before we begin, could you please enter your Participant ID (e.g. P01)?"}]
    if "total_phq9_score" not in st.session_state:
        st.session_state.total_phq9_score = 0
    if "answers_record" not in st.session_state:
        st.session_state.answers_record = []  # e.g., ["A","B",...]
    if 'inferred_answers' not in st.session_state:
        st.session_state.inferred_answers = []
    if "agent_ran" not in st.session_state:
        st.session_state.agent_ran = False
    if "agent_results" not in st.session_state:
        st.session_state.agent_results = None
    if "phq9_scores_by_question" not in st.session_state:
        st.session_state.phq9_scores_by_question = []
    if "recorded_question_numbers" not in st.session_state:
        st.session_state.recorded_question_numbers = []
    if "awaiting_confirmation" not in st.session_state:
        st.session_state.awaiting_confirmation = False
    if "closing_check" not in st.session_state:
        st.session_state.closing_check = False
    if "read_aloud_enabled" not in st.session_state:
        st.session_state.read_aloud_enabled = True
    if "play_greeting" not in st.session_state:
        st.session_state.play_greeting = False

initialize_session_state()

# 标题
st.title("HopeBot: Your Mental Health Assistant 🤖")

# ---------------------------------------------------------------------------------------------------------------------------------------------
voice_label = "Turn off voice" if st.session_state.read_aloud_enabled else "Turn on voice"
if st.button(voice_label, key="voice_reply_toggle"):
    st.session_state.read_aloud_enabled = not st.session_state.read_aloud_enabled
    st.rerun()

float_init()

# History block
for message in st.session_state.messages:
    if message["role"] == "assistant":
        avatar = "⭐" if message.get("type") == "agent" else "🤖"
    else:
        avatar = "🤗"

    with st.chat_message(message["role"], avatar = avatar):
        if message.get("type") == "agent":
            st.markdown(f"<div style='font-size: 24px;'>{message['content']}</div>",
                        unsafe_allow_html=True)
        else:
            st.markdown(
                f"<p style='font-size: 24px; margin: 0;'>{message['content']}</p>",
                unsafe_allow_html=True
            )

if st.session_state.play_greeting and st.session_state.read_aloud_enabled:
    audio_file = text_to_speech("Thank you! Let's get started - how are you doing today?")
    autoplay_audio(audio_file)
    os.remove(audio_file)
    st.session_state.play_greeting = False

footer_container = st.container()
with footer_container:
    col1, col2 = st.columns([1, 12])
    with col1:
        audio_bytes = audio_recorder(
            text = "",
            icon_size = "1x",
            icon_name="microphone",
            energy_threshold=(-1, 0.5),
            pause_threshold=30,
            sample_rate = 30000)
    with col2:
        st.markdown("<p style='font-size: 15px; color: #0a0a0a; margin: 0; margin-left: -40px; '>Click the microphone to record your answer</p>",
                unsafe_allow_html=True)

typed_input = st.chat_input("Type your message here.")
user_message_parts = []

if typed_input and typed_input.strip():
    st.session_state.messages.append({"role": "user", "content": typed_input})
    st.rerun()

if audio_bytes:
    if "last_audio" not in st.session_state or st.session_state.last_audio != audio_bytes:
        st.session_state.last_audio = audio_bytes

    with st.spinner("Transcribing..."):
        audio_path = "temp_audio.mp3"
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)

        transcript = speech_to_text(audio_path)
        if transcript:
            st.session_state.messages.append({"role": "user", "content": transcript})
            display_text(transcript)
            os.remove(audio_path)
            st.rerun()

# ---------------------------------------------------------------------------------------------------------------------------------------------
# Conversation loop
if st.session_state.participant_id is None:
    if st.session_state.messages[-1]["role"] == "user":
        st.session_state.participant_id =st.session_state.messages[-1]["content"].strip()

        greeting = "Thank you! Let's get started — what's been on your mind lately? 😊"
        st.session_state.messages.append({
            "role": "assistant",
            "content": greeting
        })
        st.session_state.play_greeting = True
        st.rerun()
# Main conversation loop - runs once participant_id is captured
elif st.session_state.messages[-1]["role"] != "assistant":
    user_message = st.session_state.messages[-1]["content"].strip().lower()
    if st.session_state.get("agent_ran") and "end" in user_message.split() and not st.session_state.closing_check:
        survey_link = "https://forms.cloud.microsoft/e/H5jHLrApqF"
        researcher_email = "rachel.lau.25@ucl.ac.uk"

        closing_message = (
            f"Thank you for completing the PHQ-9 with me today. I hope the information "
            f"and resources shared have been helpful.<br>"
            f"Please remember that support is always available.<br><br>You can reach the "
            f"Samaritans anytime on 116 123 (free, 24/7) or by email at jo@samaritans.org.<br><br>"
            f"To complete the study, please fill out the self-assessed PHQ-9 and feedback "
            f"survey here: <a href='{survey_link}'>{survey_link}</a>.<br>Your participant ID is {st.session_state.participant_id}.<br>If you have any issues, please contact the "
            f"researcher at {researcher_email}.<br><br>"
            f"Take care. I'm here if you'd like to keep talking.🌿<br><br>Warm regards,<br>HopeBot" 
        )
        
        st.session_state.closing_check = True
        st.session_state.messages.append({
            "role": "assistant",
            "content": closing_message,
            "type": "hopebot"
        })
        st.rerun()
    elif st.session_state.get("agent_ran"):
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking 🤔..."):
                continued_messages = [
                    HumanMessage(content=m["content"]) 
                    if m["role"] == "user" 
                    else AIMessage(content=m["content"])
                    for m in st.session_state.messages
                ]
                continued_result = st.session_state.agent_app.invoke(
                    {"messages": continued_messages},
                    config = {
                        "metadata": {"participant_id": st.session_state.participant_id},
                        "tags": [f"participant-{st.session_state.participant_id}"]
                        })
                continued_response = extract_agent_responses(continued_result)
        
            if continued_response:
                st.markdown(f"<div style='font-size: 24px;'>{continued_response}</div>",
                            unsafe_allow_html=True)
                if st.session_state.read_aloud_enabled:
                    with st.spinner("HopeBot is speaking 💬..."):
                            audio_file = text_to_speech(continued_response)
                    autoplay_audio(audio_file)
                    os.remove(audio_file)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": continued_response,
                    "type": 'agent'
                })
    # Otherwise HopeBot handles it
    else:
        with st.chat_message("assistant", avatar="🤖"):
            # Step 1: HopeBot generates response
            with st.spinner("Thinking 🤔..."):
                responses, openai_messages = get_assistant_response(st.session_state.messages)

            # Extract the message object
            message = responses.choices[0].message

            # Get display text (replaces cleaned_text)
            display_messages = message.content or ""
            
            if message.tool_calls:
                tool_call = message.tool_calls[0]
                data = json.loads(tool_call.function.arguments)
                q_num = data['question_answer']

                # Only record if this question hasn't been recorded yet
                if q_num not in st.session_state.recorded_question_numbers:
                    st.session_state.recorded_question_numbers.append(q_num)
                    st.session_state.answers_record.append(data['answer_category'])
                    st.session_state.phq9_scores_by_question.append(data["score"])
                    print(f"RECORDED: Q{q_num} = {data['answer_category']}")
                else:
                    idx = st.session_state.recorded_question_numbers.index(q_num)
                    st.session_state.answers_record[idx] = data['answer_category']
                    st.session_state.phq9_scores_by_question[idx] = int(data['score'])
                    print(f"UPDATED: Q{q_num} = {data['answer_category']}")
                
                st.session_state.total_phq9_score = sum(st.session_state.phq9_scores_by_question)

                if data.get('inferred') and q_num not in st.session_state.inferred_answers:
                    st.session_state.inferred_answers.append(q_num)
            
                if data.get('inferred'):
                    st.session_state.awaiting_confirmation = True
                else:
                    st.session_state.awaiting_confirmation = False

                # Send tool result back so model asks next question
                tool_result_messages = openai_messages + [
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments
                            }
                        }]
                    },
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": "Answer recorded successfully."
                    }
                ]

                follow_up = openai.chat.completions.create(
                    model="gpt-4o",
                    temperature=0.4,
                    messages=tool_result_messages,
                    tools=tools,
                    tool_choice="none"
                )

                display_messages = follow_up.choices[0].message.content or ""
            
            if not message.tool_calls:
                st.session_state.awaiting_confirmation = False
            
            # Display HopeBot response
            if display_messages:
                # Display text
                display_text(display_messages)

                if st.session_state.read_aloud_enabled:
                    with st.spinner("HopeBot is speaking 💬..."):
                        audio_file = text_to_speech(display_messages)  # Generate audio in advance
                    autoplay_audio(audio_file)  # Play audio
                    os.remove(audio_file)

                # Add response to session state
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": display_messages,
                    "type": "hopebot"})
            # Agent fires immediately after, appends its own message
            if phq9_complete() and not st.session_state.get("agent_ran") and not st.session_state.get('awaiting_confirmation'):
                try:
                    summary_text = build_score_summary()
                    with st.chat_message("assistant", avatar="🤖"):
                        display_text(summary_text)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": summary_text
                        })

                    screening_data = {
                        "email": None,
                        "score": st.session_state.total_phq9_score,
                        "assessment_type": "PHQ-9",
                        "question_9": st.session_state.phq9_scores_by_question[8]
                    }
                    
                    agent_results, agent_app = run_pipeline(screening_data, participant_id = st.session_state.participant_id)
                    st.session_state.agent_app = agent_app
                    agent_message = extract_agent_responses(agent_results)

                    with st.chat_message("assistant", avatar="⭐"):
                        st.markdown(f"<div style='font-size: 24px;'>{agent_message}</div>", unsafe_allow_html=True)

                        if st.session_state.read_aloud_enabled:
                            with st.spinner("HopeBot is speaking 💬..."):
                                audio_file = text_to_speech(agent_message)
                            autoplay_audio(audio_file)
                            os.remove(audio_file)
                    
                    st.session_state.agent_results = agent_results
                    st.session_state.agent_ran = True
                    st.session_state.messages.append({
                        'role': 'assistant',
                        'content': agent_message,
                        'type': 'agent'
                    })
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    st.error(f"Agent error: {e}")

# Floating microphone button
footer_container.float("bottom: 1rem; z-index: 9999;")

