import streamlit as st
import os
from audio_recorder_streamlit import audio_recorder
from streamlit_float import float_init
import base64
import sys
import json
import chardet

# --------------------------------------------------------------------------------------------------------------------------logic2END
import openai
import os
from dotenv import load_dotenv
import base64
import streamlit as st
import openai
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
# import pysqlite3 as sqlite3

from my_agent.agent import run_pipeline

st.set_page_config(page_title="HopeBot: Your Mental Health Assistant", layout="wide")
# sys.modules["sqlite3"] = sqlite3
load_dotenv()
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Define system prompt
SYSTEM_PROMPT = """
You are HopeBot, a professional psychotherapist specialising in Cognitive Behavioural Therapy. Your role is to focus on your clients' words and emotions, guiding them to reflect on their thoughts and behaviours through open-ended questions and guiding them through the PHQ-9 test. Always show empathy and understanding of their feelings and help them to recognise how their behaviour affects their emotions. Your responses should not be too long or presented in bullet point form, and all your responses should be spoken. You need to focus on listening, encourage clients to express themselves through short and precise language, and help them sort out and explore their emotions and thoughts. If a customer comes to you for advice, give up to 2 at a time. You need to provide helpful advice and assistance to users when they are experiencing extreme emotions, and start by adding encouraging sentences such as "You don't have to face this alone." 

    You must complete three tasks in turn:
    Task 1: Start by warmly greeting the client and creating a comfortable space for conversation. As a professional counselor, your goal is to listen attentively and engage in a natural flow of dialogue. As the conversation progresses, pay close attention to what the client shares. If they indicate that they have nothing else to share, or if the dialogue reaches about 20 exchanges, you must smoothly transition to introducing the PHQ-9 questionnaire and ask the user if they would like to take the PHQ-9 test. When doing this, acknowledge and validate what the client has shared so far, emphasizing how valuable their input has been.
    Task 2: After the user agrees to use the PHQ-9, ask each question in turn. Accurately categorise the user's answers as options A, B, C or D. If the user's answer is not precise enough, ambiguous or cannot be accurately categorised, you must ask the user to provide a clearer answer to ensure that the most accurate answer is collected, and you will need to ensure that the user completes all of the questions in turn. If the user answers A, they get 0 points; B, 1 point; C, 2 points; and D, 3 points. Track the score cumulatively without displaying it, and move to Task 3 after completing the test.
    Task 3: You must first tell the user of their answer distribution. In the format: Here’s how each answer was interpreted: Question 1: X (X point), etc. Then sum each question's mark up, and tell the user of their total score in number on the PHQ-9. In the format: You scored X points. If the user skipped questions, you need to mention how many questions the user skipped in your summary. And provide the appropriate depression severity results. Be sure to make it clear that you are a virtual mental health assistant, not a doctor, and that whilst you will offer help, you are not a substitute for professional medical advice. Then let them know you are connecting them with HopeBot's care coordinator for further support and resources.
    At the end you will need to provide a brief summary of your conversation, including the confusion raised by the user in Task 1, as well as their PHQ-9 test results, and your corresponding recommendations. You need to ask the user if they have any further questions about the result and answer them.
    
    Please maintain the demeanour of a professional psychologist at all times and show empathy in your interactions. Please keep your responses concise and avoid giving long, repetitive answers.
    Here is some additional background information to help guide your responses:\n\n{context}
"""

# Define function calling for recording PHQ9 scores
tools = [{
    'type': 'function',
    'function': {
    'name': 'record_phq9_answer',
    'description': """ Call this function ONLY when you are confident that you can classify the user's PHQ-9 answer - 
    whether they choose an option explicitly or you inferred their answers from natural language.
    Do not call during clarification turns or when still explaining options.""",
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
    retrieved_docs1 = retriever1.get_relevant_documents(user_input)
    retrieved_docs2 = retriever2.get_relevant_documents(user_input)
    retrieved_docs3 = retriever3.get_relevant_documents(user_input)
    combined_context = "\n".join([
        doc.page_content for doc in 
        retrieved_docs1 + retrieved_docs2 + retrieved_docs3
    ])

    # Manually inject context into system prompt
    system_prompt = SYSTEM_PROMPT.replace("{context}", combined_context)

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
    return response

def extract_agent_responses(agent_results):
    messages = agent_results['messages']
    for message in reversed(messages):
        if hasattr(message, 'content') and message.content:
            return message.content
    return ""

def display_text(content):
    st.markdown(
        f"<p style='font-size: 24px; margin: 0;'>{content}</p>",
        unsafe_allow_html=True
    )

# ------------------------------------------------------------------------------------------------------------------------------------------------logic2END

# 初始化会话状态
def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "This is HopeBot, your mental health assistant. It's good to hear from you, how are you doing today? 😊"}
        ]
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

initialize_session_state()

# 标题
st.title("HopeBot: Your Mental Health Assistant 🤖")

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
        response.stream_to_file(audio_path)
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

# ------------------------------------------------------------------------------------------------------------------------------------------------

# Function to trigger when phq-9 assessment is completed
PHQ9_TOTAL_QUESTIONS = 9

def phq9_complete():
    return len(st.session_state.answers_record) == PHQ9_TOTAL_QUESTIONS

# 浮动容器（用于麦克风）
float_init()
footer_container = st.container()
with footer_container:
    audio_bytes = audio_recorder(energy_threshold=(-1, 0.5), pause_threshold=30, sample_rate = 30000)

# 显示聊天历史（使用气泡样式和头像）
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🤖" if message["role"] == "assistant" else "🤗"):
        st.markdown(
            f"<p style='font-size: 24px; margin: 0;'>{message['content']}</p>",
            unsafe_allow_html=True
        )

# (1) Input from user
typed_input = st.chat_input("Type your message here, or use the microphone.")
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

# (2) Generate HopeBot response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant", avatar="🤖"):
        
        # Step 1: HopeBot generates Task 3 response
        with st.spinner("Thinking 🤔..."):
            responses = get_assistant_response(st.session_state.messages)  # Generate text response

        # Extract the message object
        message = responses.choices[0].message

        # Get display text (replaces cleaned_text)
        display_messages = message.content or ""
        
        if message.tool_calls:
            tool_call = message.tool_calls[0]
            data = json.loads(tool_call.function.arguments)
            
            st.session_state.answers_record.append(data['answer_category'])
            st.session_state.total_phq9_score += int(data['score'])
            st.session_state.phq9_scores_by_question.append(data["score"])
        
            if data.get('inferred'):
                st.session_state.inferred_answers.append(data["question_answer"])
        
        # Step 2: Agent fires immediately after, appends its own message
        if phq9_complete() and not st.session_state.get("agent_ran"):
            screening_data = {
                "email": None,
                "score": st.session_state.total_phq9_score,
                "assessment_type": "PHQ-9",
                "question_9": st.session_state.phq9_scores_by_question[8]
            }

            agent_results = run_pipeline(screening_data)
            agent_message = extract_agent_responses(agent_results)
        
            st.session_state.agent_results = agent_results
            st.session_state.agent_ran = True
            st.session_state.messages.append({
                'role': 'assistant',
                'content': agent_message
            })

            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(
                    f"<p style='font-size: 24px; margin: 0;'>{agent_message}</p>",
                    unsafe_allow_html=True
                )
        
        with st.spinner("HopeBot is speaking 💬..."):
            audio_file = text_to_speech(display_messages)  # Generate audio in advance

        # Display text and play audio simultaneously
        display_text(display_messages)
        autoplay_audio(audio_file)  # Play audio

        # Add response to session state
        st.session_state.messages.append({"role": "assistant", "content": display_messages})
        os.remove(audio_file)

# Floating microphone button
footer_container.float("bottom: 0rem;")

