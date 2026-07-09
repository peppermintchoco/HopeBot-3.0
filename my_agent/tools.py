from langchain_core.tools import tool

# Tool 1: Email function
import os.path

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

@tool
def send_email(recipient: str, subject: str, body: str, attachment_path:str = None) -> str:
    """Send an email to the user with the provided subject and body content."""

    gmail_address = os.getenv('GMAIL_ADDRESS')
    gmail_password = os.getenv('GMAIL_APP_PASSWORD')
    
    if not gmail_address:
            try:
                import streamlit as st
                gmail_address = st.secrets.get('GMAIL_ADDRESS')
                gmail_password = st.secrets.get('GMAIL_APP_PASSWORD')
            except:
                pass
                
    try:
        message = MIMEMultipart()
        message['From'] = gmail_address
        message['To'] = recipient
        message['Subject'] = subject

        message.attach(MIMEText(body, 'html'))

        if attachment_path and os.path.exists(attachment_path):
            with open(attachment_path, 'rb') as f:
                part = MIMEBase('text', 'calendar')
                part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(attachment_path)}')
                message.attach(part)

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(gmail_address, gmail_password)
        server.send_message(message)
        server.quit()
        
        return f"Email sent to {recipient} with subject: {subject}"
    except Exception as error:
        return f"An error occurred: {error}"

# Tool 2: Calendar Function
from datetime import datetime, timedelta
from icalendar import Calendar, Event

@tool
def calendar_input(recipient: str, appointment_datetime: datetime, session_name: str) -> str:
    """Takes in appointment details and generates a .ics file for user to add to their calendar."""
    event = Event()
    calendar = Calendar()

    event.add('summary', session_name)
    event.add('dtstart', appointment_datetime)
    event.add('dtend', appointment_datetime + timedelta(hours = 1))
    event.add('description', f'Appointment for {recipient}')

    calendar.add_component(event)

    filename = f"{session_name.replace(' ', '_')}.ics"
    with open(filename, 'wb') as f:
        f.write(calendar.to_ical())

    return f"Calendar file '{filename}' created for {session_name} on {appointment_datetime}. Please download and open to add to your calendar."


# Tool 3: Personalized psychoeducation function
PSYCHOEDUCATION_CONTENT = {
    "PHQ-9": {
    "Minimal": {
        "title": "Understanding Your PHQ-9 Results",
        "explanation": "Your results suggest minimal symptoms of depression.",
        "what_this_means": "It is normal to experience low mood occasionally. Many people experience brief periods of low mood, this is a common part of life and does not necessarily indicate clinical concern",
        "self_care_tips": {
            "Lifestyle": [
                "Maintain regular sleep patterns",
                "Stay physically active - any form of physical activity can enhance your sense of well-being",
                "Maintain a healthy diet and stay hydrated"
            ],
            "Social": [
                "Keep connected with friends and family"
            ]
        },
        "psychoeducational_resources": ["NHS Every Mind Matters: https://www.nhs.uk/every-mind-matters/"],
        "disclaimer": "HopeBot is not a diagnostic tool and it is recommended to seek professional advice."
    },
    "Mild": {
        "title": "Understanding Your PHQ-9 Results",
        "explanation": "Your results suggest mild symptoms of depression.",
        "what_this_means": "It is normal to experience some mild depressive symptoms. While it may cause some difficulty in daily activities, these symptoms are normally manageable and many people find improvement with small, proactive steps.",
        "self_care_tips": {
            "Lifestyle": [
                "Maintain regular sleep patterns",
                "Stay physically active - any form of physical activity can enhance your sense of well-being",
                "Maintain a healthy diet and stay hydrated"
            ],
            "Social": [
                "Keep connected with friends and family",
                "Talk to someone you trust about how you are feeling - sharing experiences can help even when it feels hard."
            ],
            "Coping Techniques": [
                "Set aside ample time for self-care - do things that make you feel good.",
                "Learn to notice unhelpful thought patterns - challenging and replacing negative thoughts is a good way to boost your mood.",
                "Use a planner to schedule enjoyable activities in advance."
            ]
        },
        "recommended_interventions": ["Guided self-help - including materials following principles of structured CBT", "Support from a trained practitioner"],
        "when_to_seek_help": "If symptoms persist for more than two weeks, consider seeing a trained professional and arranging for further assessment. Consider self-monitoring and retaking the assessment in a few weeks - using the HopeBot",
        "psychoeducational_resources": ["NHS Every Mind Matters: https://www.nhs.uk/every-mind-matters/", "NICE Guidelines for Depression in Adults: https://www.nice.org.uk/guidance/ng222"],
        "disclaimer": "HopeBot is not a diagnostic tool and it is recommended to seek professional advice."
    },
    "Moderate": {
        "title": "Understanding Your PHQ-9 Results",
        "explanation": "Your results suggest moderate symptoms of depression",
        "what_this_means": "It suggests that you may be experiencing moderate depressive symptoms that may be starting to affect your daily life and well-being.",
        "self_care_tips": {
            "Lifestyle": [
                "Maintain regular sleep patterns",
                "Stay physically active - any form of physical activity can enhance your sense of well-being",
                "Maintain a healthy diet and stay hydrated",
                "Limit alcohol intake as it can worsen mood over time."
            ],
            "Social": [
                "Keep connected with friends and family",
                "Talk to someone you trust about how you are feeling - sharing experiences can help even when it feels hard.",
                "Try spending time in nature or volunteering - both can help improve mood and sense of purpose."
            ],
            "Coping Techniques": [
                "Set aside ample time for self-care - do things that make you feel good.",
                "Learn to notice unhelpful thought patterns - challenging and replacing negative thoughts is a good way to boost your mood.",
                "Use a planner to schedule enjoyable activities in advance."
            ],
            "Monitoring": [
                "Keep a mood journal to track patterns and identify triggers."
            ]
        },
        "recommended_interventions": ["Consider seeking professional help, a mental health professional will be able to guide you on the best course of action. NICE guidelines suggest guided self-help and/or group CBT"],
        "when_to_seek_help": "Consider seeing a mental health professional especially if it affects your daily functioning and quality of life.",
        "psychoeducational_resources": ["NHS Every Mind Matters: https://www.nhs.uk/every-mind-matters/", "NICE Guidelines for Depression in Adults: https://www.nice.org.uk/guidance/ng222", "Mental Health Foundation Self-Care Tips: https://www.mentalhealth.org.uk/explore-mental-health/blogs/self-care-tips", "NHS talking therapies self-referral: https://www.nhs.uk/mental-health/talking-therapies-medicine-treatments/talking-therapies-and-counselling/nhs-talking-therapies/"],
        "disclaimer": "HopeBot is not a diagnostic tool and it is recommended to seek professional advice."
    },
    "Moderately Severe": {
        "title": "Understanding Your PHQ-9 Results",
        "explanation": "Your results suggest moderately severe symptoms of depression",
        "what_this_means": "It suggests that you may be experiencing more severe depressive symptoms that may be interfering with your daily functioning and quality of life.",
        "self_care_tips": {
            "Lifestyle": [
                "Maintain regular sleep patterns",
                "Stay physically active - any form of physical activity can enhance your sense of well-being",
                "Maintain a healthy diet and stay hydrated",
                "Limit alcohol intake as it can worsen mood over time."
            ],
            "Social": [
                "Keep connected with friends and family",
                "Talk to someone you trust about how you are feeling - sharing experiences can help even when it feels hard.",
                "Try spending time in nature or volunteering - both can help improve mood and sense of purpose."
            ],
            "Coping Techniques": [
                "Set aside ample time for self-care - do things that make you feel good.",
                "Learn to notice unhelpful thought patterns - challenging and replacing negative thoughts is a good way to boost your mood.",
                "Use a planner to schedule enjoyable activities in advance.",
                "Break tasks into smaller steps and space them across the day to reduce feeling overwhelmed."
            ],
            "Monitoring": [
                "Keep a mood journal to track patterns and identify triggers."
            ],
            "Recovery Support": [
                "Be kind to yourself about what you can manage right now - appreciate every small step you take."
            ]
        },
        "recommended_interventions": ["We recommend seeking professional help to discuss the best course of action with a mental health professional. NICE guidelines suggest individual CBT (16 sessions), behavioural activation (12-16 sessions), antidepressant medication, or a combination of CBT and antidepressant."],
        "when_to_seek_help": "We recommend speaking to a mental health professional.",
        "psychoeducational_resources": ["NHS Every Mind Matters: https://www.nhs.uk/every-mind-matters/", "NICE Guidelines for Depression in Adults: https://www.nice.org.uk/guidance/ng222", "Mental Health Foundation Self-Care Tips: https://www.mentalhealth.org.uk/explore-mental-health/blogs/self-care-tips", "NHS talking therapies self-referral: https://www.nhs.uk/mental-health/talking-therapies-medicine-treatments/talking-therapies-and-counselling/nhs-talking-therapies/"],
        "disclaimer": "HopeBot is not a diagnostic tool and it is recommended to seek professional advice."
    },
    "Severe": {
        "title": "Understanding Your PHQ-9 Results",
        "explanation": "Your results suggest severe symptoms of depression",
        "what_this_means": "It suggests that you may be experiencing severe depressive symptoms that may be significantly disrupting your daily activities and quality of life",
        "self_care_tips": {
            "Lifestyle": [
                "Maintain regular sleep patterns",
                "Stay physically active - any form of physical activity can enhance your sense of well-being",
                "Maintain a healthy diet and stay hydrated",
                "Avoid using drugs or alcohol to cope with difficult feelings - in the long run they can make you feel a lot worse"
            ],
            "Social": [
                "Keep connected with friends and family",
                "Talk to someone you trust about how you are feeling - sharing experiences can help even when it feels hard.",
                "Try spending time in nature or volunteering - both can help improve mood and sense of purpose."
            ],
            "Coping Techniques": [
                "Set aside ample time for self-care - do things that make you feel good.",
                "Learn to notice unhelpful thought patterns - challenging and replacing negative thoughts is a good way to boost your mood.",
                "Use a planner to schedule enjoyable activities in advance."
            ],
            "Monitoring": [
                "Keep a mood journal to track patterns and identify triggers."
            ],
            "Recovery Support": [
                "Focus on basic physical health even when it feels hard - getting good sleep, thinking about your diet and gentle movement can make a difference.",
                "Lean on your support network - identify people you can reach out to on difficult days.",
                "Maintain a consistent daily routine - even small structure helps when motivation is low."
            ]
        },
        "recommended_interventions": ["We strongly encourage seeking professional help as soon as possible - a mental health professional will be able to guide you on the best course of action. NICE guidelines suggest individual CBT (16 sessions), behavioural activation (12-16 sessions), antidepressant medication, or a combination of CBT and antidepressant."],
        "when_to_seek_help": "We strongly recommend speaking with a mental health professional to discuss the treatment options that are right for you",
        "psychoeducational_resources": ["NHS Every Mind Matters: https://www.nhs.uk/every-mind-matters/", "NICE Guidelines for Depression in Adults: https://www.nice.org.uk/guidance/ng222", "Mental Health Foundation Self-Care Tips: https://www.mentalhealth.org.uk/explore-mental-health/blogs/self-care-tips", "NHS talking therapies self-referral: https://www.nhs.uk/mental-health/talking-therapies-medicine-treatments/talking-therapies-and-counselling/nhs-talking-therapies/"],
        "disclaimer": "HopeBot is not a diagnostic tool and it is recommended to seek professional advice."
    }
    },
    "GAD-7": {
    "Minimal": {
        "title": "Understanding Your GAD-7 Results",
        "explanation": "Your results suggest minimal symptoms of general anxiety.",
        "what_this_means": "It is normal to experience anxious moods every once in awhile in everyday situations. Many people experience occasional anxiety - this is a common part of life and does not necessarily indicate clinical concern.",
        "self_care_tips": {
            "Lifestyle": [
                "Maintain regular sleep patterns",
                "Stay physically active - aim for 20-30 minutes of outdoor exercise, 3 times a week if possible"
            ],
            "Social": [
                "Keep connected with friends and family"
            ],
            "Coping Techniques": [
                "Try mindfulness and meditation, including breathing exercises and relaxation, to calm anxiety by focusing on the present moment."
            ]
        },
        "psychoeducational_resources": ["NHS Every Mind Matters: https://www.nhs.uk/every-mind-matters/", "NICE Guidelines for Anxiety in Adults: https://www.nice.org.uk/guidance/cg113/"],
        "disclaimer": "HopeBot is not a diagnostic tool and it is recommended to seek professional advice."
    },
    "Mild": {
        "title": "Understanding Your GAD-7 Results",
        "explanation": "Your results suggest mild symptoms of anxiety.",
        "what_this_means": "It is common to experience mild symptoms of anxiety especially during stressful periods - though these symptoms typically go away with time and/or some positive low-intensity interventions.",
        "self_care_tips": {
            "Lifestyle": [
                "Maintain regular sleep patterns",
                "Stay physically active - aim for 20-30 minutes of outdoor exercise, 3 times a week if possible"
            ],
            "Social": [
                "Keep connected with friends and family"
            ],
            "Coping Techniques": [
                "Try mindfulness and meditation, including breathing exercises and relaxation, to calm anxiety by focusing on the present moment.",
                "Try using the 5,4,3,2,1 grounding method - focus on 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, and 1 you can taste.",
                "Try doing something creative like drawing to help distract from difficult thoughts."
            ],
            "Avoidance & Habits": [
                "Try not to avoid situations that make you anxious - avoidance can worsen problems and make such situations more intimidating."
            ]
        },
        "recommended_interventions": ["Consider seeking one or more of the following low intensity interventions: individual non-facilitated self-help, individual guided self-help or psychoeducational groups. NICE guidelines recommend guided self-help (usually 5 to 7 sessions of 20-30 minutes) or psychoeducational groups (usually 6 weekly sessions of 2 hours)."],
        "when_to_seek_help": "Monitor symptoms and if symptoms persist for more than two weeks, consider seeking professional help.",
        "psychoeducational_resources": ["NHS Every Mind Matters: https://www.nhs.uk/every-mind-matters/", "NICE Guidelines for Anxiety in Adults: https://www.nice.org.uk/guidance/cg113/", "NHS Inform Anxiety Self-Help Guide: https://www.nhsinform.scot/illnesses-and-conditions/mental-health/mental-health-self-help-guides/anxiety-self-help-guide/", "Mind UK Self-Care for Anxiety: https://www.mind.org.uk/information-support/types-of-mental-health-problems/anxiety-and-panic-attacks/self-care/"],
        "disclaimer": "HopeBot is not a diagnostic tool and it is recommended to seek professional advice."
    },
    "Moderate": {
        "title": "Understanding Your GAD-7 Results",
        "explanation": "Your results suggest moderate symptoms of anxiety.",
        "what_this_means": "You are not alone in experiencing these symptoms of anxiety and it may be affecting your daily functioning and well-being. With the appropriate interventions, these symptoms can be properly managed and reduced.",
        "self_care_tips": {
            "Lifestyle": [
                "Maintain regular sleep patterns",
                "Stay physically active - aim for 20-30 minutes of outdoor exercise, 3 times a week if possible"
            ],
            "Social": [
                "Keep connected with friends and family"
            ],
            "Coping Techniques": [
                "Try mindfulness and meditation, including breathing exercises and relaxation, to calm anxiety by focusing on the present moment.",
                "Try using the 5,4,3,2,1 grounding method - focus on 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, and 1 you can taste.",
                "Try doing something creative like drawing to help distract from difficult thoughts.",
                "Practice calming breathing techniques regularly as part of your daily routine.",
                "Make a digital self-care kit on your phone with photos, music, videos, messages or sayings that are helpful or notes that remind you of management techniques for difficult situations."
            ],
            "Avoidance & Habits": [
                "Try not to avoid situations that make you anxious - avoidance can worsen problems and make such situations more intimidating.",
                "Limit caffeine and alcohol intake as both can worsen anxiety symptoms"
            ]
        },
        "recommended_interventions": ["It is recommended to seek high-intensity professional help. NICE recommends either a talking therapy (CBT or applied relaxation, usually 12-15 weekly sessions) or medication - the choice is yours based on your preference."],
        "when_to_seek_help": "It is recommended to seek professional help - together with the help of a mental health professional, the most appropriate and effective intervention can be chosen",
        "psychoeducational_resources": ["NHS Every Mind Matters: https://www.nhs.uk/every-mind-matters/", "NICE Guidelines for Anxiety in Adults: https://www.nice.org.uk/guidance/cg113/", "NHS Inform Anxiety Self-Help Guide: https://www.nhsinform.scot/illnesses-and-conditions/mental-health/mental-health-self-help-guides/anxiety-self-help-guide/", "Mind UK Self-Care for Anxiety: https://www.mind.org.uk/information-support/types-of-mental-health-problems/anxiety-and-panic-attacks/self-care/", "NHS Breathing Exercises for Stress: https://www.nhs.uk/mental-health/self-help/guides-tools-and-activities/breathing-exercises-for-stress/"],
        "disclaimer": "HopeBot is not a diagnostic tool and it is recommended to seek professional advice."
    },
    "Severe": {
        "title": "Understanding Your GAD-7 Results",
        "explanation": "Your results suggest severe symptoms of anxiety.",
        "what_this_means": "You are not alone in experiencing these symptoms of anxiety and it is likely that you are experiencing severe anxiety with marked functional impairments that significantly affect daily functioning and well-being. With the appropriate interventions, these symptoms can be properly managed and reduced.",
        "self_care_tips": {
            "Lifestyle": [
                "Maintain regular sleep patterns",
                "Stay physically active - aim for 20-30 minutes of outdoor exercise, 3 times a week if possible",
                "Remember to prioritise self-care basics (sleep, nutrition, hydration) even when motivation is low!"
            ],
            "Social": [
                "Keep connected with friends and family",
                "Lean on your support network - identify specific people you can call when anxiety feels unmanageable"
            ],
            "Coping Techniques": [
                "Try mindfulness and meditation, including breathing exercises and relaxation, to calm anxiety by focusing on the present moment.",
                "Try using the 5,4,3,2,1 grounding method - focus on 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, and 1 you can taste.",
                "Try doing something creative like drawing to help distract from difficult thoughts.",
                "Practice calming breathing techniques regularly as part of your daily routine.",
                "Make a digital self-care kit on your phone with photos, music, videos, messages or sayings that are helpful or notes that remind you of management techniques for difficult situations."
            ],
            "Avoidance & Habits": [
                "Try not to avoid situations that make you anxious - avoidance can worsen problems and make such situations more intimidating.",
                "Limit caffeine and alcohol intake as both can worsen anxiety symptoms"
            ],
            "Crisis Management": [
                "If you're feeling overwhelmed, try interrupting your senses - changing your environment, making the room cooler, or changing the lighting can help"
            ]
        },
        "recommended_interventions": ["It is strongly recommended to seek high-intensity professional support.", "NICE guidelines recommend either high intensity psychological treatment such as CBT or applied relaxation, usually consisting of 12 to 15 weekly sessions each lasting 1 hour and/or drug treatment", "Consider a referral to specialist mental health services for a comprehensive assessment of your needs."],
        "when_to_seek_help": "It is strongly recommended to seek professional help as soon as possible - with the help of a mental health professional the best combination of psychological and drug treatments can be utilized to ensure best outcome.",
        "psychoeducational_resources": ["NHS Every Mind Matters: https://www.nhs.uk/every-mind-matters/", "NICE Guidelines for Anxiety in Adults: https://www.nice.org.uk/guidance/cg113/", "NHS Inform Anxiety Self-Help Guide: https://www.nhsinform.scot/illnesses-and-conditions/mental-health/mental-health-self-help-guides/anxiety-self-help-guide/", "Mind UK Self-Care for Anxiety: https://www.mind.org.uk/information-support/types-of-mental-health-problems/anxiety-and-panic-attacks/self-care/", "NHS Breathing Exercises for Stress: https://www.nhs.uk/mental-health/self-help/guides-tools-and-activities/breathing-exercises-for-stress/", "NHS Talking Therapies Self-Referral: https://www.nhs.uk/mental-health/talking-therapies-medicine-treatments/talking-therapies-and-counselling/nhs-talking-therapies/"],
        "disclaimer": "HopeBot is not a diagnostic tool and it is recommended to seek professional advice."
    }
    },
    "MDQ": {
    "Positive": {
        "title": "Understanding Your MDQ Results",
        "explanation": "Your results suggest a positive screening for bipolar disorder.",
        "what_this_means": "A positive screen suggests it would be helpful to explore your experiences further with a professional to better understand what you may be going through.",
        "self_care_tips": {
            "Lifestyle": [
                "Establish a consistent daily routine for stability - including regular times for meals, sleep, relaxation, hobbies and social plans.",
                "Prioritise good sleep.",
                "Eat a balanced and nutritious diet",
                "Try gentle exercises like yoga to help relax and manage stress."
            ],
            "Monitoring": [
                "Learn to recognize mood patterns and manage stress where possible."
            ],
            "Avoidance": [
                "Avoid using drugs or alcohol to cope as they can exacerbate symptoms."
            ]
        },
        "recommended_interventions": ["It is recommended that you should seek further clinical evaluation with a psychiatrist or GP."],
        "when_to_seek_help": "We recommend speaking with your GP to discuss your screening results and arrange a comprehensive assessment.",
        "psychoeducational_resources": ["NHS Every Mind Matters: https://www.nhs.uk/every-mind-matters/", "NHS Bipolar Disorder Overview: https://www.nhs.uk/mental-health/conditions/bipolar-disorder/", "NICE CG185 Bipolar Disorder Guideline: https://www.nice.org.uk/guidance/cg185", "Mind UK Self-Managing Bipolar: https://www.mind.org.uk/information-support/types-of-mental-health-problems/bipolar-disorder/self-managing-bipolar/", "NHS Talking Therapies Self-Referral: https://www.nhs.uk/mental-health/talking-therapies-medicine-treatments/talking-therapies-and-counselling/nhs-talking-therapies/"],
        "disclaimer": "HopeBot is not a diagnostic tool and it is recommended to seek professional advice. MDQ is a screening flag and not a diagnostic tool, further assessment by a mental health professional is needed."
    },
    "Negative": {
        "title": "Understanding Your MDQ Results",
        "explanation": "It is normal to experience mood changes from time to time and this does not necessarily indicate a clinical concern. However, if you have any ongoing concerns, speaking with a professional is always worthwhile.",
        "what_this_means": "Your screening did not flag indicators for bipolar disorder. It is normal to experience mood changes from time to time and it does not necessarily indicate a clinical concern.",
        "self_care_tips": [
            "Maintain regular sleep patterns - aim to wake up and go to bed at the same time each day.",
            "Stay physically active - aim for 20-30 minutes of exercise 3 times a week.",
            "Maintain a balanced and healthy diet.",
            "Keep connected with friends and family.",
            "Be aware that it is normal to experience mood changes from time to time.",
            "Monitor your mood - if you notice persistent or unusual mood changes, consider speaking to a professional."
        ],
        "psychoeducational_resources": ["NHS Every Mind Matters: https://www.nhs.uk/every-mind-matters/", "NHS Bipolar Disorder Overview: https://www.nhs.uk/mental-health/conditions/bipolar-disorder/", "NICE CG185 (Bipolar Disorder guideline): https://www.nice.org.uk/guidance/cg185"],
        "disclaimer": "HopeBot is not a diagnostic tool. While your screening did not flag concerns, if you experience persistent mood changes, please consult a professional."
    }
    }
}

@tool
def psychoeducation(recipient: str, assessment_type: str, severity: str) -> str:
    """Return relevant mental health psychoeducational content from a predefined dictionary as a guide 
    tailored to the needs of the user based on the mental health assessment taken and the severity level."""
    
    assessment_content = PSYCHOEDUCATION_CONTENT.get(assessment_type, {})

    content = assessment_content.get(severity)

    if not content:
        for key in assessment_content:
            if key.lower() == severity.lower():
                content = assessment_content[key]
                break

    if not content:
        return f"No psychoeducational content available for this {assessment_type} - {severity}."
    
    return str(content)

# Tool 4: Pre-appointment preparation function
SESSION_PREPARATION = {
    "PRE": {
        "title": "Preparing for your first session",
        "preparation": ['Think about and write down your reasons for seeking therapy and what you hope to achieve', 'Prepare any questions you have about the therapy approach and process', 'Jot down notes or reflect beforehand', 'Check practical details before the session (location, time, what to bring)', 'Try to set realistic expectations.', 'Understand the type of therapy you are getting.'],
        "resources": ["Mind UK How to get the most of therapy: https://www.mind.org.uk/information-support/drugs-and-treatments/talking-therapy-and-counselling/getting-the-most-from-therapy/#TipsForAllTypesOfTherapySessions"]
    },
    "ONGOING": {
        "title": "How to prepare for your therapy session.",
        "preparation": ["Share how you are feeling right at the start", "Be open and honest about what is working and what isn't", "Focus on what matetrs most to you."],
        "resources": ["Mind UK How to get the most of therapy: https://www.mind.org.uk/information-support/drugs-and-treatments/talking-therapy-and-counselling/getting-the-most-from-therapy/#TipsForAllTypesOfTherapySessions"]
    },
    "BETWEEN": {
        "title": "How to prepare between therapy sessions.",
        "preparation": ["Complete any tasks or exercises agreed with your therapist.", "Write down important thoughts, feelings or situations that come up between sessions.", "Try to attend all appointments and be on time", "Keep your therapist updated on any changes in your health or circumstances.", "Try to set and maintain realistic expectations for therapy."],
        "resources": ["Mind UK How to get the most of therapy: https://www.mind.org.uk/information-support/drugs-and-treatments/talking-therapy-and-counselling/getting-the-most-from-therapy/#TipsForAllTypesOfTherapySessions"]
    }
}

@tool
def session_prep(recipient: str, therapy_stage: str = "PRE") -> str:
    """Prepare therapy session preparation material for the user pulled from a structured resource bank based on the user's current stafe in their therapy journey."""
    
    content = SESSION_PREPARATION.get(therapy_stage, {})

    if not content:
        return "No relevant session preparation material is avaailable for this combination"
    
    return str(content)
