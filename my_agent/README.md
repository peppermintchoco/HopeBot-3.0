# AI Agent for HopeBot

This repository contains the code for the AI Agent extension of the HopeBot for dissertation testing.

## Overview
The agent handles post-screening mental health care coordination, taking over after HopeBot completes a PHQ-9, GAD-7, or MDQ assessment. It provides personalised psychoeducation, session preparation materials, email communication, and calendar booking.

## Files
- `agent.py` - Main agent pipeline with triage, routing, and LangGraph conversation flow
- `tools.py` - Four tool functions (email, psychoeducation, session preparation, and calendar)
- `handoff.py` - Bridges HopeBot's scoring output to agent's input format

## Setup
1. Clone the repository
2. Create a virtual environment: `python -m venv venv` (If you already have a virtual environment for HopeBot, you may go straight to step 4)
3. Activate it: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Add your `.env` file with your OpenAI API key: `OPENAI_API_KEY=your_key_here`
6. Add your Google OAuth `credentials.json` for email functionality

## Integration with HopeBot
Call `handoff_to_agent(state)` from `handoff.py` after `state['completed'] = True` in `questionnaire_flow.py`. This happens in two places: after PHQ-9/GAD-7 scoring and after MDQ scoring.

## Tools
- **Email** - Sends assessment results and resources via Gmail API
- **Psychoeducation** - Returns NICE-guideline-aligned content tailored to assessment type and severity
- **Session Preparation** - Provides therapy preparation materials based on the user's stage in therapy
- **Calendar** - Generates .ics files for appointment scheduling
