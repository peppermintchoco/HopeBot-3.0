from agent import run_pipeline

def build_screening_data(dtate, user_info = None):
    """Converts results from HopeBot screening into the screening_data format required by the agent pipeline."""

    user_info = user_info or {}

    result = state['result']
    tool = state['tool']

    if tool in ('PHQ-9', 'GAD-7'):
        screening_data = {
            "assessment_type": tool,
            "score": result["total"],
            "max_score": result["max_total"],
            "question_9": result.get("safety_item", {}).get("score", 0),
            "email": user_info.get("email", "")
        }
    else:
        screening_data = {
            "assessment_type": "MDQ",
            "score": "Positive" if result["positive_screen"] else "Negative",
            "email": user_info.get("email", "")
        }
    
    return screening_data

def handoff_to_agent(state, user_info):
    """Calls build_screening_data to convert HopeBot results into screening_data, then runs the agent pipeline."""
    
    screening_data = build_screening_data(state, user_info)
    
    return run_pipeline(screening_data)