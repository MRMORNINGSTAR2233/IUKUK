import os
import sys
import json
import shutil
import time
import re  # Added for robust JSON extraction
import concurrent.futures
from typing import Dict, Any, List

# Third-party imports
import gymnasium as gym
from groq import Groq
import google.generativeai as genai
from dotenv import load_dotenv
from mcp import StdioServerParameters
from colorama import init, Fore, Style

# Local imports
from mcp_gym import MCPEnv
from rewards import KeywordReward, LLMReward

# Initialize Colorama
init(autoreset=True)

# --- CONFIGURATION ---
load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
API_TIMEOUT = 30  # Seconds to wait before giving up on an AI model

# NEW MISSION: Requires Research + Action (Multi-step)
MISSION_GOAL = "Find the current price of Bitcoin and Ethereum. Then, save a summary to a file named 'crypto_report.txt'."

# UPDATED KEYWORDS (For the fallback critic)
SUCCESS_KEYWORDS = ["saved", "crypto_report.txt", "SUCCESS"]

# --- LOGGING HELPERS ---
def log_info(msg): print(f"{Fore.WHITE}{Style.DIM}{msg}{Style.RESET_ALL}", flush=True)
def log_thought(msg): print(f"\n{Fore.CYAN}🧠 MIND: {msg}{Style.RESET_ALL}", flush=True)
def log_action(msg): print(f"{Fore.YELLOW}⚡ ACT:  {msg}{Style.RESET_ALL}", flush=True)
def log_obs(msg): print(f"{Fore.BLUE}📜 OBS:  {msg}{Style.RESET_ALL}", flush=True)
def log_success(msg): print(f"\n{Fore.GREEN}{Style.BRIGHT}🎉 SUCCESS: {msg}{Style.RESET_ALL}", flush=True)
def log_error(msg): print(f"\n{Fore.RED}❌ ERROR: {msg}{Style.RESET_ALL}", flush=True)
def log_debug(msg): print(f"{Fore.MAGENTA}🐛 DEBUG: {msg}{Style.RESET_ALL}", flush=True)

# --- CORE AI LOGIC WITH TIMEOUTS ---

def _gemini_task(api_key: str, system_prompt: str, history: List[Dict]):
    """The actual blocking call to Gemini using REST API."""
    import requests
    
    # Use REST API with v1beta endpoint and gemini-2.5-flash model
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    
    # Build messages - Gemini expects alternating user/model roles
    contents = []
    
    # Combine system prompt with first user message
    first_message = system_prompt + "\n\nBegin mission."
    
    if history:
        # Add history
        for msg in history:
            role = "user" if msg["role"] == "user" else "model"
            contents.append({
                "role": role,
                "parts": [{"text": msg["content"]}]
            })
        # Prepend system instruction to first user message
        if contents and contents[0]["role"] == "user":
            contents[0]["parts"][0]["text"] = system_prompt + "\n\n" + contents[0]["parts"][0]["text"]
    else:
        # No history, just system prompt + start
        contents.append({
            "role": "user",
            "parts": [{"text": first_message}]
        })
    
    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": 0.0,
        }
    }
    
    response = requests.post(url, json=payload, timeout=API_TIMEOUT)
    response.raise_for_status()
    
    result = response.json()
    try:
        return result['candidates'][0]['content']['parts'][0]['text'].strip()
    except KeyError:
        return "Error: Gemini returned empty response or safety filter triggered."

def _call_gemini(api_key: str, system_prompt: str, history: List[Dict]) -> str:
    """Wrapper that enforces a timeout on Gemini."""
    log_debug(f"Contacting Gemini (Timeout: {API_TIMEOUT}s)...")
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(_gemini_task, api_key, system_prompt, history)
        try:
            result = future.result(timeout=API_TIMEOUT)
            log_debug(f"Gemini replied in {time.time() - start_time:.2f}s")
            return result
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Gemini took longer than {API_TIMEOUT}s")

def _groq_task(api_key: str, system_prompt: str, history: List[Dict]):
    """The actual blocking call to Groq."""
    client = Groq(api_key=api_key)
    messages = [{"role": "system", "content": system_prompt}] + history
    
    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
        temperature=0.0,
        stop=None
    )
    return completion.choices[0].message.content.strip()

def _call_groq(api_key: str, system_prompt: str, history: List[Dict]) -> str:
    """Wrapper that enforces a timeout on Groq."""
    log_debug(f"Contacting Groq (Timeout: {API_TIMEOUT}s)...")
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(_groq_task, api_key, system_prompt, history)
        try:
            result = future.result(timeout=API_TIMEOUT)
            log_debug(f"Groq replied in {time.time() - start_time:.2f}s")
            return result
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Groq took longer than {API_TIMEOUT}s")

def get_llm_action(tools_info: str, history: list) -> str:
    """Orchestrates the LLM call with fallback."""
    system_prompt = f"""
    You are an autonomous agent connected to a Tool Environment.
    
    AVAILABLE TOOLS:
    {tools_info}
    
    YOUR MISSION:
    {MISSION_GOAL}
    
    CRITICAL RULES:
    1. ALWAYS output ONLY valid JSON - NO other text, NO explanations, NO summaries.
    2. Even if you have information, you must call a tool (like save_report) to complete the mission.
    3. The JSON MUST follow this EXACT structure:
       {{
         "thought": "What I'm doing and why",
         "tool_name": "exact_tool_name_from_list",
         "args": {{ "arg_name": "value" }} 
       }}
    4. Never output plain text responses - you must always call a tool via JSON.
    5. If the mission has multiple steps (like search then save), do them one at a time.
    """

    # 1. Try Groq first (more reliable)
    if GROQ_API_KEY:
        try:
            return _call_groq(GROQ_API_KEY, system_prompt, history)
        except Exception as e:
            log_error(f"Groq Failed: {e}")
            log_info("⚠️ Switching to fallback (Gemini)...")
    
    # 2. Fallback to Gemini
    if GEMINI_API_KEY:
        try:
            return _call_gemini(GEMINI_API_KEY, system_prompt, history.copy())
        except Exception as e:
            log_error(f"Gemini Failed: {e}")
    
    # 3. No LLM available - return a default action to try web search
    log_error("No LLM available. Using default web search action.")
    return json.dumps({
        "thought": "No LLM available, trying web search with mission query",
        "tool_name": "web_search",
        "args": {"query": MISSION_GOAL, "max_results": 5}
    })

def run_search_agent():
    # 1. Verify Environment
    if not GEMINI_API_KEY and not GROQ_API_KEY:
        log_error("No AI API Keys found. Check your .env file.")
        return

    # 2. Setup DuckDuckGo Server
    script_dir = os.path.dirname(os.path.abspath(__file__))
    server_script = os.path.join(script_dir, "ddg_server.py")
    server_params = StdioServerParameters(
        command=sys.executable, 
        args=[server_script],
        env=dict(os.environ)
    )

    # 3. Setup Reward (USING SMART CRITIC) - Prioritize Groq
    judge_provider = "groq" if GROQ_API_KEY else "gemini"
    judge_key = GROQ_API_KEY if GROQ_API_KEY else GEMINI_API_KEY
    
    log_info(f"⚖️  Initializing Smart Critic using {judge_provider.upper()}...")

    try:
        critic = LLMReward(
            mission_description=MISSION_GOAL,
            api_key=judge_key,
            provider=judge_provider
        )
    except Exception as e:
        log_error(f"Failed to init LLM Critic, falling back to Keywords: {e}")
        critic = KeywordReward(target_keywords=SUCCESS_KEYWORDS, success_reward=10.0)

    # 4. Initialize Environment
    env = None
    try:
        log_info("🚀 Initializing Agent (Strict Timeout Mode)...")
        env = MCPEnv(server_params=server_params, reward_function=critic, max_steps=8) # Increased max steps
        
        log_debug("Resetting Environment...")
        obs, _ = env.reset()
        log_debug("Environment Reset Complete.")
        
        # Extract tool info safely and verbose log
        tools_count = 0
        try:
            tools_part = obs.split("JSON: ")[1]
            tools_list = json.loads(tools_part)
            tools_count = len(tools_list)
            # Print the names of tools found to verify save_report is there
            tool_names = [t['name'] for t in tools_list]
            log_info(f"🛠️  Tools Connected: {tools_count} -> {tool_names}")
        except:
            tools_part = "[]"
            log_error("Could not parse tools from observation!")
        
        log_info(f"🎯 Mission: {MISSION_GOAL}")

        history = [] 
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            print("\n" + "-"*50, flush=True)
            log_info("🤔 Agent is thinking...")
            
            # A. DECIDE
            raw_response = get_llm_action(tools_part, history)
            
            # B. CLEAN & PARSE RESPONSE (ROBUST)
            try:
                # 1. Clean Markdown blocks
                if raw_response.startswith("```"):
                    raw_response = raw_response.replace("```json", "").replace("```", "").strip()
                
                # 2. Extract JSON using Regex (Handles cases where LLM talks before/after JSON)
                json_match = re.search(r"\{[\s\S]*\}", raw_response)
                if json_match:
                    json_str = json_match.group()
                else:
                    json_str = raw_response
                
                # 3. Parse
                data = json.loads(json_str)
                thought = data.get("thought", "No thought provided.")
                tool_name = data.get("tool_name")
                tool_args = data.get("args", {})
                
                log_thought(thought)
                log_action(f"{tool_name} ({json.dumps(tool_args)})")
                
                # Re-serialize for the Env
                action_json_for_env = json.dumps({"name": tool_name, "args": tool_args})
                
            except json.JSONDecodeError:
                log_error(f"LLM returned invalid JSON: {raw_response}")
                # Fallback: try to pass it anyway or skip
                action_json_for_env = raw_response 
            except Exception as e:
                log_error(f"Parsing error: {e}")
                action_json_for_env = raw_response

            # C. ACT
            log_debug(f"Sending action to MCP Server: {tool_name}...")
            new_obs, reward, terminated, truncated, info = env.step(action_json_for_env)
            log_debug("MCP Server responded.")
            
            # D. LOG RESULTS
            display_obs = (new_obs[:200] + '...') if len(new_obs) > 200 else new_obs
            log_obs(display_obs)
            
            if reward > 0:
                print(f"{Fore.GREEN}💎 Reward: {reward}", flush=True)
            else:
                print(f"{Fore.RED}💎 Reward: {reward}", flush=True)

            # E. UPDATE HISTORY
            history.append({"role": "assistant", "content": raw_response})
            history.append({"role": "user", "content": f"Observation: {new_obs}"})

            # Higher threshold since mission requires multiple steps
            if reward > 9.5:
                log_success("Mission Accomplished! The agent found the information and saved the file.")
                break

    except Exception as e:
        log_error(str(e))
        import traceback
        traceback.print_exc()
    
    finally:
        if env:
            env.close()
            log_info("🔌 Connection closed.")

if __name__ == "__main__":
    run_search_agent()