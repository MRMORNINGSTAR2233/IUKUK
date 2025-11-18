import json
import os
from typing import Optional
from groq import Groq
import google.generativeai as genai

class RewardFunction:
    """Base class for Reward Strategies."""
    def __call__(self, action_json: str, observation: str) -> float:
        raise NotImplementedError

class KeywordReward(RewardFunction):
    """Simple Critic: Rewards if specific keywords appear."""
    def __init__(self, target_keywords: list, success_reward=10.0, step_penalty=-0.1):
        self.target_keywords = target_keywords
        self.success_reward = success_reward
        self.step_penalty = step_penalty

    def __call__(self, action_json: str, observation: str) -> float:
        reward = self.step_penalty
        for keyword in self.target_keywords:
            if keyword.lower() in observation.lower():
                return reward + self.success_reward
        return reward

class LLMReward(RewardFunction):
    """
    Smart Critic: Uses an LLM to judge if the observation satisfies the mission.
    """
    def __init__(self, mission_description: str, api_key: str, provider="gemini"):
        self.mission = mission_description
        self.api_key = api_key
        self.provider = provider
        self.step_penalty = -0.1

        if provider == "groq":
            self.client = Groq(api_key=api_key)

    def _evaluate_with_gemini(self, prompt):
        """Use REST API to avoid DNS issues"""
        import requests
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={self.api_key}"
        
        payload = {
            "contents": [{
                "role": "user",
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.0,
            }
        }
        
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        return result['candidates'][0]['content']['parts'][0]['text'].strip()

    def _evaluate_with_groq(self, prompt):
        completion = self.client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return completion.choices[0].message.content.strip()

    def __call__(self, action_json: str, observation: str) -> float:
        # Construct a prompt for the Judge
        prompt = f"""
        MISSION: {self.mission}
        
        AGENT ACTION: {action_json}
        RESULT OBSERVATION: {observation}
        
        Did the agent COMPLETELY FINISH the ENTIRE mission?
        Consider: Are ALL parts of the mission done? If the mission has multiple steps, have ALL been completed?
        
        Reply with ONLY a number from 0.0 to 10.0.
        10.0 = ALL mission requirements 100% completed.
        5.0-8.0 = Partial progress, but mission not fully complete.
        0.0 = Irrelevant or failure.
        """
        
        try:
            if self.provider == "gemini":
                score_str = self._evaluate_with_gemini(prompt)
            else:
                score_str = self._evaluate_with_groq(prompt)
            
            # Extract number from potential text
            import re
            match = re.search(r"[\d\.]+", score_str)
            if match:
                score = float(match.group())
                return score + self.step_penalty
                
        except Exception as e:
            print(f"Critic Error: {e}")
            
        return self.step_penalty