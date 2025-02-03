import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

from browser_use.agent.views import ActionResult

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import asyncio
import gradio as gr
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from langchain_openai import ChatOpenAI

from browser_use import Agent, Controller
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import BrowserContext, BrowserContextConfig

from dotenv import load_dotenv
from playwright.sync_api import sync_playwright
import json
import time
import subprocess

load_dotenv()

# Path to Chrome executable
CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'

# Path to your second Chrome profile
CHROME_PROFILE = "Profile 2"  # Chrome profiles are numbered starting from 0, so Profile 1 is the second profile

@dataclass
class ActionResult:
	is_done: bool
	extracted_content: Optional[str]
	error: Optional[str]
	include_in_memory: bool

class BrowserOperator:
	def __init__(self):
		self.browser = Browser(
			config=BrowserConfig(
				chrome_instance_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
				headless=False,
				extra_chromium_args=['--profile-directory=Profile 2']  # Changed from args to extra_chromium_args
			)
		)
		self.agent = None

	async def run_task(self, task: str, headless: bool = False) -> str:
		try:
			self.agent = Agent(
				task=task,
				llm=ChatOpenAI(model='gpt-4'),
				browser=self.browser,
				use_vision=False,
				generate_gif=False
			)
			
			result = await self.agent.run()
			return f"""Task: {task}\n\nExecution Results:\n{result}"""

		except Exception as e:
			return f"Error: {str(e)}"
		finally:
			if self.browser:
				await self.browser.close()

def create_ui():
	operator = BrowserOperator()
	
	with gr.Blocks(title='Browser Operator') as interface:
		gr.Markdown('# Browser Task Automation')
		gr.Markdown("""Examples:
		- "Go to Gmail and compose a new email"
		- "Search for flights from New York to London"
		- "Create a new Google Doc"
		- "Go to YouTube and play music"
		""")
		
		with gr.Row():
			with gr.Column():
				task = gr.Textbox(
					label='Task Description',
					placeholder='E.g., "Go to Gmail and compose a new email"',
					lines=3,
				)
				headless = gr.Checkbox(label='Run Headless', value=False)
				submit_btn = gr.Button('Execute Task', variant='primary')
			
			with gr.Column():
				output = gr.Textbox(
					label='Execution Log',
					lines=15,
					interactive=False
				)
		
		submit_btn.click(
			fn=lambda t, h: asyncio.run(operator.run_task(t, h)),
			inputs=[task, headless],
			outputs=output,
		)
	
	return interface

if __name__ == '__main__':
	if not os.getenv('OPENAI_API_KEY'):
		print("Error: OPENAI_API_KEY not found in environment variables")
		print("Please set it using: export OPENAI_API_KEY='your-key-here'")
		exit(1)
		
	demo = create_ui()
	demo.launch()