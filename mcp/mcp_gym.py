import gymnasium as gym
from gymnasium import spaces
import numpy as np
import asyncio
import threading
import json
import concurrent.futures
from typing import Any, Dict, Optional, Callable, Tuple, Union
from contextlib import AsyncExitStack

# Import MCP SDK components
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class AsyncBridge:
    """
    Runs a background event loop for async operations.
    """
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._start_background_loop, daemon=True)
        self.thread.start()

    def _start_background_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def stop(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join()

class MCPEnv(gym.Env):
    """
    The Universal Adapter: Turns any MCP Server into a Gym Environment.
    Uses a persistent background task to manage the MCP session lifecycle strictly.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self, 
        server_params: StdioServerParameters, 
        reward_function: Callable[[str, str], float],
        max_steps: int = 10
    ):
        super().__init__()
        
        self.server_params = server_params
        self.calc_reward = reward_function
        self.max_steps = max_steps
        self.current_step = 0
        
        # Initialize Bridge
        self.bridge = AsyncBridge()
        
        # We use a persistent worker task to ensure 'anyio' contexts are 
        # entered and exited in the same task.
        self._init_future = concurrent.futures.Future()
        
        # Launch the worker on the background loop
        print(f"🔌 Connecting to MCP Server: {server_params.command}...")
        asyncio.run_coroutine_threadsafe(self._session_worker(), self.bridge.loop)
        
        # Wait for connection to be established (with timeout)
        try:
            self._init_future.result(timeout=10)
        except Exception as e:
            self.bridge.stop()
            raise RuntimeError(f"Failed to connect to MCP server: {e}")

        # 2. Discover Tools
        print("🔍 Discovering Tools...")
        self.tools = self._run_on_worker("list_tools")
        tool_names = [t.name for t in self.tools.tools]
        print(f"✅ Found {len(tool_names)} tools: {tool_names}")

        # 3. Define Spaces
        self.action_space = spaces.Text(max_length=2000, charset="runic") 
        self.observation_space = spaces.Text(max_length=10000, charset="runic")

    def reset(self, seed=None, options=None):
        """
        Resets the environment to an initial state and returns the initial observation.
        """
        super().reset(seed=seed)
        self.current_step = 0
        
        # The initial observation is the list of available tools
        # We use the tools discovered during init
        tools_info = [
            {"name": t.name, "description": t.description, "schema": t.inputSchema} 
            for t in self.tools.tools
        ]
        
        # Create a string observation that describes the available tools
        observation = f"Environment Started. Available Tools JSON: {json.dumps(tools_info)}"
        
        return observation, {}

    async def _session_worker(self):
        """
        The ONLY task that interacts with the MCP session context.
        This runs on the background thread and stays alive for the session duration.
        """
        # Create an asyncio queue *on the loop* for receiving commands
        self._async_queue = asyncio.Queue()
        
        try:
            async with AsyncExitStack() as stack:
                # 1. Enter Contexts
                read, write = await stack.enter_async_context(stdio_client(self.server_params))
                self.session = await stack.enter_async_context(ClientSession(read, write))
                await self.session.initialize()
                
                # Signal Success to Main Thread
                self._init_future.set_result(True)
                
                # 2. Command Loop
                while True:
                    # Wait for a command: (command_name, args, result_future)
                    cmd_name, args, result_fut = await self._async_queue.get()
                    
                    if cmd_name == "STOP":
                        result_fut.set_result(True)
                        break # Exit loop -> Exit Stack -> Close Session
                        
                    try:
                        # Execute MCP actions
                        if cmd_name == "list_tools":
                            res = await self.session.list_tools()
                            result_fut.set_result(res)
                            
                        elif cmd_name == "call_tool":
                            res = await self.session.call_tool(args['name'], arguments=args['args'])
                            result_fut.set_result(res)
                        
                        else:
                            result_fut.set_exception(ValueError(f"Unknown command: {cmd_name}"))
                            
                    except Exception as e:
                        if not result_fut.done():
                            result_fut.set_exception(e)
                        
        except Exception as e:
            if not self._init_future.done():
                self._init_future.set_exception(e)
            print(f"MCP Worker Task ended: {e}")

    def _run_on_worker(self, cmd_name: str, **kwargs) -> Any:
        """
        Helper to send a command to the worker and wait for the result synchronously.
        """
        # Create a thread-safe Future to wait for the result
        fut = concurrent.futures.Future()
        
        def _enqueue():
            # This runs on the loop. Put (cmd, args, fut) into the async queue.
            # We check if queue exists to avoid race conditions during shutdown.
            if hasattr(self, '_async_queue'):
                self._async_queue.put_nowait((cmd_name, kwargs, fut))
            else:
                fut.set_exception(RuntimeError("Worker queue not initialized"))
            
        self.bridge.loop.call_soon_threadsafe(_enqueue)
        return fut.result()

    def step(self, action: str):
        """
        Run one timestep of the environment's dynamics.
        """
        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        terminated = False
        
        try:
            # 1. Parse Action
            if not isinstance(action, str):
                action = str(action)
            
            cmd = json.loads(action)
            tool_name = cmd.get("name")
            tool_args = cmd.get("args", {})

            if not tool_name:
                raise ValueError("Action JSON must contain 'name' field.")

            # 2. Execute Tool (via Worker Task)
            result = self._run_on_worker("call_tool", name=tool_name, args=tool_args)
            
            # 3. Process Observation
            # MCP results are a list of content objects (Text or Image). We extract text.
            obs_text_list = [c.text for c in result.content if c.type == 'text']
            observation = "\n".join(obs_text_list) if obs_text_list else "Success (No Text Output)"

            # 4. Calculate Reward
            reward = self.calc_reward(action, observation)

        except json.JSONDecodeError:
            observation = "Error: Invalid JSON format in action."
            reward = -0.1
            
        except Exception as e:
            observation = f"Error executing tool: {str(e)}"
            reward = -1.0

        return observation, float(reward), terminated, truncated, {}

    def close(self):
        """Clean up the background thread and connection."""
        if hasattr(self, 'bridge') and self.bridge.thread.is_alive():
            try:
                # Send STOP signal to the worker task
                self._run_on_worker("STOP")
            except Exception:
                # Ignore errors if worker is already dead
                pass
            self.bridge.stop()