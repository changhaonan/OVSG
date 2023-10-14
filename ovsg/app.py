"""  Generate LLM app
"""
from __future__ import annotations
import os
import json
import pickle
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from ovsg.core.conception import Chatter
from ovsg.core.env import EnvBase
from ovsg.core.llm import LLM
from ovsg.core.interpreter import Interpreter
from ovsg.env import env_builder
from ovsg.utils.misc_utils import user_input, action_verbose

from ovsg.core.task import TaskBase, TaskBase
from ovsg.core.prompt import PromptManagerDataBase
from ovsg.core.llm import ChatGPTWeb, ChatGPTAPI
from ovsg.core.interpreter import InterpreterXML, InterpreterMD

interpreter_map = {"chatgpt_api": InterpreterMD, "chatgpt_web": InterpreterMD}
llm_map = {"chatgpt_web": ChatGPTWeb, "chatgpt_api": ChatGPTAPI}
prompt_manager_map = {"data_base": PromptManagerDataBase}


def task_builder(cfg, task_name: str, **kwargs) -> TaskBase:
    """build task"""
    task = TaskBase(task_name, cfg.task_dir, cfg.prompt_dir, prompt_manager_map[cfg.prompt_manager_name](cfg), **kwargs)
    return task


class AppSeq(Chatter):
    """AppSeq: Sequentially call LLM, Interpreter, and Env"""

    llm: LLM
    env: EnvBase
    interpreter: Interpreter

    def __init__(self, cfg):
        # module init
        self.prompt_manager = prompt_manager_map[cfg.prompt_manager_name](cfg)
        self.env = env_builder(cfg)
        self.llm = llm_map[cfg.llm_name](cfg)
        self.interpreter = interpreter_map[cfg.llm_name](cfg)
        # bind task builder
        self.task_builder = lambda task_name: task_builder(cfg, task_name)
        # params
        self.log_path = cfg.log_path
        self.end_token = cfg.end_token
        self.verbose = cfg.verbose
        self.record = cfg.record
        self.execute_options = OmegaConf.to_container(cfg.execute_options, resolve=True)
        # log
        self.exp_name = cfg.exp_name
        self.history = []

    def reset(self):
        self.env.reset()
        self.llm.reset()

    def log(self, task_name: str):
        """log history of task"""
        # save history
        if self.log_path:
            exp_folder = os.path.join(self.log_path, self.env.name, "history", self.exp_name, task_name)
            if not os.path.exists(exp_folder):
                os.makedirs(exp_folder)
            file = os.path.join(exp_folder, "history.json")
            with open(file, "w") as f:
                json.dump(self.history, f, indent=4)
            file = os.path.join(exp_folder, "history.pkl")
            with open(file, "wb") as f:
                pickle.dump(self.history, f)
        # reset history
        self.history = []

    # interface
    def chat(
        self,
        str_msg: str | list[any],
        img_msg: list[Image.Image] | list[np.ndarray] | None = None,
        **kwargs
    ) -> tuple[any, bool]:
        """main interface"""
        # think
        llm_plan, llm_flag = self.llm.chat(str_msg)
        # execute
        kwargs = {**self.execute_options, **kwargs}
        reply, action_executed = self.execute(llm_plan, img_info=img_msg, **kwargs)
        return reply, action_executed

    def execute(
        self,
        llm_plan: str,
        img_info: list[Image.Image] | list[np.ndarray] | None = None,
        **kwargs
    ) -> tuple[str, bool]:
        """Execute the llm plan"""
        # params
        verbose = kwargs.get("verbose", self.verbose)
        record = kwargs.get("record", self.record)
        action_executed = False
        # parse
        action, comment, question, check = self.interpreter.interprete(llm_plan)
        # print comment first
        if not comment.isempty() and verbose:
            print("* LLM:", comment)

        # run the action
        reply = None
        if not action.isempty() and not action_executed:
            with action_verbose(
                action, verbose, record, log_path=self.log_path, env_name=self.env.name
            ):
                reply, done = self.env.chat(str(action), **kwargs)
                # update history
                self.history.append({})
                label = kwargs.get("label", "")
                self.history[-1]["label"] = str(label)
                self.history[-1]["action"] = str(action)
                for k, v in reply.items():
                    self.history[-1][k] = v
            if done:
                action_executed = True
        return reply, action_executed

    # eval method
    def eval(self, task_name, hitl=False):
        """Human in the loop"""
        task = self.task_builder(task_name)
        # init
        self.reset()
        action_executed = False
        round = 0
        while True:
            if round == 0:
                current_content = task.prompt() + [task.instruction(round)]
                current_label = task.label(round)
            else:
                if hitl:
                    # if human in the loop, wait for user input
                    current_content = user_input(self.end_token)
                    current_label = ""
                    action_executed = False
                    # reset
                    if current_content.upper() == "RESET":
                        self.reset()
                        action_executed = False
                        round = 0
                        continue
                    elif current_content.upper() in ["STOP", "QUIT", "EXIT", "Q"]:
                        break
                else:
                    if round < len(task):
                        current_content = task.instruction(round)["user"]
                        current_label = task.label(round)
                    else:
                        break

            # chat
            reply, action_executed = self.chat(current_content, verbose=self.verbose, label=current_label)
            # update round
            round = round + 1
        # log
        self.log(task_name)
        # eval
        if action_executed:
            if task.check:
                reply, done = self.env.chat(task.check)
                return done
            else:
                return True  # no check
        else:
            return False

    # testing method
    def debug(self):
        """debug"""
        self.reset()
        self.env.debug()


class AppEnvOnly(Chatter):
    """AppEnvOnly: Only call Env"""

    env: EnvBase

    def __init__(self, cfg):
        # module init
        self.env = env_builder(cfg)

    def eval(self, task_name, **kwargs):
        """main interface"""
        reply, done = self.env.eval(task_name, **kwargs)
        return done

    def reset(self):
        """reset"""
        self.env.reset()

    def exit(self):
        pass

    def chat(self, str_msg):
        """main interface"""
        reply, done = self.env.chat(str_msg)
        return reply, done

    # testing method
    def debug(self):
        """debug"""
        self.reset()
        self.env.debug()
