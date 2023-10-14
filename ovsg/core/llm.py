from __future__ import annotations
import openai
from PIL import Image
import numpy as np
from ovsg.core.conception import Chatter
from ovsg.utils.misc_utils import print_type_indicator, user_input


class LLM(Chatter):
    """Abstract class for large language model"""

    def __init__(self):
        self._is_api = False

    @property
    def is_api(self):
        return self._is_api

    @is_api.setter
    def is_api(self, is_api):
        self._is_api = is_api


class ChatGPTAPI(LLM):
    def __init__(self, cfg):
        super().__init__()
        self.is_api = True
        # Notice that the API key is not in the config file
        openai.api_key = cfg.api_key[cfg.gpt_model]
        self.gpt_model = cfg.gpt_model
        self.system_prompt = {"role": "system", "content": "You are a helpful assistant."}
        self.conversation = [self.system_prompt]

    def chat(
        self,
        str_msg: str | list[any],
        img_msg: list[Image.Image] | list[np.ndarray] | None = None,
        **kwargs
    ) -> tuple[str, bool]:
        # Print typing indicator
        print_type_indicator("LLM")
        if isinstance(str_msg, list):
            return self.talk_prompt_list(str_msg), True
        elif isinstance(str_msg, str):
            return self.talk_prompt_string(str_msg), True

    def talk_prompt_string(self, msg: str) -> str:
        if isinstance(msg, list):
            msg = " ".join(msg)
        # Send the message to OpenAI
        self.conversation.append({"role": "user", "content": str(msg)})
        while True:
            try:
                reply = openai.ChatCompletion.create(
                    model=self.gpt_model,
                    messages=self.conversation,
                )
                break
            except Exception as e:
                print(e)
                print("Retrying...")
                continue
        reply_content = reply["choices"][0]["message"]["content"]
        total_token = reply["usage"]["total_tokens"]
        self.conversation.append({"role": "assistant", "content": reply_content})
        return reply_content

    def talk_prompt_list(self, prompt_list) -> str:
        """prompt_list is a list of dict, each dict has one key and one value"""
        for prompt in prompt_list:
            for key, value in prompt.items():
                self.conversation.append({"role": key, "content": value})
        reply = openai.ChatCompletion.create(
            model=self.gpt_model,
            messages=self.conversation,
        )
        reply_content = reply["choices"][0]["message"]["content"]
        total_token = reply["usage"]["total_tokens"]
        self.conversation.append({"role": "assistant", "content": reply_content})
        return reply_content

    def reset(self):
        """Clear the conversation history"""
        self.conversation = [self.system_prompt]

    def clear_last(self):
        """Clear the last message"""
        self.conversation.pop()


class ChatGPTWeb(LLM):
    def __init__(self, cfg):
        super().__init__()
        self.is_api = False
        self.end_token = cfg.end_token

    def chat(self, str_msg: str | list[any], img_msg: list[Image.Image] | list[np.ndarray] | None = None, **kwargs) -> tuple[str, bool]:
        if isinstance(str_msg, list):
            return self.talk_prompt_list(str_msg), True
        elif isinstance(str_msg, str):
            return self.talk_prompt_string(str_msg), True

    def talk_prompt_string(self, msg: str) -> str:
        """Talk with LLM"""
        print("-------------- Copy and paste the following to ChatGPT Web --------------")
        print(msg)
        print("------------ Copy and paste the reply from ChatGPT Web below ------------")
        return user_input(self.end_token)

    def talk_prompt_list(self, prompt_list) -> str:
        msg_list = []
        for prompt in prompt_list:
            if "user" in prompt.keys():
                msg_list.append("user:")
                msg_list.append(prompt["user"])
            if "assistant" in prompt.keys():
                msg_list.append("assistant:")
                msg_list.append(prompt["assistant"])
        return self.talk_prompt_string("\n".join(msg_list))

    def reset(self):
        print("-------------- Please open a new ChatGPT page --------------")

    def clear_last(self):
        print("-------- Manually remove your last ChatGPT message --------")
