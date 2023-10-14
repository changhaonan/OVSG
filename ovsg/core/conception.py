from __future__ import annotations
import abc
import os
from abc import ABC
import numpy as np
from PIL import Image


class Question(ABC):
    """
    Abstract class for question
    Question is a printable structure that are used ask for addtional information
    """

    question_str: str

    def __init__(self, question_str: str):
        self.question_str = question_str

    def __str__(self):
        return self.question_str

    def isempty(self):
        return self.question_str == ""


class Comment(ABC):
    """
    Abstract class for comment
    Comment is a printable structure that are used to illustrate the LLM's idea
    """

    comment_str: str

    def __init__(self, comment_str: str):
        self.comment_str = comment_str

    def __str__(self):
        return self.comment_str

    def isempty(self):
        return self.comment_str == ""


class Action(ABC):
    """
    Abstract class for action
    Action should be executable structure that are used to control the robot
    """

    action_str: str

    def __init__(self, action_str: str):
        self.action_str = action_str

    def __str__(self):
        # DO some preprocessing here
        # Romve leading and trailing spaces
        self.action_str = self.action_str.strip()
        return self.action_str

    def isempty(self):
        return self.action_str == ""

    def save(self, filename):
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        with open(filename, "w") as f:
            f.write(self.action_str)


class Check(ABC):
    """
    Abstract class for check
    Check should be executable structure that are used to check the env's state
    """

    check_str: str

    def __init__(self, check_str: str):
        self.check_str = check_str

    def run(self):
        exec(self.check_str)

    def __str__(self):
        return self.check_str

    def isempty(self):
        return self.check_str == ""


class Flag(ABC):
    """
    Abstract class for flag
    """

    flag: bool

    def __init__(self, flag: bool):
        self.flag = flag

    def __bool__(self):
        return self.flag


# Conceptions Implementations
class Chatter(ABC):
    """Abstract class for chatter"""

    @abc.abstractmethod
    def chat(
        self,
        str_msg: str | list[any],
        img_msg: list[Image.Image] | list[np.ndarray] | None = None,
        **kwargs
    ) -> tuple[any, bool]:
        """Talk with chatter using text and image"""
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self):
        """Reset conversation"""
        raise NotImplementedError

    def exit(self):
        """Exit conversation"""
        raise NotImplementedError
