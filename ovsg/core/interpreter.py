import abc
from abc import ABC
import re
import xml.etree.ElementTree as ET
from ovsg.core.conception import Action, Check, Comment, Question


class Interpreter(ABC):
    """Abstract class for Interpreter"""

    # Set this in SOME subclasses
    metadata = {}

    # Set this in ALL subclasses
    name = ""

    @abc.abstractmethod
    def interprete(self, plan, **kwargs):
        """interprete the code from LLM to executable function"""
        raise NotImplementedError


class InterpreterXML(Interpreter):
    """XML Interpreter"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.name = "XML Interpreter"

    def interprete(self, xml_string, **kwargs):
        """interprete the code from LLM to executable function"""
        # parse xml from plan_string

        # Use regular expressions to extract the XML command from the string
        match = re.search(r"<root>.*?</root>", xml_string, re.DOTALL)
        if match:
            xml_string = match.group(0)  # use the first match
        else:
            print("No XML found. Treat the whole string as action string.")
            action = Action(xml_string)
            return action, Comment(""), Question(""), Check("")

        # parse the plan string block by xml tag
        root = ET.fromstring(xml_string)

        # Access elements and values
        action = root.find("action")
        comment = root.find("comment")
        question = root.find("question")
        check = root.find("check")

        # action
        if action is not None:
            action = Action(action.text)
        else:
            action = Action("")

        # comment
        if comment is not None:
            comment = Comment(comment.text)
        else:
            comment = Comment("")

        # question
        if question is not None:
            question = Question(question.text)
        else:
            question = Question("")

        # check
        if check is not None:
            check = Check(check.text)
        else:
            check = Check("")

        return action, comment, question, check


class InterpreterMD(Interpreter):
    """Markdown Interpreter"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.name = "Markdown Interpreter"

    def interprete(self, md_string, **kwargs):
        """interprete the code from LLM to executable function"""
        # parse markdown syntax from plan_string

        # parse code
        # try ```python ``` syntax
        match = re.search(r"```python(.*?)```", md_string, re.DOTALL)
        if match:
            code_string = match.groups()[0]  # use the first match
        else:
            # try ``` ``` syntax
            match = re.search(r"```(.*?)```", md_string, re.DOTALL)
            if match:
                code_string = match.groups()[0]  # use the first match
            else:
                code_string = ""
        action = Action(code_string)

        # parse question
        # try @@question@@ syntax
        match = re.search(r"@@(.*?)@@", md_string, re.DOTALL)
        if match:
            question_string = match.groups()[0]
        else:
            question_string = ""
        question = Question(question_string)

        # parse check
        # try ^^check^^ syntax
        match = re.search(r"\^\^(.*?)\^\^", md_string, re.DOTALL)
        if match:
            check_string = match.groups()[0]
        else:
            check_string = ""
        check = Check(check_string)

        # parse comment
        # remove code, question, check, left is comment
        md_string = re.sub(r"```(.*?)```", "", md_string, flags=re.DOTALL)
        md_string = re.sub(r"@@(.*?)@@", "", md_string, flags=re.DOTALL)
        md_string = re.sub(r"\^\^(.*?)\^\^", "", md_string, flags=re.DOTALL)
        comment = Comment(md_string)
        return action, comment, question, check
