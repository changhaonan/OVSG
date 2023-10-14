from __future__ import annotations
import os
import xml.etree.ElementTree as ET


class TaskBase:
    """Base class for task"""

    def __init__(self, task_name, task_dir, prompt_dir, prompt_manager):
        # load task by task_name
        self.task_name = task_name
        self.task_type = self.task_name.split("_")[0]
        # parse task config
        task_file = os.path.join(task_dir, "task_" + task_name + ".xml")
        with open(task_file, "r") as f:
            xml_string = f.read()
        root = ET.fromstring(xml_string)

        # load env
        env_xml = root.find("env")
        if env_xml is not None and env_xml.text is not None:
            self._env_name = env_xml.text
        else:
            raise ValueError("env is not specified in task config.")

        # load task instruction
        instruction_xml_list = root.findall("instruction")
        if instruction_xml_list is not None:
            self.task_instruction = []
            self.task_label = []
            for instruction_xml in instruction_xml_list:
                if instruction_xml.text is not None:
                    self.task_instruction.append({"user": instruction_xml.text})
                    self.task_label.append(instruction_xml.attrib["label"])
        else:
            self.task_instruction = [{"user": "Are you ready?"}]
            self.task_label = [""]

        # load check script
        check_xml = root.find("check")
        if check_xml is not None and check_xml.text is not None:
            self.check_script = check_xml.text
        else:
            self.check_script = ""

        # load prompt
        prompt_file = os.path.join(prompt_dir, self.task_type + "_env.txt")
        with open(prompt_file, "r") as f:
            self._background = [{"user": f.read()}]
        # load examples
        self.prompt_manager = prompt_manager
        # query_mode has: none, bf, sim
        self.examples = prompt_manager.get_prompt(
            self.task_type, query=self.task_instruction[0]["user"], query_mode="sim"
        )
        # load knowledge
        knowledge_file = os.path.join(prompt_dir, self.task_type + "_knowledge.txt")
        try:
            with open(knowledge_file, "r") as f:
                self._knowledge = [{"user": f.read()}]
        except:
            self._knowledge = [{"user": ""}]

    def background(self):
        """Return task instruction"""
        return self._background

    def knowledge(self):
        """Return task knowledge"""
        return self._knowledge

    def prompt(self) -> list[dict[str, str]]:
        """Return task prompt"""
        return self._background + self.examples + self._knowledge

    def instruction(self, round: int):
        """Return task instruction"""
        return self.task_instruction[round]

    def label(self, round: int):
        """Return task label"""
        return self.task_label[round]

    def __len__(self):
        return len(self.task_instruction)

    def __str__(self):
        str_list = []
        str_list.append(self.background()[0]["user"])
        str_list.append(self.knowledge()[0]["user"])
        str_list.append(self.task_instruction[0]["user"])
        return "\n".join(str_list)

    @property
    def check(self):
        return self.check_script