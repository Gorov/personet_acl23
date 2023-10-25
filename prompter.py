"""
A dedicated helper to manage templates and prompt building.

In this version, PrompterPersonet can generate prompts without history module.
"""

import json
import os.path as osp
from typing import Union, List


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = osp.join("templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()


class PrompterPersonet(object):
    def __init__(self, with_history=True):
        self.selections = ["(a)", "(b)", "(c)", "(d)", "(e)"]
        self.with_history = with_history

    def generate_prompt(self,
                        snippet_underlined: str = None,
                        snippet_former_context: str = None,
                        snippet_post_context: str = None,
                        history: str = None,
                        book_name: str = None,
                        options: List = None,
                        character: str = None,
                        answer: str = None,
                        gt: str = None):
        if self.with_history:
            prompt1 = "Given a snippet, which is split into three parts by an underline, and its history context from the " \
                      "book \"{}\". Analyze the personality trait of the character \"{}\".\n\n[The history context]:\n".format(book_name, character)
            prompt2 = "{}".format(history)
            prompt3 = "\n\n[The former part before the underlined text]:\n{}\n\n".format(snippet_former_context)
            prompt3 += "[The underlined text]:\n{}\n\n".format(snippet_underlined)
            prompt3 += "[The post part after the underlined text]:\n{}\n\n".format(snippet_post_context)
            prompt3 += "According to the underlined text, its former and post parts, and its history context, which personality trait " \
                       "of the character \"{}\" can be reflected by the underlined text? Select from:\n".format(character)

            # prompt += "\nWhich personality trait of the character {} can be reflected by the snippet? Select from:\n".format(character)
            for option_id, option in enumerate(options):
                prompt3 += self.selections[option_id] + " " + option + "\n"
            prompt3 += "\nAnswer: "
            if answer is not None:
                prompt3 += answer + ' ' + gt
        else:
            prompt1 = "Given a snippet, which is split into three parts by an underline, from the " \
                      "book \"{}\", analyze the personality trait of the character \"{}\".\n\n".format(book_name,
                                                                                                       character)
            prompt1 += "[The former part before the underlined text]:\n{}\n\n".format(snippet_former_context)
            prompt1 += "[The underlined text]:\n{}\n\n".format(snippet_underlined)
            prompt1 += "[The post part after the underlined text]:\n{}\n\n".format(snippet_post_context)
            prompt1 += "According to the underlined text and its former and post parts, which personality trait " \
                       "of the character \"{}\" can be reflected by the underlined text? Select from:\n".format(character)

            # prompt += "\nWhich personality trait of the character {} can be reflected by the snippet? Select from:\n".format(character)
            for option_id, option in enumerate(options):
                prompt1 += self.selections[option_id] + " " + option + "\n"
            prompt1 += "\nAnswer: "
            if answer is not None:
                prompt1 += answer + ' ' + gt
            prompt2 = None
            prompt3 = None
        return prompt1, prompt2, prompt3
