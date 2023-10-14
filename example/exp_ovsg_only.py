"""
This is an example of how to use the ovsg as a separate module without llm and interpreter.
"""
import hydra
from ovsg.env import env_builder


@hydra.main(config_path="../ovsg/config", config_name="config")
def launch_ovsg_only(cfg):
    """Launch chatgpt_m"""
    # parameters
    query_method = 'prob'
    enable_render = True
    verbose = True
    # init ovsg
    env = env_builder(cfg)
    env.reset()
    # if not using llm, ovsg consume structured language input
    query = ("target @ book {object}\n" +  # target
             "book {object} -- on [spatial] -- bed {object}")  # specification
    env.chat(query, query_method=query_method, enable_render=enable_render, verbose=verbose)


if __name__ == "__main__":
    launch_ovsg_only()
