"""
This is an example of how to use the ovsg as a separate module with llm and interpreter.
"""
import hydra
import numpy as np
from ovsg.app import AppSeq, AppEnvOnly


# builder
def app_builder(cfg, **kwargs):
    """build app"""
    app_map = {"base": AppSeq, "env_only": AppEnvOnly}
    if cfg.app_name not in app_map:
        raise ValueError("App name not found.")
    else:
        app = app_map[cfg.app_name](cfg, **kwargs)
    return app


@hydra.main(config_path="../ovsg/config", config_name="config")
def launch_ovsg_llm(cfg):
    """Launch chatgpt_m"""
    app = app_builder(cfg)
    task_name_list = cfg.task_name

    if cfg.debug:
        app.debug()
    else:
        if len(task_name_list) == 1 and task_name_list[0].endswith("hitl"):
            app.eval(task_name_list[0], hitl=True)
        else:
            # eval mode
            success_list = []
            for task_name in task_name_list:
                success = app.eval(task_name, hitl=False)
                success_list.append(success)
            print("Success rate: ", np.mean(success_list))


if __name__ == "__main__":
    launch_ovsg_llm()
