from ovsg.core.conception import Chatter


class EnvBase(Chatter):
    def __init__(self, cfg):
        self.name = cfg.env_name
        super().__init__()

    def reset(self):
        """Reset environment"""
        return True

    def step_render(self, action):
        """Step environment"""
        pass

    def render(self, camera_id=0):
        """Render environment, return image from camera_id"""
        pass

    def close(self):
        """Close environment"""
        pass

    def seed(self, seed):
        """Set seed"""
        pass

    def get_observation(self):
        """Get observation"""
        pass

    def get_reward(self):
        """Get reward"""
        return 0

    def get_done(self):
        """Get done"""
        return True

    def idle(self, step):
        """Idle run"""
        pass

    def debug(self):
        """Debug"""
        pass
