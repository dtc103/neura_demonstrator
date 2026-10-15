"""One close/open action per click; no deferred clicks or retargeting."""


class ToggleCommands:
    def __init__(self):
        self.closed = False
        self.active = None

    def next(self, clicked):
        if not clicked or self.active is not None:
            return None
        self.active = not self.closed
        return self.active

    def complete(self, success):
        if self.active is None:
            raise RuntimeError('No gripper command in progress')
        if success:
            self.closed = self.active
        self.active = None
