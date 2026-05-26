class TestAgent:
    """
    An agent that can be used to control the agent in a gridworld with text input commands.

    It is not capable of learning (the update function will do nothing)
    """
    def draw_action(self, state):
        action_s = input('select action (w/a/s/d):')
        while action_s not in ['w','a','s','d']:
            action_s = input('select action (w/a/s/d):')

        if action_s == 'w':
            return 0
        elif action_s == 's':
            return 1
        elif action_s == 'a':
            return 2
        elif action_s == 'd':
            return 3
        else:
            raise RuntimeError('invalid action')

    def update(self, state, action, reward, next_state, absorbing):
        pass

