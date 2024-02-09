class Module:
    def __init__(self):
        self.current = 0
        self.target = 1


def increase(module):
    target = module.target
    current = module.current
    current += 0.2 * target
    module.current = min(target, current)


module = Module()

for i in range(20):
    increase(module)
    print("c: ", module.current, "t: ", module.target)
