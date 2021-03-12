from tvm.driver.build_module import lower, build

class Compiler:
    def __init__(self, module):
        self.module = module

    def compile(self):
        for fn_name in self.module:
            self.compile_func(self.module[fn_name])

    def compile_func(self, func):
        import pdb; pdb.set_trace()
