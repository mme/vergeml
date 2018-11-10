from vergeml.command import command, CommandPlugin
from vergeml.option import option

@command('roc', descr="Plot a ROC curve.")
class ROCPlot(CommandPlugin):

    def __call__(self, args, env):
        print("TESTING")