from vergeml.command import command, CommandPlugin
from vergeml.plugins import PLUGINS
from vergeml.option import option
from vergeml.utils import VergeMLError
import shutil
import os.path
import zipfile
import tarfile


@command('run', descr="Run a service or external program.")
@option('service', type=str, subcommand='vergeml.run')
class RunCommand(CommandPlugin):

    def __call__(self, args, env):
        plugin = self.plugins.get('vergeml.run', args['service'])(args['service'])
        plugin(args, env)