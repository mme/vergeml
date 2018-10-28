__version__ = "0.1.4"


from vergeml.utils import VergeMLError
from vergeml.command import command, train, predict, CommandPlugin
from vergeml.option import option
from vergeml.model import ModelPlugin, model
from vergeml.data import Data
from vergeml.utils import VergeMLError, SPLITS
