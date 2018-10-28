from vergeml.data import Data
from vergeml.utils import VergeMLError
import os.path
from vergeml.command import command
from vergeml.option import option
from vergeml.command import CommandPlugin

@command('preprocess', descr="Preprocess samples and save the output. ")
@option('<directory>', type='Optional[str]')
@option('num-samples', type='Optional[int]', default=None, descr="The number of samples to process.", short='n')
class PreprocessCommand(CommandPlugin):
    
    def __call__(self, args, env):  
        pass
        # dest = args['<project-name>']   
        # if not env.model:
        #     template = "# model:\n#   name: <name of your model>\n"
        # else:
        #     templates = "TODO"
        # # elif hasattr(env.model, '_TEMPLATE'):
        # #     template = env.model._TEMPLATE
        # # else:
        # #     template = "model:\n  name: {}\n".format(env.get('model.name'))

        # # dest = args['<project-name>']
        # if os.path.exists(dest):
        #     raise VergeMLError("Directory already exists: {}".format(dest))

        # os.makedirs(dest)
        # os.makedirs(os.path.join(dest, "samples"))
        # with open(os.path.join(dest, "vergeml.yaml"), "w") as f:
        #     f.write(template)
        # print("Created new project: {}".format(dest))


# def preprocess(argv, env=None):
#     """Run the preprocessing pipeline and store the resulting samples on disk.
# The results of running preprocess can be optionally written to <output-dir>.

# Usage:
#   ml preprocess [<output-dir>]"""
#     args = parseargv(preprocess.__doc__, argv)

#     # if the model has defaults settings for training, apply them
#     if hasattr(env.model, '_set_train_defaults'):
#         env.model._set_train_defaults(env)

#     if args['<output-dir>']:
#         if os.path.exists(args['<output-dir>']):
#             raise Error("Output dir exists: {}".format(args['<output-dir>']))
#         env.set("data.output_dir", args['<output-dir>'])

#     data = Data(env)
#     if data.output.output_dir and os.path.exists(data.output.output_dir):
#         print("Data already preprocessed in a previous run: {}".format(data.output.output_dir))
#     elif not len(data.ops):
#         print("Nothing to do - no preprocessing operations to apply.")
#     else:
#         data._process()
