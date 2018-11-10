from vergeml.plugins import PLUGINS
import inspect 
import getopt
from vergeml.utils import did_you_mean, VergeMLError, parse_ai_names
from vergeml.option import Option
from copy import deepcopy

_CMD_META_KEY = '__vergeml_command__'

def command(name=None, descr=None, long_descr=None, examples=None, free_form=False, kind='command'):
    """Define a model command.

    :param name:        An optional name of the command. Defaults to the name of the function.
    :param descr:       A short description of the command
    :param long_descr:  A long description
    :param examples:    Usage examples
    :param free_form:   When true, only the @AI parameters are parsed- the rest of the arguments 
                        are passed as array.
    """
    def decorator(o):
        assert(getattr(o, _CMD_META_KEY, None) is None)
        _name = name or getattr(o, '__name__', None)
        options = list(reversed(Option.discover(o)))
        cmd = Command(_name, 
                      descr=descr, 
                      long_descr=long_descr, 
                      examples=examples, 
                      options=options,
                      free_form=free_form,
                      kind=kind)
        setattr(o, _CMD_META_KEY, cmd)
        return o
    return decorator

def train(name=None, descr=None, long_descr=None, examples=None, free_form=False):
    """Define a training command.

    See `command` for parameter documentation."""
    return command(name=name, descr=descr, long_descr=long_descr, examples=examples, 
                   free_form=free_form, kind='train')

def predict(name=None, descr=None, long_descr=None, examples=None, free_form=False):
    """Define a prediction command.

    See `command` for parameter documentation."""
    return command(name=name, descr=descr, long_descr=long_descr, examples=examples, 
                   free_form=free_form, kind='predict')

class Command:
    """A command can be called directly from the command line.
       It is either a vergeml.cmd plugin or a model command."""

    def __init__(self, name, descr=None, long_descr=None, examples=None, free_form=False, 
                 kind='command', options=None, plugins=PLUGINS):
        """Construct a command.

        See the documentation of the decorator function ´command`.
        """
        self.name = name
        self.descr = (descr or long_descr or "")
        self.long_descr = (long_descr or descr or "")
        self.examples = examples
        self.options = options or []
        self.plugins = plugins
        self.kind = kind
        self.free_form = free_form

        ai_param = list(filter(lambda o: o.is_ai_option(), options))
        assert len(ai_param) <= 1, "Can only have one AI option."
        if ai_param:
            ai_param = ai_param[0]
            assert ai_param.type in (None, 'AI', 'Optional[AI]', 'List[AI]', list, str)
        arg_param = list(filter(lambda o: o.is_argument_option(), options))
        assert len(arg_param) <= 1, "Can only have one argument parameter."
    

    @staticmethod
    def discover(o, plugins=PLUGINS):
        """Discover the command configuration defined on a method or object."""
        res = None
        if hasattr(o, _CMD_META_KEY):
            res = getattr(o, _CMD_META_KEY)
            res.plugins = plugins
            for option in res.options:
                option.plugins = plugins
        return res
    

    @staticmethod
    def find_functions(o, plugins=PLUGINS):
        """Find all functions of an object or class that define a command."""
        # get all functions defined by the model
        fns = [m[1] for m in inspect.getmembers(o) if not m[0].startswith("_") and callable(m[1])]
        
        # sort by the order defined in code
        fns = list(sorted(fns, key=lambda f: f.__code__.co_firstlineno))

        # filter methods where a command is defined
        fns = filter(lambda f: hasattr(f, _CMD_META_KEY), fns)

        return list(fns)

        
    def usage(self, short=False, parent_command=None):
        """Get the command usage.
        
        :param short: Return a short version of the command usage."""
        if self.long_descr and not short:
            result = self.long_descr.strip() + "\n\n"
        else:
            result = ""
    
        result += "Usage:\n"
        
        ai_option = None
        argument_option = None
        subcommand_option = None
        mandatory = []
        optional = []

        for option in self.options:
            if option.is_ai_option():
                ai_option = option
            elif option.is_argument_option():
                argument_option = option
            elif bool(option.subcommand):
                subcommand_option = option
            elif option.is_optional():
                optional.append(option)
            else:
                mandatory.append(option)
        
        result += f"  ml"

        if ai_option:
            if ai_option.type in (list, 'List[AI]'):
                result += f" [{ai_option.name} ...]"
            elif ai_option.is_optional():
                result += f" [{ai_option.name}]"
            else:
                result += f" {ai_option.name}"
        
        if parent_command:
            result += f" {parent_command}:{self.name}"
        else:
            result += f" {self.name}"
        if subcommand_option:
            result += f":{subcommand_option.name}"
        
        if mandatory:
            val = " ".join(map(lambda o: f"--{o.name}=<{o.name}>", mandatory))
            result += f" {val}"
        
        if optional:
            result += " [options]"
        
        if argument_option:
            str_list = isinstance(argument_option.type, str) and argument_option.type.startswith("List")
            list_type = hasattr(argument_option.type, '__origin__') and argument_option.type.__origin__ == list
            if str_list or list_type:
                result += f" [{argument_option.name} ...]"
            elif argument_option.is_optional():
                result += f" [{argument_option.name}]"
            else:
                result += f" {argument_option.name}"
        
        options = []
        if ai_option:
            if ai_option.type in (list, 'List[AI]'):
                options.append((ai_option.name, "A list of trained AIs."))
            else:
                options.append((ai_option.name, "The name of a trained AI."))

        for opt in self.options:
            if opt.is_ai_option() or opt.is_argument_option() or bool(opt.subcommand):
                continue
            opt_name = "--" + opt.name
            if opt.short:
                opt_name = "-" + opt.short + ", " + opt_name
            descr = (opt.descr or "")
            if opt.default is not None:
                if isinstance(opt.default, bool):
                    default_str = 'true' if opt.default else 'false'
                else:
                    default_str = str(opt.default)
                if descr:
                    descr += " "
                descr += f"[default: {default_str}]"
            options.append((opt_name, descr))

        if argument_option and argument_option.descr:
            options.append((argument_option.name, argument_option.descr or ""))


        if options:
            max_name = max(map(lambda o: len(o[0]), options))
            IND = 2
            SPACE = 4
            result += "\n\nOptions:"
            for k, v in options:
                result += "\n" + str(IND * ' ')
                space = (max_name + SPACE) - len(k)
                if v:
                    result += k + str(space * ' ') + v
                else:
                    result += k
        
        if subcommand_option:
            plugins = self.plugins.all(subcommand_option.subcommand)
            max_name = max(map(len, plugins.keys())) if plugins.keys() else 0
            IND = 2
            SPACE = 4
            name = subcommand_option.name.capitalize() + "s"
            if plugins.keys():
                result += f"\n\n{name}:"
                for k, v in plugins.items():
                    result += "\n" + str(IND * ' ')
                    space = (max_name + SPACE) - len(k)
                    cmd = Command.discover(v)
                    if cmd.descr:
                        result += k + str(space * ' ') + cmd.descr
                    else:
                        result += k

        if self.examples and not short:
            result += "\n\nExamples:\n"
            result += "\n".join(map(lambda l: "  " + l, self.examples.splitlines()))
            
        return result
    
    def _invalid_arguments(self, message=None, help_topic=None):
        message = message or "Invalid arguments."
        raise VergeMLError(message, help_topic=help_topic)

    def parse(self, argv, env_options={}):
        """Parse the command and return the result."""
        res = {}
        ai_names, rest = parse_ai_names(argv)

        # subcommand
        subcommand_param = next((filter(lambda o: bool(o.subcommand), self.options)), None)

        if subcommand_param:
            if not ":" in rest[0]:
                raise VergeMLError(f"Missing {subcommand_param.name}.", help_topic=self.name)
            command, subcommand = rest[0].split(":", 1)
            assert command == self.name
            argv = deepcopy(argv)
            argv[argv.index(rest[0])] = subcommand

            plugin = self.plugins.get(subcommand_param.subcommand, subcommand)
            if not plugin:
                raise VergeMLError(f"Invalid {subcommand_param.name}.", help_topic=self.name)
            
            cmd = Command.discover(plugin)
            try:
                res = cmd.parse(argv, env_options)
                res[subcommand_param.name] = subcommand
                for opt in cmd.options:
                    if opt.name not in res:
                        res[opt.name] = opt.default
                return res
            except VergeMLError as e:
                e.help_topic = f"{command}:{subcommand}"
                raise e

        # AI params
        ai_param = next((filter(lambda o: o.is_ai_option(), self.options)), None)
        
        if ai_param:
            if ai_param.type in ('AI', None, str):
                ai_conf = 'required'
            elif ai_param.type == 'Optional[AI]':
                ai_conf = 'optional'
            elif ai_param.type in (list, 'list', 'List[AI]'):
                ai_conf = 'list'
        else:
            ai_conf = 'none'
        

        if (ai_conf == 'optional' and len(ai_names) > 1) or \
           (ai_conf == 'required' and len(ai_names) != 1) or \
           (ai_conf == 'none' and len(ai_names) != 0):
            raise self._invalid_arguments(help_topic=self.name)
        
        if ai_conf in ('required', 'optional'):
            res[ai_param.name] = next(iter(ai_names), None)
        elif ai_conf == 'list':
            res[ai_param.name] = ai_names

        # command name
        assert self.name == rest.pop(0)

        # in case of free form commands, just return AI and rest
        if self.free_form:
            ai_res = None
            if ai_param:
                ai_res = res.get(ai_param.name)
            return (ai_res, rest)

        longopts = []
        shortopts = ""
        
        for opt in self.options:
            if opt.is_ai_option() or opt.is_argument_option():
                continue

            if opt.flag:
                opt_type = eval(opt.type) if isinstance(opt.type, str) else opt.type
                assert opt_type in (bool, None)
                longopts.append(opt.name)
            else:
                longopts.append(opt.name + "=")
            
            if opt.short:
                
                letter = opt.short
                assert letter not in shortopts

                if opt.type == bool:
                    shortopts += letter
                else:
                    shortopts += letter + ":"
            
        try:
            args, extra = getopt.getopt(rest, shortopts, longopts)
        except getopt.GetoptError as err:
            if err.opt:
                candidates = list(shortopts.replace(":", "")) + list(map(lambda o: o.rstrip("="), longopts))
                suggestion = did_you_mean(candidates, err.opt)
                dashes = '-' if len(err.opt) == 1 else '--'
                raise VergeMLError(f"Invalid option {dashes}{err.opt}", suggestion, help_topic=self.name)
            else:
                raise VergeMLError(f"Invalid option.", help_topic=self.name)
        
        
        shorts_dict = {}
        longs_dict = {}
        for k, v in args:
            if k.startswith("--"):
                longs_dict[k.lstrip("-")] = v
            else:
                shorts_dict[k.lstrip("-")] = v    

        extra_param = next((filter(lambda o: o.is_argument_option(), self.options)), None)

        if extra_param:
            if extra_param.is_optional():
                extra_conf = 'optional'
            elif isinstance(extra_param.type, str) and extra_param.type.startswith("List"):
                extra_conf = 'list'
            elif hasattr(extra_param.type, '__origin__') and extra_param.type.__origin__ == list:
                extra_conf = 'list'
            elif extra_param.type == list:
                extra_conf = 'list'
            else:
                extra_conf = 'required'
        else:
            extra_conf = 'none'

        if (extra_conf == 'optional' and len(extra) > 1) or \
            (extra_conf == 'none' and len(extra) != 0):
            raise self._invalid_arguments(help_topic=self.name)
        
        elif extra_conf == 'required' and len(extra) == 0:
            raise self._invalid_arguments(f"Missing argument {extra_param.name}.", help_topic=self.name)
        
        elif extra_conf == 'required' and len(extra) > 1:
            raise self._invalid_arguments(f"Invalid arguments.", help_topic=self.name)

        if extra_conf in ('optional', 'required'):
            res[extra_param.name] = next(iter(extra), None)
        elif extra_conf == 'list':
            res[extra_param.name] = extra

        for opt in self.options:
            if opt.is_ai_option() or opt.is_argument_option():
                continue
            
            value = None
            if opt.flag:
                if opt.name in longs_dict:
                    value = True
            elif opt.name in longs_dict:
                value = longs_dict[opt.name]

            if opt.short:
                letter = opt.short
                if letter in shorts_dict:
                    if opt.type == bool:
                        value = True
                    else:
                        value = shorts_dict[letter]
            
            if value is None and opt.name in env_options:
                value = env_options[opt.name]

            if value is None and not opt.is_optional():
                raise self._invalid_arguments(message=f'Missing argument --{opt.name}.', help_topic=self.name)
            elif value is not None:
                try:
                    value = opt.cast_value(value)
                    value = opt.transform_value(value)
                    opt.validate_value(value)
                
                    res[opt.name] = value
                except VergeMLError as err:
                    err.message = f"Invalid value for option --{opt.name}."
                    raise err
        
        return res


class CommandPlugin:
    def __init__(self, name, plugins=PLUGINS):
        self.name = name
        self.plugins = plugins

        # avoid circular dependency
        from vergeml.command import Command

        cmd = Command.discover(self)
        assert(cmd)
        cmd.name = name
    
    def __call__(self, argv, env):
        raise NotImplementedError
