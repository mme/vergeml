from vergeml.command import command, CommandPlugin, Command
from vergeml.option import option
from vergeml.utils import VergeMLError
from vergeml.display import DISPLAY
from copy import deepcopy
import sys
from collections import OrderedDict
import os
import os.path
import json
import csv
from datetime import datetime
import yaml
import io

EXAMPLES = """
$ ml list -sacc 
# sort by acc value

$ ml list status -eq RUNNING
# show trainings that are currently running

$ ml list test_acc -gt 0.8
# show AIs with a test accuracy that is greater than 0.8

# available comparison operations:
# -gt, -lt, -eq, -neq, -gte and -lte
""".strip()
@command('list', descr="List trained models.", free_form=True, examples=EXAMPLES)
@option('sort', descr="By which column to sort.", default='created_at', short='s')
@option('order', descr="Sort order.", default='asc', short='o', validate=('asc', 'desc'))
@option('columns', descr="Which columns to show.", type='Optional[str]', short='c')
@option('output', descr="Output format.", default='table', validate=('table', 'csv', 'json'))
class ListCommand(CommandPlugin):

     def __call__(self, args, env):
        args = args[1]

        comps = []
        for ix, arg in enumerate(args):
            if arg in ('-gt', '-lt', '-eq', '-neq', '-gte', '-lte'):
                start, end = ix - 1, ix + 1
                if start < 0 or end >= len(args):
                    raise VergeMLError("Invalid options.", help_topic='list')
                comps.append((start, end))

        comp_args = []
        for start, end in reversed(comps):
            comp_args.append(args[start:end+1])
            del args[start:end+1]

        cmd = deepcopy(Command.discover(ListCommand))
        cmd.free_form = False
        args.insert(0, 'list')
        args = cmd.parse(args)
        args.setdefault('sort', 'created_at')
        args.setdefault('order', 'asc')
        args.setdefault('columns', None)
        args.setdefault('output', 'table')

        train_dir = env.get("trainings-dir")
        if not os.path.exists(train_dir):
            return

        info = {}
        hyper = {}

        for AI in os.listdir(train_dir):
            data_yaml = os.path.join(train_dir, AI, 'data.yaml')
            if os.path.isfile(data_yaml):
                with open(data_yaml) as f:
                    doc = yaml.load(f)
            else:
                doc = {}
            info[AI] = {}
            hyper[AI] = {}

            if 'model' in doc:
                info[AI]['model'] = doc['model']

            if 'results' in doc:
                info[AI].update(doc['results'])

            if 'hyperparameters' in doc:
                hyper[AI].update(doc['hyperparameters'])

        if args['columns']:
            theader = ['AI'] + [s.strip() for s in args['columns'].split(",")]
            exclude = []
        else:
            theader = ['AI', 'model', 'status', 'num_samples', 'training_start', 'epochs']
            exclude = ['training_end', 'steps', 'created_at']

        sort = [s.strip() for s in args['sort'].split(",")]

        info = OrderedDict(sorted(info.items(), reverse=(args['order'] == 'asc'),
                           key=lambda x: [x[1].get(s, 0) for s in sort]))

        tdata = []
        left_align = set([0])

        for AI, results in info.items():
            rdata = [""] * len(theader)
            rdata[0] = "@" + AI

            if not _filter(results, hyper[AI], comp_args):
                continue

            for k, v in sorted(results.items()):
                if k in exclude and not args['columns']:
                    continue

                if not k in theader and not args['columns'] and isinstance(v, (str, int, float)):
                    theader.append(k)
                    rdata.append(None)
                if k in theader:
                    pos = theader.index(k)

                    if k in ('training_start', 'training_end', 'created_at'):
                        v = datetime.utcfromtimestamp(v)
                        v = v.strftime("%Y-%m-%d %H:%M")
                    elif isinstance(v, float):
                        v = "%.4f" % v
                    elif isinstance(v, str):
                        left_align.add(pos)

                    rdata[pos] = v

            for k, v in sorted(hyper[AI].items()):
                if k in theader:
                    pos = theader.index(k)
                    if isinstance(v, float):
                        v = "%.4f" % v
                    elif isinstance(v, str):
                        left_align.add(pos)

                    rdata[pos] = v

            tdata.append(rdata)
        
        if args['output'] == 'table':
            if not tdata:
                return
            tdata.insert(0, theader)
            print(DISPLAY.table(tdata, left_align=left_align).getvalue(fit=True))
        elif args['output'] == 'json':
            res = []
            for row in tdata:
                res.append(dict(zip(theader, row)))
            print(json.dumps(res))

        elif args['output'] == 'csv':
            buffer = io.StringIO()

            writer = csv.writer(buffer)
            writer.writerow(theader)
            for row in tdata:
                writer.writerow(row)
            val = buffer.getvalue()
            val = val.replace('\r', '')
            print(val.strip())

def _filter(info, hyper, comp_args):
    try:
        cols = {}
        cols.update(hyper)
        cols.update(info)
        res = True
        for col, op, val in comp_args:
            
            if not col in cols:
                return False

            cval = cols[col]
            
            if isinstance(cval, int):
                val = int(val)
            elif isinstance(cval, float):
                val = float(val)

            if op == '-eq':
                res = res and (cval == val)
            elif op == '-neq':
                res = res and (cval != val)
            elif op == '-gt':
                res = res and (cval > val)
            elif op == '-lt':
                res = res and (cval < val)
            elif op == '-gte':
                res = res and (cval >= val)
            elif op == '-lte':
                res = res and (cval <= val)
            
            if not res:
                return False
        return res
    except Exception:
        return False
