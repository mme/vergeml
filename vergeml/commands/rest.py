from vergeml import Error, parseargv

import os
import http.server
import cgi
import json

 
class RESTHandler(http.server.BaseHTTPRequestHandler):
 
    def do_GET(self):
        """Serve a GET request."""
        self.send_response(404)
        self.send_header('Content-type','text/plain')
        self.end_headers()
        self.wfile.write(b"Not found.")
        return
 
    def do_POST(self):
        """Serve a POST request."""
        self.send_response(200)
        self.send_header('Content-type','text/html')
        self.end_headers()

        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={'REQUEST_METHOD':'POST'})
        
        vars = {}
        files = {}

        for k in form.keys():
          
            it = form[k]

            if isinstance(it, list):
                var = None
                isfile = True
                for li in it:
                    if li.file:
                        var.append(dict(file=li.file, filename=li.filename))
                    else:
                        isfile = False
                        var.append(li.value)
                if isfile:
                    files[k] = var
                else:
                    vars[k] = var
            else:
                if it.file:
                    files[k] = [dict(file=it.file, filename=it.filename)]
                else:
                    vars[k] = it.value
        
        res = self.server.predict(vars, files)
        self.wfile.write(res)


class RESTServer(http.server.HTTPServer):

    def __init__(self, conf, handler, predict):
        self.predict = predict
        super().__init__(conf, handler)

def rest(argv, env):
    """Serve a trained model via a REST interface.

Usage:
  ml @instance rest [--address=<address>] [--port=<port>]

Options:
  --port=<port>         REST server port [default: 2204]
  --address=<address>   The address to use [default: localhost]
"""
    args = parseargv(rest.__doc__, argv)

    address = env.get('rest.address')
    if not address or getattr(args['--address'], 'is_default', False):
        address = args['--address']
    
    port = env.get('rest.port')
    if not port or getattr(args['--port'], 'is_default', False):
        port = args['--port']

    if hasattr(env.model, '_predict'):
        predict = getattr(env.model, '_predict')(env)
    else:
        raise Error("REST Interface not supported.")
    

    def local_predict(vars, files):
        
        res = predict(vars, files)
        assert isinstance(res, (bytes, int, float, str, dict, list, set))

        if isinstance(res, bytes):
            return res
        elif isinstance(res, (int, float, str)):
            return str(res).encode()
        elif isinstance(res, (dict, list)):
            return json.dumps(res).encode()
        elif isinstance(res, (set)):
            return json.dumps(list(res)).encode()

    try:
        server = RESTServer((address, int(port)), RESTHandler, local_predict)
        print ('Started REST service: http://{}:{}'.format(address, port))

        server.serve_forever()

    except KeyboardInterrupt:
        print ('Shutting down the web server')
        server.socket.close()