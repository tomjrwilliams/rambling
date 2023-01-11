
import inspect

from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.lexers import XmlLexer
from pygments.formatters import HtmlFormatter

import IPython

def render_source(f):
    code = inspect.getsource(f)
    html = highlight(code, PythonLexer(), HtmlFormatter())
    formatter = HtmlFormatter()
    IPython.display.display(IPython.core.display.HTML(
        '<style type="text/css">{}</style>{}'.format(
            formatter.get_style_defs('.highlight'),
            html
        )
    ))