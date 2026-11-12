"""Apply XPath expressions to plain dicts (used by ``extractor.JsonXpathExtractor``).

The dict is rendered as a throwaway ``<record>`` XML document so lxml can evaluate the XPath, then the
matched element paths are translated back into dict/list lookups on the original data. Data types are
not preserved in the XML because the operation is never reversed.
"""

import re

from lxml import etree

# what a dict key has to look like to be an element name (after '@' has become '__')
_ELEMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")


def _escape_xml(s):
    s = s.replace("&", "&amp;").replace('"', "&quot;").replace("'", "&apos;")
    s = s.replace("<", "&lt;").replace(">", "&gt;")
    return s


# Doesn't need to preserve data type as we never reverse the operation
def _convert(what, output):
    if type(what) is dict:
        for k, v in what.items():
            tag = str(k).replace("@", "__")
            if not _ELEMENT_NAME.match(tag):
                raise ValueError(f"the key {k!r} cannot be an XPath step (it is not a usable element name)")
            if type(v) is not list:
                v = [v]
            for item in v:
                output.append(f"<{tag}>")
                _convert(item, output)
                output.append(f"</{tag}>")
    elif isinstance(what, str):
        output.append(_escape_xml(what))
    elif isinstance(what, bool):
        output.append("true" if what else "false")
    elif what is None:
        output.append("")
    elif type(what) is list:
        # a list inside a list has no key of its own: its items are text of the enclosing element
        for item in what:
            _convert(item, output)
    else:
        # numbers, and anything else a record may hold: only its text is needed, for predicates
        output.append(_escape_xml(str(what)))


def dicttoxml(what):
    """Render dict *what* as a ``<record>...</record>`` XML string ('@' in keys becomes '__')."""
    output = ["<record>"]
    _convert(what, output)
    output.append("</record>")
    xml = "".join(output)
    return xml


def xpath_on_record(what, xpath):
    """Evaluate *xpath* against dict *what*; return matched element paths (without '/record').

    A record that cannot be rendered (a key that is not a usable element name) and an XPath that
    does not compile both raise ``ValueError``; an XPath that matches nothing returns ``[]``.
    """
    xml = dicttoxml(what)
    try:
        dom = etree.XML(xml)
    except etree.XMLSyntaxError as e:
        raise ValueError(f"the record cannot be addressed by an XPath: {e}") from e
    tree = dom.getroottree()
    try:
        matches = dom.xpath("/record" + xpath, namespaces={})
    except etree.XPathError as e:
        raise ValueError(f"the XPath {xpath!r} is not valid: {e}") from e
    paths = []
    for m in matches:
        paths.append(tree.getpath(m).replace("/record", ""))
    return paths


def extract_xpath(what, xpath):
    """Return the value in dict *what* addressed by *xpath* (first match only), ``None`` for no match."""
    paths = xpath_on_record(what, xpath)
    for p in paths:
        bits = p[1:].split("/")
        path = []
        for bit in bits:
            if bit[-1] == "]":
                sqidx = bit.find("[")
                # indexes are 1 based, not 0 based
                idx = int(bit[sqidx + 1 : -1]) - 1
                key = bit[:sqidx]
            else:
                key = bit
                idx = 0
            path.append((key, idx))

        tgt = what
        for tag, idx in path[:-1]:
            if tag in tgt:
                tgt = tgt[tag]
            if type(tgt) is list:
                tgt = tgt[idx]

        tag, idx = path[-1]
        return tgt[tag]
