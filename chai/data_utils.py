from lxml import etree


def _escape_xml(s):
    s = s.replace("&", "&amp;").replace('"', "&quot;").replace("'", "&apos;")
    s = s.replace("<", "&lt;").replace(">", "&gt;")
    return s


# Doesn't need to preserve data type as we never reverse the operation
def _convert(what, output):
    if type(what) is dict:
        for k, v in what.items():
            tag = k.replace("@", "__")
            if type(v) is not list:
                v = [v]
            for item in v:
                output.append(f"<{tag}>")
                _convert(item, output)
                output.append(f"</{tag}>")
    elif isinstance(what, str):
        output.append(_escape_xml(what))
    elif type(what) in [int, float]:
        output.append(str(what))
    elif what is None:
        output.append("")
    else:
        print(f"Unknown (to model) type: {type(what)} from {what}")


def dicttoxml(what):
    output = ["<record>"]
    _convert(what, output)
    output.append("</record>")
    xml = "".join(output)
    return xml


def xpath_on_record(what, xpath):
    xml = dicttoxml(what)
    try:
        dom = etree.XML(xml)
    except Exception:
        print(f"failed to parse XML:\n{xml}")
        return []
    tree = dom.getroottree()
    try:
        matches = dom.xpath("/record" + xpath, namespaces={})
    except Exception:
        print(f"Failed to compile xpath: {xpath}")
        return []
    paths = []
    for m in matches:
        paths.append(tree.getpath(m).replace("/record", ""))
    return paths


def extract_xpath(what, xpath):
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
