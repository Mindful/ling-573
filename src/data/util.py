
def get_child(parent, child_tag):
    try:
        return next(child for child in parent if child.tag == child_tag)
    except StopIteration:
        return None


def get_child_text(parent, child_tag):
    child = get_child(parent, child_tag)
    return child.text.strip() if child is not None else None
