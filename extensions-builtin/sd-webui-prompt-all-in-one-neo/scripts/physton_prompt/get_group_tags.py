import os
import re

current_dir = os.path.dirname(os.path.abspath(__file__))
group_tags_dir = os.path.realpath(os.path.join(current_dir, '../../group_tags'))

def _get_tags_filename(name):
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', name):
        return None
    file = os.path.realpath(os.path.join(group_tags_dir, name + '.yaml'))
    try:
        if os.path.commonpath((group_tags_dir, file)) != group_tags_dir:
            return None
    except ValueError:
        return None
    return file

def get_group_tags(lang):
    tags_file = _get_tags_filename('custom')
    is_exists = os.path.exists(tags_file)
    if is_exists:
        try:
            with open(tags_file, 'r', encoding='utf8') as f:
                data = f.read()
            is_exists = len(data.strip()) > 0
        except:
            is_exists = False

    if not is_exists:
        tags_file = _get_tags_filename(lang)
        if not tags_file or not os.path.exists(tags_file):
            tags_file = _get_tags_filename('default')
    if not tags_file or not os.path.exists(tags_file):
        return ''

    tags = ''

    try:
        prepend_file = _get_tags_filename('prepend')
        with open(prepend_file, 'r', encoding='utf8') as f:
            prepend = f.read()
        tags += prepend + "\n\n"
    except:
        pass

    try:
        with open(tags_file, 'r', encoding='utf8') as f:
            data = f.read()
        tags += data + "\n\n"
    except:
        pass

    try:
        append_file = _get_tags_filename('append')
        with open(append_file, 'r', encoding='utf8') as f:
            append = f.read()
        tags += append + "\n\n"
    except:
        pass

    return tags
