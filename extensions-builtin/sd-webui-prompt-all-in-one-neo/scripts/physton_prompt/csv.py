import os
from pathlib import Path

base_dir = str(Path().absolute())
self_base_dir = os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__)), '../', '../'))
self_tags_dir = os.path.join(self_base_dir, 'tags')
dirs = [
    self_tags_dir,
    # os.path.join(base_dir, 'extensions', 'sd-webui-prompt-all-in-one', 'tags'),
    os.path.join(base_dir, 'extensions', 'a1111-sd-webui-tagcomplete', 'tags'),
]


def get_csvs():
    global base_dir
    global self_base_dir
    global self_tags_dir
    csvs = []
    for dir in dirs:
        if not os.path.exists(dir):
            continue
        for file in os.listdir(dir):
            if file.endswith('.csv'):
                path = os.path.join(dir, file)
                name = os.path.basename(file)
                size = os.path.getsize(path)
                if dir == self_tags_dir:
                    # 去除 self_tags_dir 后的路径
                    key = path.replace(self_tags_dir, '')
                    key = '\\extensions\\sd-webui-prompt-all-in-one\\tags\\' + name
                else:
                    # 去除 base_dir 后的路径
                    key = path.replace(base_dir, '')
                csvs.append({
                    'key': key,
                    'name': name,
                    'size': size,
                    'path': path
                })
    return csvs


def get_csv(key):
    # ``key`` comes from an HTTP query parameter. Resolve it through the
    # server-generated inventory instead of treating it as a filesystem path.
    # This preserves all keys emitted by get_csvs() while preventing absolute
    # paths, ``..`` traversal, and symlink escapes from exposing host files.
    allowed_roots = [os.path.realpath(directory) for directory in dirs]
    for item in get_csvs():
        if item['key'] != key:
            continue

        path = os.path.realpath(item['path'])
        try:
            inside_allowed_root = any(
                os.path.commonpath((root, path)) == root
                for root in allowed_roots
            )
        except ValueError:
            inside_allowed_root = False

        if inside_allowed_root and path.lower().endswith('.csv') and os.path.isfile(path):
            return path
        return None
    return None
