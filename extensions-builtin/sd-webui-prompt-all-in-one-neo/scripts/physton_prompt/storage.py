import os
import json
import shutil
import threading
from contextlib import contextmanager

from modules import paths_internal


# One in-process lock replaces the old .lock files: a lock file leaked by an
# exception made every later request spin forever (on the event loop).
_STORAGE_LOCK = threading.RLock()


class Storage:
    storage_path = ''
    _storage_ready = False

    def __init__():
        Storage.__dispose_all_locks()

    def __get_storage_path():
        # Resolve, create and migrate once; this runs for every storage call
        # (including each 10 ms lock-wait spin).
        if Storage._storage_ready:
            return Storage.storage_path
        Storage.storage_path = os.path.join(
            paths_internal.data_path,
            'extension-data',
            'sd-webui-prompt-all-in-one-neo',
            'storage',
        )
        if not os.path.exists(Storage.storage_path):
            os.makedirs(Storage.storage_path)

        legacy_storage_path = os.path.normpath(
            os.path.dirname(os.path.abspath(__file__)) + '/../../storage'
        )
        if os.path.isdir(legacy_storage_path):
            for filename in os.listdir(legacy_storage_path):
                if not filename.endswith('.json'):
                    continue
                old_file_path = os.path.join(legacy_storage_path, filename)
                new_file_path = os.path.join(Storage.storage_path, filename)
                if not os.path.exists(new_file_path):
                    try:
                        shutil.copy2(old_file_path, new_file_path)
                        os.chmod(new_file_path, 0o600)
                    except OSError as e:
                        print(f"Prompt All-in-One storage migration failed: {e}")

        Storage._storage_ready = True
        Storage.__dispose_all_locks()
        return Storage.storage_path

    def __key_path(key, suffix):
        """Path for a storage key; keys come from HTTP and must stay inside storage."""
        key = str(key)
        if (
            not key
            or key.startswith('.')
            or any(char in key for char in ('/', '\\', ':', '\0'))
            or '..' in key
        ):
            raise ValueError(f"Invalid Prompt All-in-One storage key: {key!r}")
        directory = os.path.realpath(Storage.__get_storage_path())
        path = os.path.realpath(os.path.join(directory, key + suffix))
        if os.path.dirname(path) != directory:
            raise ValueError(f"Invalid Prompt All-in-One storage key: {key!r}")
        return path

    def __get_data_filename(key):
        return Storage.__key_path(key, '.json')


    def __dispose_all_locks():
        directory = Storage.__get_storage_path()
        for filename in os.listdir(directory):
            # 检查文件是否以指定后缀结尾
            if filename.endswith('.lock'):
                file_path = os.path.join(directory, filename)
                try:
                    os.remove(file_path)
                    print(f"Disposed lock: {file_path}")
                except Exception as e:
                    print(f"Dispose lock {file_path} failed: {e}")

    @contextmanager
    def __locked(key):
        with _STORAGE_LOCK:
            yield

    def __get(key):
        filename = Storage.__get_data_filename(key)
        if not os.path.exists(filename):
            return None
        if os.path.getsize(filename) == 0:
            return None
        try:
            import launch
            if not launch.is_installed("chardet"):
                with open(filename, 'r') as f:
                    data = json.load(f)
            else:
                import chardet
                with open(filename, 'rb') as f:
                    data = f.read()
                    encoding = chardet.detect(data).get('encoding')
                    data = json.loads(data.decode(encoding))
        except Exception as e:
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)
            except Exception as e:
                print(e)
                return None
        return data

    def __set(key, data):
        file_path = Storage.__get_data_filename(key)
        temp_path = file_path + '.tmp'
        opener = lambda path, flags: os.open(path, flags, 0o600)
        with open(temp_path, 'w', opener=opener) as f:
            json.dump(data, f, indent=4, ensure_ascii=True)
        os.chmod(temp_path, 0o600)
        os.replace(temp_path, file_path)

    def set(key, data):
        with Storage.__locked(key):
            Storage.__set(key, data)

    def get(key):
        return Storage.__get(key)

    def delete(key):
        file_path = Storage.__get_data_filename(key)
        if os.path.exists(file_path):
            os.remove(file_path)

    def __get_list(key):
        data = Storage.get(key)
        if not data:
            data = []
        return data

    # 向列表中添加元素
    def list_push(key, item):
        with Storage.__locked(key):
            data = Storage.__get_list(key)
            data.append(item)
            Storage.__set(key, data)

    # 从列表中删除和返回最后一个元素
    def list_pop(key):
        with Storage.__locked(key):
            data = Storage.__get_list(key)
            item = data.pop()
            Storage.__set(key, data)
            return item

    # 从列表中删除和返回第一个元素
    def list_shift(key):
        with Storage.__locked(key):
            data = Storage.__get_list(key)
            item = data.pop(0)
            Storage.__set(key, data)
            return item

    # 从列表中删除指定元素
    def list_remove(key, index):
        with Storage.__locked(key):
            data = Storage.__get_list(key)
            data.pop(index)
            Storage.__set(key, data)

    # 获取列表中指定位置的元素
    def list_get(key, index):
        data = Storage.__get_list(key)
        return data[index]

    # 清空列表中的所有元素
    def list_clear(key):
        with Storage.__locked(key):
            Storage.__set(key, [])
