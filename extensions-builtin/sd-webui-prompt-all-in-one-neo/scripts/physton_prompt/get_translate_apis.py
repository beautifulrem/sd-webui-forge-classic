import os
import json
import re
from scripts.physton_prompt.storage import Storage

# from scripts.physton_prompt.storage import Storage

translate_apis = {}


def get_translate_apis(reload=False):
    global translate_apis
    if reload or not translate_apis:
        translate_apis = {}
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(current_dir, '../../translate_apis.json')
        config_file = os.path.normpath(config_file)
        with open(config_file, 'r', encoding='utf8') as f:
            translate_apis = json.load(f)

        # for group in translate_apis['apis']:
        #     for item in group['children']:
        #         if 'config' not in item:
        #             continue
        #         config_name = 'translate_api.' + item['key']
        #         config = Storage.get(config_name)
        #         if not config:
        #             config = {}
        #         for config_item in item['config']:
        #             if config_item['key'] in config:
        #                 config_item['value'] = config[config_item['key']]
        #             else:
        #                 if 'default' in config_item:
        #                     config_item['value'] = config_item['default']
        #                 else:
        #                     config_item['value'] = ''

    return translate_apis


# Config fields that decide where requests (and therefore secrets) are sent.
_ENDPOINT_CONFIG_KEYS = ('api_base', 'host')
# Values an unset endpoint field falls back to (see gen_openai).
_ENDPOINT_DEFAULTS = {'api_base': 'https://api.openai.com/v1'}


def _endpoint_value(config, field):
    value = str((config or {}).get(field) or _ENDPOINT_DEFAULTS.get(field, '')).strip()
    return value.rstrip('/').casefold()


def _secret_api_for_key(data_key):
    """API name whose secrets a storage key holds (case-insensitive, since the
    same file is reachable with any casing on case-insensitive filesystems)."""
    key = str(data_key).casefold()
    if key == 'chatgpt_key':
        return 'openai'
    start = 'translate_api.'
    if key.startswith(start):
        return key[len(start):]
    return None


def privacy_translate_api_config(data_key, data):
    # 如果 data 为空或者不是 dict
    if not data or not isinstance(data, dict):
        return data
    # 如果 data_key 是 translate_api. 开头
    api = _secret_api_for_key(data_key)
    if api is None:
        return data
    apis = get_translate_apis()
    find = False
    for group in apis['apis']:
        for item in group['children']:
            if str(item['key']).casefold() == api:
                find = item
                break
    if not find:
        return data
    api_item = find
    if 'config' not in api_item or not api_item['config']:
        return data

    for config in api_item['config']:
        # 如果有 privacy 的属性并且为 True
        if 'privacy' in config and config['privacy'] and config['type'] == 'input':
            if config['key'] in data:
                # 前面6个字符可见，后面的字符用 * 替换
                value = data[config['key']]
                if len(value) > 6:
                    value = value[:6] + '*' * (len(value) - 6)
                data[config['key']] = value

    return data

def unprotected_translate_api_config(data_key, data):
    api = _secret_api_for_key(data_key)
    if api is None:
        return data

    apis = get_translate_apis()
    find = False
    for group in apis['apis']:
        for item in group['children']:
            if str(item['key']).casefold() == api:
                find = item
                break
    if not find:
        return data
    api_item = find
    if 'config' not in api_item or not api_item['config']:
        return data

    storage_data = Storage.get(data_key)
    # Restoring a masked secret is only allowed for the endpoint it was saved
    # with; otherwise a caller could send it to a URL of their choosing.
    endpoint_changed = bool(storage_data) and any(
        _endpoint_value(data, field) != _endpoint_value(storage_data, field)
        for field in _ENDPOINT_CONFIG_KEYS
        if field in data
    )

    for config in api_item['config']:
        # 如果有 privacy 的属性并且为 True
        if 'privacy' in config and config['privacy'] and config['type'] == 'input':
            if storage_data and config['key'] in storage_data:
                if config['key'] in data:
                    value = data[config['key']]
                    # 如果包含 * 号，并且前面6个字符等于 storage_data 的前面6个字符
                    if '*' in value and value[:6] == storage_data[config['key']][:6]:
                        data[config['key']] = '' if endpoint_changed else storage_data[config['key']]

                    # 多个 * 替换成一个 *
                    # value = re.sub(r'\*+', '*', value)
                    # if value == '*' and storage_data and config['key'] in storage_data:
                        # value = storage_data[config['key']]
                        # data[config['key']] = value

    return data